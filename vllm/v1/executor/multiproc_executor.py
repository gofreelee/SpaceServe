# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import signal
import sys
import time
import weakref
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing.process import BaseProcess
import torch.multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
import psutil
import zmq

from vllm.config import VllmConfig
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.distributed.device_communicators.shm_broadcast import (Handle,
                                                                 MessageQueue)
from vllm.executor.multiproc_worker_utils import (
    _add_prefix, set_multiprocessing_worker_envs)
from vllm.logger import init_logger
from vllm.utils import (get_distributed_init_method, get_mp_context,
                        get_open_port, get_open_zmq_ipc_path, zmq_socket_ctx)
from vllm.v1.executor.abstract import Executor
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 5000
POLLING_TIMEOUT_S = POLLING_TIMEOUT_MS // 1000


class MultiprocExecutor(Executor):

    def _init_executor(self, encoder_cache = None) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        logger.info(f"in multiproc executor encoder_cache is {encoder_cache}, id is {id(encoder_cache)}")
        self.shutting_down = False
        self.encoder_cache = encoder_cache
        self.encoder_req_ids = []
        self._finalizer = weakref.finalize(self, self.shutdown)

        # The child processes will send SIGUSR1 when unrecoverable
        # errors happen.
        def sigusr1_handler(signum, frame):
            logger.fatal(
                "MulitprocExecutor got fatal signal from worker processes, "
                "shutting down. See stack trace above for root cause issue.")
            # Propagate error up to parent process.
            parent_process = psutil.Process().parent()
            parent_process.send_signal(signal.SIGUSR1)
            self.shutdown()

        signal.signal(signal.SIGUSR1, sigusr1_handler)

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        assert self.world_size == tensor_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}). "
            f"Pipeline parallelism is not yet implemented in v1")

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        self.rpc_broadcast_mq = MessageQueue(self.world_size, self.world_size)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        self.workers: List[WorkerProcHandle] = []
        self.workers_encoder_results_queue = []
        for rank in range(self.world_size):
            self.workers_encoder_results_queue.append(mp.Queue(10))
            worker = WorkerProc.make_worker_process(self.vllm_config, rank,
                                                    rank,
                                                    distributed_init_method,
                                                    scheduler_output_handle,
                                                    self.workers_encoder_results_queue[-1],)
            self.workers.append(worker)

        # Ensure message queues are ready. Will deadlock if re-ordered
        # Must be kept consistent with the WorkerProc
        #logger.info(f"encoder cache  is {self.encoder_cache}, id is  {id(self.encoder_cache)}")
        self.rpc_broadcast_mq.wait_until_ready()
        for w in self.workers:
            w.worker_response_mq.wait_until_ready()


    def dispatch_encoder_result_to_workers(self, encoder_result):
        s_time = time.time()
        for encoder_res_item in encoder_result:
            for req_id in encoder_res_item.keys():
                for mm_key in encoder_res_item[req_id].keys():
                    if encoder_res_item[req_id][mm_key] is not None:
                        encoder_res_item[req_id][mm_key] = encoder_res_item[req_id][mm_key].cpu()
                logger.info(f"req id's mm done {req_id}")
                self.encoder_req_ids.append(req_id)


        for encoder_result_queue in self.workers_encoder_results_queue:
            #logger.info(f"put the result {encoder_result}")
            encoder_result_queue.put(encoder_result)

            while encoder_result_queue.empty():
                continue

            logger.info(f"queue size {encoder_result_queue.qsize()}")
            #logger.info(f"dispatch to queue : id is {id(encoder_result_queue)}, ifempty is {encoder_result_queue.empty()}")
        e_time = time.time()
        #logger.info(f"dispatch encoder result to workers time is {1000 * (e_time - s_time)} ms")

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None) -> List[Any]:
        start_time = time.monotonic()
        kwargs = kwargs or {}
        # logger.info(f"collective rpc method is {method}")
        # logger.info(f"in multiproc executor collective rpc is {self.encoder_cache}, id is {id(self.encoder_cache)}")
        if method == "execute_model":
            kwargs["encoder_req"] = self.encoder_req_ids
        # NOTE: If the args are heterogeneous, then we pack them into a list,
        # and unpack them in the method of every worker, because every worker
        # knows their own rank.
        try:
            if isinstance(method, str):
                send_method = method
            else:
                send_method = cloudpickle.dumps(
                    method, protocol=pickle.HIGHEST_PROTOCOL)
            self.rpc_broadcast_mq.enqueue((send_method, args, kwargs))

            responses = [None] * self.world_size
            for w in self.workers:
                dequeue_timeout = timeout - (time.monotonic() - start_time
                                             ) if timeout is not None else None
                status, result = w.worker_response_mq.dequeue(
                    timeout=dequeue_timeout)

                if status != WorkerProc.ResponseStatus.SUCCESS:
                    if isinstance(result, Exception):
                        raise result
                    else:
                        raise RuntimeError("Worker failed")

                responses[w.rank] = result

            if method == "execute_model":
                self.encoder_req_ids = []
            return responses
        except TimeoutError as e:
            raise TimeoutError(f"RPC call to {method} timed out.") from e
        except Exception as e:
            # Re-raise any other exceptions
            raise e
        

    def _ensure_worker_termination(self):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        def wait_for_termination(procs, timeout):
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # Send SIGTERM if still running
        active_procs = [w.proc for w in self.workers if w.proc.is_alive()]
        for p in active_procs:
            p.terminate()
        if not wait_for_termination(active_procs, 4):
            # Send SIGKILL if still running
            active_procs = [p for p in active_procs if p.is_alive()]
            for p in active_procs:
                p.kill()

        self._cleanup_sockets()

    def _cleanup_sockets(self):
        for w in self.workers:
            # Remove the zmq ipc socket file
            socket_path = w.ready_path.replace("ipc://", "")
            if os and os.path.exists(socket_path):
                os.remove(socket_path)

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if getattr(self, 'shutting_down', False):
            self.shutting_down = True
            for w in self.workers:
                w.worker_response_mq = None
            self._ensure_worker_termination()

        self.rpc_broadcast_mq = None

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    ready_path: str
    worker_response_mq: MessageQueue  # The worker process writes to this MQ


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        ready_path: str,
        encoder_result_queue: Optional[mp.Queue] = None,
    ):
        self.rank = rank
        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
        # TODO: move `init_worker` to executor level as a collective rpc call
        all_kwargs: List[Dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper.worker
        self.rank  = rank
        pid = os.getpid()
        _add_prefix(sys.stdout, f"VllmWorker rank={rank}", pid)
        _add_prefix(sys.stderr, f"VllmWorker rank={rank}", pid)

        # Initialize MessageQueue for receiving SchedulerOutput
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank)

        # Initializes a message queue for sending the model output
        self.worker_response_mq = MessageQueue(1, 1)
        worker_response_mq_handle = self.worker_response_mq.export_handle()

        # Send Readiness signal to EngineCore process.
        with zmq_socket_ctx(ready_path, zmq.constants.PUSH) as ready_socket:
            payload = pickle.dumps(worker_response_mq_handle,
                                   protocol=pickle.HIGHEST_PROTOCOL)
            ready_socket.send_string(WorkerProc.READY_STR)
            ready_socket.send(payload)

        self.encoder_cache = {}
        self.encoder_result_queue = encoder_result_queue
    
        self.worker.init_device()
        self.worker.load_model()
        self.worker.add_encoder_cache_to_modelrunner(self.encoder_cache)     

    @staticmethod
    def make_worker_process(
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            input_shm_handle,  # Receive SchedulerOutput
            encoder_result_queue = None,  # Send ModelRunnerOutput
    ) -> WorkerProcHandle:
        context = get_mp_context()

        # ZMQ path for worker to send ready message and shm_broadcast handle
        # back to core process.
        ready_path = get_open_zmq_ipc_path()

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_path": ready_path,
            "encoder_result_queue": encoder_result_queue,
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=WorkerProc.worker_main,
                               kwargs=process_kwargs,
                               daemon=True)
        proc.start()

        # Wait for startup
        worker_response_mq_handle = WorkerProc.wait_for_startup(
            proc, ready_path)

        worker_response_mq = MessageQueue.create_from_handle(
            worker_response_mq_handle, 0)

        return WorkerProcHandle(proc, rank, ready_path, worker_response_mq)

    def shutdown(self):
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def worker_main(*args, **kwargs):
        """ Worker initialization and execution loops.
        This runs a background process """

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        try:
            worker = WorkerProc(*args, **kwargs)

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()

            worker.worker_busy_loop()

        except SystemExit:
            logger.debug("Worker interrupted.")

        except Exception:
            # worker_busy_loop sends exceptions exceptons to Executor
            # for shutdown, but if there is an error in startup or an
            # error with IPC itself, we need to alert the parent.
            psutil.Process().parent().send_signal(signal.SIGUSR1)
            raise

        finally:
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()
                worker = None

    @staticmethod
    def wait_for_startup(
        proc: BaseProcess,
        ready_path: str,
    ) -> Optional[Handle]:
        """Wait until the Worker is ready."""
        with zmq_socket_ctx(ready_path, zmq.constants.PULL) as socket:

            # Wait for Worker to send READY.
            while socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                logger.debug("Waiting for WorkerProc to startup.")

                if not proc.is_alive():
                    raise RuntimeError("WorkerProc failed to start.")

            message = socket.recv_string()
            assert message == WorkerProc.READY_STR
            handle_frame = socket.recv(copy=False)
            handle = pickle.loads(handle_frame.buffer)
            return handle

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        while True:
            s_time = time.time()
            method, args, kwargs = self.rpc_broadcast_mq.dequeue()
            e_time = time.time()
            logger.info(f"worker {self.rank}  dequeue execute time is {1000 * (e_time - s_time)} ms")
            # logger.info(f"encoder result queue in worker is {id(self.encoder_result_queue)}")
            #logger.info(f"WorkerProc got method {method} with args {args} and kwargs {kwargs}")
            # you need to transfer the encoder cache to the worker cross GPU,  noted by lizhicheng
            # req_id_scheduled = []
            # if method == "execute_model":  
            #     logger.info(f"args type is {type(args)}")
            #     for arg in args:
            #         logger.info(f"arg type is {type(arg)}")
            #         logger.info(f"{arg.scheduled_new_reqs}")
            #         for req in arg.scheduled_new_reqs:
            #             logger.info(f"req is {req}")
            #             req_id_scheduled.append(req.req_id)
                
            #     logger.info(f"req_id_scheduled is {req_id_scheduled}")
            #     for req_id in req_id_scheduled:
            #         if req_id not in self.encoder_cache.keys():
            #             #todo 
            #             while req_id not in self.encoder_cache.keys():
            #                 logger.info(f"worker's encoderqueue is {id(self.encoder_result_queue)}")
            #                 while self.encoder_result_queue is not None and self.encoder_result_queue.empty():
            #                     continue
            #                 encoder_result = self.encoder_result_queue.get_nowait()
            #                 logger.info(f"worker get encoder result is {encoder_result}")
            #                 for encoder_item in encoder_result:
            #                     for req_id in encoder_item.keys():
            #                         if req_id in self.encoder_cache.keys():
            #                             continue
            #                         self.encoder_cache[req_id] = {}
            #                         for mm_key in encoder_item[req_id].keys():
            #                             self.encoder_cache[req_id][mm_key] = encoder_item[req_id][mm_key].cuda(self.rank)

            #add to then worker encoder cache
            if method == "execute_model":
                #logger.info(f"encoder result queue id is {id(self.encoder_result_queue)}")
                while self.encoder_result_queue is not None and not self.encoder_result_queue.empty():
                    s_time = time.time()
                    encoder_result = self.encoder_result_queue.get_nowait()
                    #logger.info(f"worker get encoder result is {encoder_result}, encoder result queue ifempty is {self.encoder_result_queue.empty()}")  
                    for encoder_item in encoder_result:
                        for req_id in encoder_item.keys():
                            if req_id in self.encoder_cache.keys():
                                continue
                            logger.info(f"worker doing req id is {req_id}")
                            self.encoder_cache[req_id] = {}
                            for mm_key in encoder_item[req_id].keys():
                                self.encoder_cache[req_id][mm_key] = encoder_item[req_id][mm_key].cuda(self.rank)

                                #tmp = encoder_item[req_id][mm_key].cuda(self.rank)
                    e_time = time.time() 
                
                if kwargs is not None and 'encoder_req' in kwargs.keys(): 
                    req_ids = kwargs['encoder_req']
                    for req_id in req_ids:
                        if req_id in self.encoder_cache.keys():
                            continue
                        while self.encoder_result_queue is not None and self.encoder_result_queue.empty():
                            continue

                        encoder_result = self.encoder_result_queue.get_nowait()
                        for encoder_item in encoder_result:
                            for req_id in encoder_item.keys():
                                if req_id in self.encoder_cache.keys():
                                    continue
                                self.encoder_cache[req_id] = {}
                                for mm_key in encoder_item[req_id].keys():
                                    self.encoder_cache[req_id][mm_key] = encoder_item[req_id][mm_key].cuda(self.rank)

                    kwargs.pop('encoder_req')
                    #logger.info(f"worker get encoder result time is {1000 * (e_time - s_time)} ms") 
                    
                # for req_id in kwargs['encoder_cache'].keys(): 
                #     #logger.info(f"current gpu is {self.rank}, the encoder cache in {kwargs['encoder_cache'][req_id]}")
                #     if req_id in self.encoder_cache.keys():
                #         continue
                #     self.encoder_cache[req_id] = kwargs['encoder_cache'][req_id]
                #     for mm_item in self.encoder_cache[req_id].keys():
                #         s_time = time.time() 
                #         self.encoder_cache[req_id][mm_item] = self.encoder_cache[req_id][mm_item].cuda(self.rank)
                #         e_time = time.time()
                #         logger.info(f"encoder cache transfer time is {1000 * (e_time - s_time)} ms")
                
                #kwargs.pop('encoder_cache') 
                # if req_id in kwargs['encoder_cache'].keys():
                #     if req_id not in self.encoder_cache.keys(): 
                #         self.encoder_cache[req_id] = kwargs['encoder_cache'][req_id]

            # logger.info(f"worker encoder cache  is {self.encoder_cache}")
            # logger.info(f"kwargs is {kwargs}")  

            
            # if 'encoder_cache' in kwargs.keys():
            #     encoder_cache = kwargs.pop('encoder_cache')
            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)
                output = func(*args, **kwargs)
            except Exception as e:
                self.worker_response_mq.enqueue(
                    (WorkerProc.ResponseStatus.FAILURE, e))
                logger.exception("WorkerProc hit an exception: %s", exc_info=e)
                continue

            self.worker_response_mq.enqueue(
                (WorkerProc.ResponseStatus.SUCCESS, output))
