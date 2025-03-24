# SPDX-License-Identifier: Apache-2.0
import sys
import pickle
import queue
import signal
import threading
import time
from multiprocessing.connection import Connection
from typing import List, Tuple, Type
import torch

import psutil
import zmq
import zmq.asyncio
from msgspec import msgpack

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.utils import get_exception_traceback, zmq_socket_ctx
from vllm.v1.core.kv_cache_utils import get_kv_cache_config
from vllm.v1.core.scheduler import Scheduler
from vllm.v1.core.encoder_scheduler import EncoderScheduler
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreProfile,
                            EngineCoreRequest, EngineCoreRequestType,
                            EngineCoreRequestUnion, EngineCoreResetPrefixCache)
from vllm.v1.engine.mm_input_mapper import MMInputMapperServer
from vllm.v1.executor.abstract import Executor
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import PickleEncoder
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
# 在EngineCoreProc类中添加以下代码
# import sys
# _gil_tracker = {}

# def gil_trace(frame, event, arg):
#     if event == 'call' and frame.f_code.co_name == 'PyEval_RestoreThread':
#         ident = threading.get_ident()
#         _gil_tracker[ident] = time.monotonic()
#     elif event == 'return' and frame.f_code.co_name == 'PyEval_SaveThread':
#         ident = threading.get_ident()
#         start = _gil_tracker.pop(ident, None)
#         if start:
#             duration = time.monotonic() - start
#             if duration > 0.1:
#                 print(f"Thread {ident} held GIL for {duration:.2f}s")

# sys.settrace(gil_trace)


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        encoder_result_queue = None 
    ):
        assert vllm_config.model_config.runner_type != "pooling"

        logger.info("Initializing a V1 LLM engine (v%s) with config: %s",
                    VLLM_VERSION, vllm_config)

        # Setup Model.
        logger.info(vllm_config)
        logger.info(executor_class)
        self.encoder_result_cache = {}
        self.encoder_result_queue = encoder_result_queue
        self.model_executor = executor_class(vllm_config, self.encoder_result_cache)
        logger.info(type(self.model_executor))
        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks = self._initialize_kv_caches(
            vllm_config)
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks

        # Setup scheduler.
        self.scheduler = Scheduler(
            scheduler_config=vllm_config.scheduler_config,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            lora_config=vllm_config.lora_config,
            encoder_cache = self.encoder_result_cache
        )


        self.mm_input_mapper_server = MMInputMapperServer(
            vllm_config.model_config)

    def _initialize_kv_caches(self,
                              vllm_config: VllmConfig) -> Tuple[int, int]:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_spec = self.model_executor.get_kv_cache_spec()

        # Profiles the peak memory usage of the model to determine how much
        # memory can be allocated for kv cache.
        availble_gpu_memory = self.model_executor.determine_available_memory()

        # Get the kv cache tensor size
        kv_cache_config = get_kv_cache_config(vllm_config, kv_cache_spec,
                                              availble_gpu_memory)
        num_gpu_blocks = kv_cache_config.num_blocks
        num_cpu_blocks = 0

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize(kv_cache_config)

        elapsed = time.time() - start
        logger.info(("init engine (profile, create kv cache, "
                     "warmup model) took %.2f seconds"), elapsed)
        return num_gpu_blocks, num_cpu_blocks

    def add_request(self, request: EngineCoreRequest):
        """Add request to the scheduler."""

        if request.mm_hashes is not None:
            # Here, if hash exists for an image, then it will be fetched
            # from the cache, else it will be added to the cache.
            # Note that the cache here is mirrored with the client side of the
            # MM mapper, so anything that has a hash must have a HIT cache
            # entry here as well.
            assert request.mm_inputs is not None
            request.mm_inputs = self.mm_input_mapper_server.process_inputs(
                request.mm_inputs, request.mm_hashes)

        req = Request.from_engine_core_request(request)
        #print(f"req.mm_inputs is {req.mm_inputs}")

        self.scheduler.add_request(req)

    def abort_requests(self, request_ids: List[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids,
                                       RequestStatus.FINISHED_ABORTED)
    #add by lzc
    def _handle_encoder_result(self, encoder_result):
        '''Handle encoder result from encoder process'''
        for item in encoder_result:
            #logger.info(type(item))
            #item is dict type
            for k, v in item.items():
                for v_k, v_v in v.items():
                    if k in self.encoder_result_cache:
                        self.encoder_result_cache[k][v_k] = v_v
                    else:
                        self.encoder_result_cache[k] = {v_k: v_v}


    def step(self) -> EngineCoreOutputs:
        """Schedule, execute, and make output."""

        if not self.scheduler.has_unfinished_requests():
            return EngineCoreOutputs(
                outputs=[], scheduler_stats=self.scheduler.make_stats())

        # find if have multimodal inputs, send it to do encoder and continue the steping, noted by lizhicheng
        # try:
        #     while self.encoder_result_queue.empty():
        #         pass
        #     import time;s_time = time.time()
        #     encoder_result_from_encoder_proc = self.encoder_result_queue.get_nowait()
        #     e_time = time.time()
        #     logger.info(f"encoder_result_from_encoder_proc is {encoder_result_from_encoder_proc}")
        #     logger.info(f"get encoder_result_from_encoder_proc time is {1000 * (e_time - s_time)}")
        # except queue.Empty:
        #     encoder_result_from_encoder_proc = None
        scheduler_output = self.scheduler.schedule()
        while scheduler_output.total_num_scheduled_tokens == 0:
            scheduler_output = self.scheduler.schedule()
            #logger.info("in EngineCore step, scheduler_output is 0")
            if not self.encoder_result_queue.empty():
                encoder_result = self.encoder_result_queue.get_nowait()
                self._handle_encoder_result(encoder_result)
        #logger.info(f"schedule the total number tokens are {scheduler_output.total_num_scheduled_tokens}")
        #logger.info(f"scheduler_output is {scheduler_output}")
        output = self.model_executor.execute_model(scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, output)
        return engine_core_outputs

    def shutdown(self):
        self.model_executor.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def reset_prefix_cache(self):
        self.scheduler.reset_prefix_cache()


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        ready_pipe: Connection,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        log_stats: bool = False,
        encoder_result_queue = None,
    ):
        super().__init__(vllm_config, executor_class, encoder_result_queue)

        self.log_stats = log_stats

        # Background Threads and Queues for IO. These enable us to
        # overlap ZMQ socket IO with GPU since they release the GIL,
        # and to overlap some serialization/deserialization with the
        # model forward pass.
        # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
        self.input_queue: queue.Queue[EngineCoreRequestUnion] = queue.Queue()
        self.output_queue: queue.Queue[EngineCoreOutputs] = queue.Queue()
        #self.encoder_result_queue = encoder_result_queue
        threading.Thread(target=self.process_input_socket,
                         args=(input_path, ),
                         daemon=True).start()
        threading.Thread(target=self.process_output_socket,
                         args=(output_path, ),
                         daemon=True).start()
        

        # Send Readiness signal to EngineClient.
        ready_pipe.send({"status": "READY"})

    @staticmethod
    def run_engine_core(*args, **kwargs):
        # import traceback;traceback.print_stack()    
        # exit(0)
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        parent_process = psutil.Process().parent()
        engine_core = None
        try:
            engine_core = EngineCoreProc(*args, **kwargs)
            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore interrupted.")

        except Exception:
            traceback = get_exception_traceback()
            logger.error("EngineCore hit an exception: %s", traceback)
            parent_process.send_signal(signal.SIGUSR1)

        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        
        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            if not self.scheduler.has_unfinished_requests():
                while True:
                    try:
                        req = self.input_queue.get(timeout=POLLING_TIMEOUT_S)
                        self._handle_client_request(req)
                        break
                    except queue.Empty:
                        logger.debug("EngineCore busy loop waiting.")
                        # Break out the loop so we can log_stats in step().
                        if self.log_stats:
                            break
                    except BaseException:
                        raise

            # 2) Handle any new client requests (Abort or Add).
            while not self.input_queue.empty():
                req = self.input_queue.get_nowait()
                self._handle_client_request(req)
            
            while not self.encoder_result_queue.empty():
                encoder_result = self.encoder_result_queue.get_nowait()
                self._handle_encoder_result(encoder_result)

            # 3) Step the engine core.
            outputs = self.step()

            # 5) Put EngineCoreOutputs into the output queue.
            self.output_queue.put_nowait(outputs)

    def _handle_client_request(self, request: EngineCoreRequestUnion) -> None:
        """Handle EngineCoreRequest or EngineCoreABORT from Client."""

        if isinstance(request, EngineCoreRequest):
            self.add_request(request)
        elif isinstance(request, EngineCoreProfile):
            self.model_executor.profile(request.is_start)
        elif isinstance(request, EngineCoreResetPrefixCache):
            self.reset_prefix_cache()
        else:
            # TODO: make an EngineCoreAbort wrapper
            assert isinstance(request, list)
            self.abort_requests(request)
    
    def process_input_socket(self, input_path: str):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        decoder_add_req = PickleEncoder()
        decoder_abort_req = PickleEncoder()

        with zmq_socket_ctx(input_path, zmq.constants.PULL) as socket:
            while True:
                # (RequestType, RequestData)
                type_frame, data_frame = socket.recv_multipart(copy=False)
                request_type = type_frame.buffer
                request_data = data_frame.buffer

                # Deserialize the request data.
                if request_type == EngineCoreRequestType.ADD.value:
                    request = decoder_add_req.decode(request_data)
                elif request_type == EngineCoreRequestType.ABORT.value:
                    request = decoder_abort_req.decode(request_data)
                elif request_type in (
                        EngineCoreRequestType.PROFILE.value,
                        EngineCoreRequestType.RESET_PREFIX_CACHE.value):
                    request = pickle.loads(request_data)
                else:
                    raise ValueError(f"Unknown RequestType: {request_type}")

                # Push to input queue for core busy loop.
                logger.info(f"engine core proc seed data to input_queue")
                self.input_queue.put_nowait(request)

    def process_output_socket(self, output_path: str):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = msgpack.Encoder()
        # Reuse send buffer.
        buffer = bytearray()

        with zmq_socket_ctx(output_path, zmq.constants.PUSH) as socket:
            while True:
                outputs = self.output_queue.get()
                encoder.encode_into(outputs, buffer)
                socket.send_multipart((buffer, ), copy=False)

    # add by lzc
    # def process_encoder_results(self):
    #     logger.info("process_encoder_results")
    #     while True:
    #         try:
    #             encoder_result = self.encoder_result_queue.get()
    #             logger.info(f"encoder_result is {encoder_result}")
    #             for item in encoder_result:
    #                 logger.info(item[0])
    #         except queue.Empty:
    #             pass




class EncoderCore:
    """Inner loop of vLLM's encoder Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        encoder_result_queue = None
    ):
        assert vllm_config.model_config.runner_type != "pooling"

        logger.info("Initializing a V1 LLM encoder engine (v%s) with config: %s",
                    VLLM_VERSION, vllm_config)

        # Setup Model.
        vllm_config.model_config.only_vision_encoder = True
        self.model_executor = executor_class(vllm_config)
        print(self.model_executor)
        #self.model_executor = executor_class(vllm_config)
        self.model_executor.warm_model()
        #Setup KV Caches and update CacheConfig after profiling.
        # num_gpu_blocks, num_cpu_blocks = self._initialize_kv_caches(
        #     vllm_config)
        # vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        # vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks

        #Setup scheduler.
        vllm_config.model_config.only_vision_encoder = True
        self.scheduler = EncoderScheduler(
            scheduler_config=vllm_config.scheduler_config,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            lora_config=vllm_config.lora_config,
        )
        self.encoder_result_queue = encoder_result_queue
        print(f"self.encoder_result_queue is {self.encoder_result_queue}")
        self.mm_input_mapper_server = MMInputMapperServer(
            vllm_config.model_config)

    def add_request(self, request: EngineCoreRequest):
        """Add request to the scheduler."""

        if request.mm_hashes is not None:
            # Here, if hash exists for an image, then it will be fetched
            # from the cache, else it will be added to the cache.
            # Note that the cache here is mirrored with the client side of the
            # MM mapper, so anything that has a hash must have a HIT cache
            # entry here as well.
            assert request.mm_inputs is not None
            request.mm_inputs = self.mm_input_mapper_server.process_inputs(
                request.mm_inputs, request.mm_hashes)

        req = Request.from_engine_core_request(request)
        #print(f"req.mm_inputs is {req.mm_inputs}")

        self.scheduler.add_request(req)

    def abort_requests(self, request_ids: List[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids,
                                       RequestStatus.FINISHED_ABORTED)

    def step(self, f = sys.stdout) -> EngineCoreOutputs:
        """Schedule, execute, and make output."""
        engine_core_outputs = EngineCoreOutputs(
            outputs=[], scheduler_stats=self.scheduler.make_stats())
        if not self.scheduler.has_unfinished_requests():
            return EngineCoreOutputs(
                outputs=[], scheduler_stats=self.scheduler.make_stats())

        # # find if have multimodal inputs, send it to do encoder and continue the steping, noted by lizhicheng
        # muxserver
        scheduler_output = self.scheduler.schedule()
        
        #logger.info(f"in EncoderCore step, scheduler_output is {scheduler_output}")
        output = None
        if scheduler_output.scheduled_new_reqs != None and len(scheduler_output.scheduled_new_reqs) > 0:
            output =  self.model_executor.execute_vision_encoder(scheduler_output)    
            #then i should send the output to the client
            #logger.info(f"encoder queue add {output}")
            import time;s_time = time.time()
            self.encoder_result_queue.put(output)
            e_time = time.time()
            logger.info(f"add time is {1000 * (e_time - s_time)}")
        # executor to run_encoder
        #logger.info(f"in encoder proc is {output}")
        # original_stdout = sys.stdout
        # sys.stdout = f
        # print(scheduler_output)
        # sys.stdout = original_stdout
        #logger.info(self.model_executor.execute_model)
        #output = self.model_executor.execute_model(scheduler_output)
        # engine_core_outputs = self.scheduler.update_from_output(
        #     scheduler_output, output)
        return engine_core_outputs 

    def shutdown(self):
        logger.info("shutdown")
        #self.model_executor.shutdown()

    # def profile(self, is_start: bool = True):
    #     self.model_executor.profile(is_start)

    # def reset_prefix_cache(self):
    #     self.scheduler.reset_prefix_cache()


class EncoderCoreProc(EncoderCore):
    """ZMQ-wrapper for running EncoderCore in background process."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        ready_pipe: Connection,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        log_stats: bool = False,
        encoder_result_queue = None
    ):
        super().__init__(vllm_config, executor_class, encoder_result_queue)

        self.log_stats = log_stats

        # Background Threads and Queues for IO. These enable us to
        # overlap ZMQ socket IO with GPU since they release the GIL,
        # and to overlap some serialization/deserialization with the
        # model forward pass.
        # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
        self.input_queue: queue.Queue[EngineCoreRequestUnion] = queue.Queue()
        self.output_queue: queue.Queue[EngineCoreOutputs] = queue.Queue()
        #self.encoder_result_queue = encoder_result_queue
        threading.Thread(target=self.process_input_socket,
                         args=(input_path, ),
                         daemon=True).start()
        threading.Thread(target=self.process_output_socket,
                         args=(output_path, ),
                         daemon=True).start()

        # Send Readiness signal to EngineClient.
        ready_pipe.send({"status": "READY"})

    @staticmethod
    def run_encoder_core(*args, **kwargs):
        """Launch encoder busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        parent_process = psutil.Process().parent()
        encoder_core = None
        try:
            encoder_core = EncoderCoreProc(*args, **kwargs)
            encoder_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore interrupted.")

        except Exception:
            traceback = get_exception_traceback()
            logger.error("EngineCore hit an exception: %s", traceback)
            parent_process.send_signal(signal.SIGUSR1)

        finally:
            if encoder_core is not None:
                encoder_core.shutdown()

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        #logger.info("EncoderCoreProc run_busy_loop")
        #fd = open("output.txt", "w")
        while True:
            #1) Poll the input queue until there is work to do.
            if not self.scheduler.has_unfinished_requests():
                while True:
                    try:
                        req = self.input_queue.get(timeout=POLLING_TIMEOUT_S)
                        self._handle_client_request(req)
                        break
                    except queue.Empty:
                        logger.debug("EncoderCore busy loop waiting.")
                        # Break out the loop so we can log_stats in step().
                        if self.log_stats:
                            break
                    except BaseException:
                        raise
            # logger.info(f"self.scheduler.has_unfinished_requests() is {self.scheduler.has_unfinished_requests()}")
            # # 2) Handle any new client requests (Abort or Add).
            while not self.input_queue.empty():
                #logger.info("there is new req in encoder process")
                req = self.input_queue.get_nowait()
                self._handle_client_request(req)
            pass

            # # 3) Step the engine core.
            outputs = self.step()
            #logger.info(f"outputs in encodercoreproc is {outputs}")
            
            
            # # 5) Put EngineCoreOutputs into the output queue.
            # self.output_queue.put_nowait(outputs)

    def _handle_client_request(self, request: EngineCoreRequestUnion) -> None:

        """Handle EngineCoreRequest or EngineCoreABORT from Client."""

        if isinstance(request, EngineCoreRequest):
            self.add_request(request)
        elif isinstance(request, EngineCoreProfile):
            self.model_executor.profile(request.is_start)
        elif isinstance(request, EngineCoreResetPrefixCache):
            self.reset_prefix_cache()
        else: # TODO: make an EngineCoreAbort wrapper
            assert isinstance(request, list)
            self.abort_requests(request)

    def process_input_socket(self, input_path: str):
        
        """Input socket IO thread."""
        logger.info("process_input_socket")
        #exit(0)
        # Msgpack serialization decoding.
        decoder_add_req = PickleEncoder()
        decoder_abort_req = PickleEncoder()

        with zmq_socket_ctx(input_path, zmq.constants.PULL) as socket:
            while True:
                # (RequestType, RequestData)
                type_frame, data_frame = socket.recv_multipart(copy=False)
                request_type = type_frame.buffer
                request_data = data_frame.buffer

                # Deserialize the request data.
                if request_type == EngineCoreRequestType.ADD.value:
                    request = decoder_add_req.decode(request_data)
                elif request_type == EngineCoreRequestType.ABORT.value:
                    request = decoder_abort_req.decode(request_data)
                elif request_type in (
                        EngineCoreRequestType.PROFILE.value,
                        EngineCoreRequestType.RESET_PREFIX_CACHE.value):
                    request = pickle.loads(request_data)
                else:
                    raise ValueError(f"Unknown RequestType: {request_type}")

                # Push to input queue for core busy loop.
                #logger.info(f"encoderproc add request {request}")
                self.input_queue.put_nowait(request)

    def process_output_socket(self, output_path: str):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = msgpack.Encoder()
        # Reuse send buffer.
        buffer = bytearray()

        with zmq_socket_ctx(output_path, zmq.constants.PUSH) as socket:
            while True:
                outputs = self.output_queue.get()
                encoder.encode_into(outputs, buffer)
                socket.send_multipart((buffer, ), copy=False)





