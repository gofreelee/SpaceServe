# SPDX-License-Identifier: Apache-2.0

from collections import deque
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Deque, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)

from vllm.config import CacheConfig, LoRAConfig, ModelConfig, SchedulerConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalKwargs
    from vllm.multimodal.base import PlaceholderRange

logger = init_logger(__name__)


class EncoderScheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        # TODO: Support LoRA.
        assert lora_config is None, "V1 does not support LoRA yet."

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

        num_gpu_blocks = cache_config.num_gpu_blocks
        #assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        # Create the KV cache manager.
        # self.kv_cache_manager = KVCacheManager(
        #     block_size=self.cache_config.block_size,
        #     num_gpu_blocks=num_gpu_blocks,
        #     max_model_len=self.max_model_len,
        #     sliding_window=self.cache_config.sliding_window,
        #     enable_caching=self.cache_config.enable_prefix_caching)
        # self.block_size = self.cache_config.block_size

        # req_id -> Request
        self.requests: Dict[str, Request] = {}
        # Priority queues for requests.
        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: Set[str] = set()

        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        # Request id -> CachedRequestData
        self._cached_reqs_data: Dict[str, CachedRequestData] = {}

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=model_config,
            scheduler_config=scheduler_config,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

    def schedule(self) -> "SchedulerOutput":
        logger.debug("Scheduling encoder requests...")
        scheduled_encoder_inputs: Dict[str, List[int]] = {}
        scheduled_new_reqs: List[Request] = []
        encoder_budget = self.max_num_encoder_input_tokens

        # Schedule requests for encoder processing
        while self.waiting and encoder_budget > 0:
            if len(self.running) == self.max_num_running_reqs:
                break

            request = self.waiting[0]

            # Get encoder inputs that can be scheduled
            (encoder_inputs_to_schedule,
             new_encoder_budget) = self._try_schedule_encoder_inputs(
                 request, encoder_budget)
            # logger.info(encoder_inputs_to_schedule)
            # logger.info(new_encoder_budget)
            if not encoder_inputs_to_schedule:
                break  # Can't schedule this request

            # Allocate encoder cache
            for idx in encoder_inputs_to_schedule:
                self.encoder_cache_manager.allocate(request, idx)

            self.waiting.popleft()
            #logger.info(f"waiting: {len(self.waiting)}")
            self.running.append(request)
            scheduled_encoder_inputs[request.request_id] = encoder_inputs_to_schedule

            request.status = RequestStatus.RUNNING
            scheduled_new_reqs.append(request)

            encoder_budget = new_encoder_budget
            #break
        
        # just for encoder, so I just need the req, set other fields to None or -1 
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        None, -1)
            for req in scheduled_new_reqs
        ]

        return SchedulerOutput(
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            # Empty fields for compatibility
            scheduled_new_reqs = new_reqs_data,
            # scheduled_resumed_reqs=[],
            # preempted_reqs=[],
            scheduled_cached_reqs=None,
            num_scheduled_tokens=None,
            total_num_scheduled_tokens=0,
            num_common_prefix_blocks=0,
            finished_req_ids=None,
            free_encoder_input_ids=None
        )
      

    def _make_cached_request_data(
        self,
        request: Request,
        new_block_ids: List[int],
        num_computed_tokens: int,
        resumed_from_preemption: bool,
    ) -> "CachedRequestData":
        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        if request.request_id in self._cached_reqs_data:
            req_data = self._cached_reqs_data[request.request_id]
            req_data.resumed_from_preemption = resumed_from_preemption
            req_data.new_block_ids = new_block_ids
            req_data.num_computed_tokens = num_computed_tokens
        else:
            req_data = CachedRequestData.from_request(request,
                                                      resumed_from_preemption,
                                                      new_block_ids,
                                                      num_computed_tokens)
            self._cached_reqs_data[request.request_id] = req_data
        return req_data

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        remaining_budget: int,
    ) -> Tuple[List[int], int]:
        #logger.info(f"remain encoder bugdet: {remaining_budget}")
        schedulable = []
        
        # Filter inputs that need processing
        # print(request.mm_inputs)
        # print(request.mm_positions)
        for i, item in enumerate(request.mm_positions):
            input_size = item["length"]
            if input_size <= remaining_budget:
                schedulable.append(i)
                remaining_budget -= input_size
            else:
                break

        # for input_idx in request.encoder_input_indices:
        #     if input_idx in request.processed_encoder_inputs:
        #         continue
            
        #     # Check input size against remaining budget
        #     input_size = request.get_encoder_input_size(input_idx)
        #     if input_size <= remaining_budget:
        #         schedulable.append(input_idx)
        #         remaining_budget -= input_size
        #     else:
        #         break  # Can't fit this input in current budget
        
        return schedulable, remaining_budget

    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> EngineCoreOutputs:
        # NOTE(woosuk): This method doesn't consider speculative decoding.
        sampled_token_ids = model_runner_output.sampled_token_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        new_running: List[Request] = []
        outputs: List[EngineCoreOutput] = []

        # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        for request in self.running:
            req_id = request.request_id
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                new_running.append(request)
                continue

            request.num_computed_tokens += num_tokens_scheduled
            # When the request's num_computed_tokens catches up its num_tokens,
            # the request generates output tokens. Otherwise, we ignore the
            # sampler output for the request.
            assert request.num_computed_tokens <= request.num_tokens

            cached_encoder_input_ids = (
                self.encoder_cache_manager.get_cached_input_ids(request))
            # OPTIMIZATION: Avoid list(set) if the set is empty.
            if cached_encoder_input_ids:
                for input_id in list(cached_encoder_input_ids):
                    start_pos = request.mm_positions[input_id]["offset"]
                    num_tokens = request.mm_positions[input_id]["length"]
                    if start_pos + num_tokens <= request.num_computed_tokens:
                        # The encoder output is already processed and stored
                        # in the decoder's KV cache.
                        self.encoder_cache_manager.free_encoder_input(
                            request, input_id)

            if request.num_computed_tokens == request.num_tokens:
                req_index = model_runner_output.req_id_to_index[req_id]
                # NOTE(woosuk): Currently, we assume that each request
                # generates at most one token at each step.
                token_id = sampled_token_ids[req_index]
                request.append_output_token_ids(token_id)
                num_new_tokens = 1
                # TODO: Update the KV cache manager for prefix caching.

                # Check for stop and update request state.
                # This must be called before we make the EngineCoreOutput.
                stopped = self._check_stop(request)
                if stopped:
                    self._free_request(request)

                # Add EngineCoreOutput for this Request.
                output = EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=request.output_token_ids[-num_new_tokens:],
                    finished=request.is_finished(),
                    finish_reason=request.get_finished_reason(),
                    stop_reason=request.stop_reason)
                outputs.append(output)

                # Breakout of the loop.
                if stopped:
                    continue

            new_running.append(request)
        self.running = new_running
        return EngineCoreOutputs(
            outputs=outputs,
            scheduler_stats=self.make_stats(),
        )

    def _check_stop(self, request: Request) -> bool:
        if (request.num_tokens >= self.max_model_len
                or request.num_output_tokens >= request.max_tokens):
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return True

        sampling_params = request.sampling_params
        last_token_id = request.output_token_ids[-1]
        if (not sampling_params.ignore_eos
                and last_token_id == request.eos_token_id):
            request.status = RequestStatus.FINISHED_STOPPED
            return True

        if last_token_id in (sampling_params.stop_token_ids or ()):
            request.status = RequestStatus.FINISHED_STOPPED
            request.stop_reason = last_token_id
            return True
        return False

    def add_request(self, request: Request) -> None:
        #import traceback;traceback.print_stack()
        #logger.info(f"Adding request {request.request_id} to the encoder scheduler.")
        self.waiting.append(request)
        self.requests[request.request_id] = request

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        request_ids = set(request_ids)

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            if request.status == RequestStatus.RUNNING:
                self.running.remove(request)
            else:
                self.waiting.remove(request)
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> None:
        assert request.is_finished()
        #self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        self._cached_reqs_data.pop(request.request_id, None)
        del self.requests[request.request_id]
        self.finished_req_ids.add(request.request_id)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_unfinished_requests(self) -> bool:
        return self.get_num_unfinished_requests() > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(self) -> SchedulerStats:
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            #gpu_cache_usage=self.kv_cache_manager.usage,
            gpu_cache_usage= 0,
        )


@dataclass
class NewRequestData:

    req_id: str
    prompt_token_ids: List[int]
    prompt: Optional[str]
    mm_inputs: List["MultiModalKwargs"]
    mm_hashes: List[str]
    mm_positions: List["PlaceholderRange"]
    sampling_params: SamplingParams
    block_ids: List[int]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: List[int],
        num_computed_tokens: int,
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            prompt=request.prompt,
            mm_inputs=request.mm_inputs,
            mm_hashes=request.mm_hashes,
            mm_positions=request.mm_positions,
            sampling_params=request.sampling_params,
            block_ids=block_ids,
            num_computed_tokens=num_computed_tokens,
        )


@dataclass
class CachedRequestData:

    req_id: str
    # If resumed_from_preemption is False, new_block_ids will be appended to
    # the request's block IDs. If True, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_from_preemption: bool
    new_block_ids: List[int]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: Request,
        resumed_from_preemption: bool,
        new_block_ids: List[int],
        num_computed_tokens: int,
    ) -> "CachedRequestData":
        return cls(
            req_id=request.request_id,
            resumed_from_preemption=resumed_from_preemption,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )


@dataclass
class SchedulerOutput:

    scheduled_new_reqs: List[NewRequestData]
    scheduled_cached_reqs: List[CachedRequestData]

    num_scheduled_tokens: Dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_encoder_inputs: Dict[str, List[int]]
    num_common_prefix_blocks: int

    finished_req_ids: Set[str]
    free_encoder_input_ids: List[Tuple[str, int]]

@dataclass
class EncoderSchedulerOutput:

    scheduled_encoder_inputs: Dict[str, List[int]]


