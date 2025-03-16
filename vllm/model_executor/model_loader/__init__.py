# SPDX-License-Identifier: Apache-2.0

from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.loader import (BaseModelLoader,
                                                     get_model_loader)
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name, get_model_architecture)


from vllm.logger import init_logger

logger = init_logger(__name__)

def get_model(*, vllm_config: VllmConfig) -> nn.Module:
    loader = get_model_loader(vllm_config.load_config)
    # logger.info(f"loader type is {type(loader)}")
    # logger.info(f"{loader}")
    return loader.load_model(vllm_config=vllm_config)


__all__ = [
    "get_model", "get_model_loader", "BaseModelLoader",
    "get_architecture_class_name", "get_model_architecture"
]
