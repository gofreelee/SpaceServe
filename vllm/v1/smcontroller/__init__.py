import ctypes, ctypes.util
import os
import torch
from typing import Optional
# If this is failing, make sure that the directory containing libsmctrl.so is
# in your LD_LIBRARY_PATH environment variable. You likely need something like:
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/playpen/jbakita/gpu_subdiv/libsmctrl/
libsmctrl_path = ctypes.util.find_library("libsmctrl")
if not libsmctrl_path:
    libsmctrl_path = "/home/lizhicheng/verify/encoder_llm_concurrent/vllm/v1/smcontroller/libsmctrl.so"
libsmctrl = ctypes.CDLL(libsmctrl_path)
print(libsmctrl_path)
print("in libsm")
print(libsmctrl)
print(hasattr(libsmctrl, "libsmctrl_set_stream_mask_lzc"))  # 必须输出True
print(libsmctrl.libsmctrl_set_stream_mask_lzc)              # 应该显示一个有效函数指针

def get_gpc_info(device_num):
    """
    Obtain list of thread processing clusters (TPCs) enabled for each general
    processing cluster (GPC) in the specified GPU.

    Parameters
    ----------
    device_num : int
        Which device to obtain information for (starts as 0, order is defined
        by nvdebug module). May not match CUDA device numbering.

    Returns
    -------
    list of int64
        A list as long as the number of GPCs enabled, where each list entry is
        a bitmask. A bit set at index `i` indicates that TPC `i` is part of the
        GPC at that list index. Obtained via GPU register reads in `nvdebug`.
    """
    num_gpcs = ctypes.c_uint()
    tpc_masks = ctypes.pointer(ctypes.c_ulonglong())
    res = libsmctrl.libsmctrl_get_gpc_info(ctypes.byref(num_gpcs), ctypes.byref(tpc_masks), device_num)
    if res != 0:
        print("pysmctrl: Unable to call libsmctrl_get_gpc_info(). Raising error %d..."%res)
        raise OSError(res, os.strerror(res))
    return [tpc_masks[i] for i in range(num_gpcs.value)]

def get_tpc_info(device_num):
    """
    Obtain a count of the total number of thread processing clusters (TPCs)
    enabled on the specified GPU.

    Parameters
    ----------
    device_num : int
        Which device to obtain TPC count for (starts as 0, order is defined by
        `nvdebug` module). May not match CUDA device numbering.

    Returns
    -------
    int
        Count of enabled TPCs. Obtained via GPU register reads in `nvdebug`.
    """
    num_tpcs = ctypes.c_uint()
    res = libsmctrl.libsmctrl_get_tpc_info(ctypes.byref(num_tpcs), device_num)
    if res != 0:
        print("pysmctrl: Unable to call libsmctrl_get_tpc_info(). Raising error %d..."%res)
        raise OSError(res, os.strerror(res))
    return num_tpcs.value

def get_tpc_info_cuda(device_num):
    """
    Obtain a count of the total number of thread processing clusters (TPCs)
    enabled on the specified GPU.

    Parameters
    ----------
    device_num : int
        Which device to obtain TPC count for, as a CUDA device ID.

    Returns
    -------
    int
        Count of enabled TPCs. Obtained via calculations on data from CUDA.
    """
    num_tpcs = ctypes.c_uint()
    res = libsmctrl.libsmctrl_get_tpc_info_cuda(ctypes.byref(num_tpcs), device_num)
    if res != 0:
        print("pysmctrl: Unable to call libsmctrl_get_tpc_info_cuda(). Raising error %d..."%res)
        raise OSError(res, os.strerror(res))
    return num_tpcs.value

def generate_mask(n: int, shift: Optional[int] = 0):
    """
    Generate a mask with n consecutive unset bits and a shift.

    Parameters
    ----------
    n : int
        The number of consecutive unset bits in the mask.
    shift : int
        The number of bits to shift the unset bits.

    Returns
    -------
    int
        The mask value with n consecutive unset bits and a shift.
    """
    mask = ~(int('1' * n, 2) << shift)
    return mask

def set_stream_mask(torch_stream: torch.cuda.Stream , mask):
    # original c function prototype:extern void libsmctrl_set_stream_mask(void* stream, uint64_t mask);
    stream_ptr = ctypes.c_void_p(torch_stream.cuda_stream)
    libsmctrl.libsmctrl_set_stream_mask(stream_ptr, ctypes.c_uint64(mask))

def stream_lzc_mask(torch_stream, mask_list):
    stream_ptr = ctypes.c_void_p(torch_stream.cuda_stream)
    libsmctrl.libsmctrl_set_stream_mask_lzc(stream_ptr, ctypes.c_uint32(mask_list[0]),ctypes.c_uint32(mask_list[1]), ctypes.c_uint32(mask_list[2]), ctypes.c_uint32(mask_list[3]))