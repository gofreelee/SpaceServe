import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Optional

import torch

_LIBSMCTRL_ENV_VAR = "VLLM_LIBSMCTRL_PATH"


def _candidate_libsmctrl_paths() -> list[str]:
    candidates: list[str] = []

    env_path = os.getenv(_LIBSMCTRL_ENV_VAR)
    if env_path:
        candidates.append(env_path)

    for lib_name in ("smctrl", "libsmctrl"):
        discovered = ctypes.util.find_library(lib_name)
        if discovered:
            candidates.append(discovered)

    candidates.append(str(Path(__file__).resolve().with_name("libsmctrl.so")))

    # Preserve search order while deduplicating repeated candidates.
    return list(dict.fromkeys(candidates))


def _load_libsmctrl() -> tuple[Optional[ctypes.CDLL], Optional[str], str]:
    errors: list[str] = []

    for candidate in _candidate_libsmctrl_paths():
        try:
            return ctypes.CDLL(candidate), candidate, ""
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")

    details = "\n".join(errors) if errors else "no candidate paths were found"
    return None, None, details


libsmctrl, libsmctrl_path, _libsmctrl_load_error = _load_libsmctrl()


def _require_libsmctrl() -> ctypes.CDLL:
    if libsmctrl is not None:
        return libsmctrl

    raise RuntimeError(
        "SM partitioning requires libsmctrl, but it could not be loaded. "
        f"Set {_LIBSMCTRL_ENV_VAR} to the full path of libsmctrl.so or add "
        "its directory to LD_LIBRARY_PATH.\n"
        f"Tried:\n{_libsmctrl_load_error}")


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
    lib = _require_libsmctrl()
    num_gpcs = ctypes.c_uint()
    tpc_masks = ctypes.pointer(ctypes.c_ulonglong())
    res = lib.libsmctrl_get_gpc_info(ctypes.byref(num_gpcs),
                                     ctypes.byref(tpc_masks), device_num)
    if res != 0:
        print("pysmctrl: Unable to call libsmctrl_get_gpc_info(). "
              "Raising error %d..." % res)
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
    lib = _require_libsmctrl()
    num_tpcs = ctypes.c_uint()
    res = lib.libsmctrl_get_tpc_info(ctypes.byref(num_tpcs), device_num)
    if res != 0:
        print("pysmctrl: Unable to call libsmctrl_get_tpc_info(). "
              "Raising error %d..." % res)
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
    lib = _require_libsmctrl()
    num_tpcs = ctypes.c_uint()
    res = lib.libsmctrl_get_tpc_info_cuda(ctypes.byref(num_tpcs), device_num)
    if res != 0:
        print("pysmctrl: Unable to call libsmctrl_get_tpc_info_cuda(). "
              "Raising error %d..." % res)
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


def set_stream_mask(torch_stream: torch.cuda.Stream, mask):
    lib = _require_libsmctrl()
    stream_ptr = ctypes.c_void_p(torch_stream.cuda_stream)
    lib.libsmctrl_set_stream_mask(stream_ptr, ctypes.c_uint64(mask))


def stream_lzc_mask(torch_stream: torch.cuda.Stream, mask_list):
    lib = _require_libsmctrl()
    if len(mask_list) != 4:
        raise ValueError("mask_list must contain exactly 4 uint32 values")

    stream_ptr = ctypes.c_void_p(torch_stream.cuda_stream)
    lib.libsmctrl_set_stream_mask_lzc(
        stream_ptr,
        ctypes.c_uint32(mask_list[0]),
        ctypes.c_uint32(mask_list[1]),
        ctypes.c_uint32(mask_list[2]),
        ctypes.c_uint32(mask_list[3]),
    )
