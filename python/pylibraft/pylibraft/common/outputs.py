# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import functools
import warnings

import pylibraft.config


def import_warn_(lib):
    warnings.warn(
        "%s is not available and output cannot be converted. "
        "Returning original output instead." % lib
    )


def convert_to_torch(device_ndarray):
    try:
        import torch

        return torch.as_tensor(device_ndarray, device="cuda")
    except ImportError:
        import_warn_("PyTorch")
        return device_ndarray


def convert_to_cupy(device_ndarray):
    try:
        import cupy

        return cupy.asarray(device_ndarray)
    except ImportError:
        import_warn_("CuPy")
        return device_ndarray


def no_conversion(device_ndarray):
    return device_ndarray


def convert_to_cai_type(device_ndarray):
    output_as_ = pylibraft.config.output_as_
    if callable(output_as_):
        return output_as_(device_ndarray)
    elif output_as_ == "raft":
        return device_ndarray
    elif output_as_ == "torch":
        return convert_to_torch(device_ndarray)
    elif output_as_ == "cupy":
        return convert_to_cupy(device_ndarray)
    else:
        raise ValueError("No valid type conversion found for %s" % output_as_)


def conv(ret):
    for i in ret:
        if isinstance(i, pylibraft.common.device_ndarray):
            yield convert_to_cai_type(i)
        else:
            yield i


def auto_convert_output(f):
    """Decorator to automatically convert an output device_ndarray
    (or list or tuple of device_ndarray) into the configured
    `__cuda_array_interface__` compliant type.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ret_value = f(*args, **kwargs)
        if isinstance(ret_value, pylibraft.common.device_ndarray):
            return convert_to_cai_type(ret_value)
        elif isinstance(ret_value, tuple):
            return tuple(conv(ret_value))
        elif isinstance(ret_value, list):
            return list(conv(ret_value))
        else:
            return ret_value

    return wrapper
