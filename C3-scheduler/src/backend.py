"""双后端兼容层：numpy / cupy。

本地无 GPU 时自动回退到 numpy，逻辑可在本地初步测试；
服务器有 cupy + CUDA 时自动用 GPU，满足 NVML 显存采样与性能评测。

用法：
    from src.backend import xp, as_device, as_host, is_gpu, erf
"""

try:
    import cupy as xp  # type: ignore
    _GPU = True
except ImportError:  # pragma: no cover - 本地环境
    import numpy as xp  # type: ignore
    _GPU = False


def is_gpu() -> bool:
    """当前是否运行在 GPU (cupy) 后端。"""
    return _GPU


def backend_name() -> str:
    return "cupy" if _GPU else "numpy"


def as_device(arr):
    """host ndarray -> device array。numpy 后端时原样返回。"""
    if arr is None:
        return None
    if _GPU:
        import cupy as cp  # type: ignore
        if isinstance(arr, cp.ndarray):
            return arr
        return cp.asarray(arr)
    import numpy as _np
    if isinstance(arr, _np.ndarray):
        return arr
    return _np.asarray(arr)


def as_host(arr):
    """device array -> host ndarray (numpy)。numpy 后端时原样返回。"""
    if arr is None:
        return None
    if _GPU:
        import cupy as cp  # type: ignore
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return arr
    import numpy as _np
    if isinstance(arr, _np.ndarray):
        return arr
    return _np.asarray(arr)


def empty(shape, dtype=xp.float32):
    """在当前后端分配空数组。"""
    return xp.empty(shape, dtype=dtype)


def zeros(shape, dtype=xp.float32):
    return xp.zeros(shape, dtype=dtype)


def erf(x):
    """误差函数，cupy 走 cupyx.scipy，numpy 走 scipy.special。"""
    if _GPU:
        try:
            from cupyx.scipy.special import erf as _erf  # type: ignore
            return _erf(x)
        except ImportError:
            # cupy 较新版本直接提供
            return xp.erf(x)
    else:
        from scipy.special import erf as _erf
        return _erf(x)
