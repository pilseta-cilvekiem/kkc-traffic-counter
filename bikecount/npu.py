"""VeriSilicon VIPLite binding — the NPU on the Allwinner A733.

The A733 carries a VeriSilicon VIP9000 NPU. Allwinner ships no Python (or even C++ wrapper)
API for it, only a C library: `libNBGlinker.so` exports the 40-function VIPLite runtime and
`libVIPhal.so` talks to the `/dev/vipcore` kernel driver. Both come out of Allwinner's model
zoo tarball — see README, "Running on the NPU".

That C API is narrow and stable enough to drive from `ctypes` directly, which is what this
module does: no compiler, no extension module, no build step on the board. The heavy part of
an inference happens entirely inside the NPU anyway, so the Python call overhead per frame is
a handful of microseconds against tens of milliseconds of hardware work.

Models are ACUITY-compiled `.nb` (Network Binary Graph) blobs. They are *not* ONNX and cannot
be produced by ONNX tooling — see `export_npu.md` for the conversion route.
"""

from __future__ import annotations

import ctypes as C
from pathlib import Path

import numpy as np

VIP_SUCCESS = 0
_FROM_FILE = 0x01

# vip_network_property_e
_PROP_INPUT_COUNT, _PROP_OUTPUT_COUNT = 1, 2
# vip_buffer_property_e
_BUF_QUANT_FORMAT, _BUF_NUM_DIMS, _BUF_SIZES, _BUF_DATA_FORMAT = 0, 1, 2, 3
_BUF_FIXED_POINT_POS, _BUF_TF_SCALE, _BUF_TF_ZERO_POINT, _BUF_NAME = 4, 5, 6, 7
# vip_buffer_operation_type_e
_OPER_FLUSH, _OPER_INVALIDATE = 1, 2

# vip_buffer_format_e -> numpy
_DTYPE = {
    0: np.float32, 1: np.float16, 2: np.uint8, 3: np.int8, 4: np.uint16,
    5: np.int16, 6: np.int8, 8: np.int32, 9: np.uint32,
}
_FORMAT_NAME = {0: "fp32", 1: "fp16", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 8: "int32"}


class _Affine(C.Structure):
    _fields_ = [("scale", C.c_float), ("zeroPoint", C.c_int32)]


class _Quant(C.Union):
    _fields_ = [("fixed_point_pos", C.c_int32), ("affine", _Affine)]


class _CreateParams(C.Structure):
    _fields_ = [
        ("num_of_dims", C.c_uint32), ("sizes", C.c_uint32 * 6),
        ("data_format", C.c_int32), ("quant_format", C.c_int32),
        ("quant_data", _Quant), ("memory_type", C.c_uint32),
    ]


class NpuError(RuntimeError):
    pass


class Tensor:
    """One network input or output, as a numpy view onto NPU-visible memory.

    `array` is a live mapping, not a copy: writing into an input tensor and reading out of an
    output tensor are what move data across, with no per-frame allocation. `scale`/`zero_point`
    carry the asymmetric quantisation the NPU works in, for tensors that are not already fp32.
    """

    def __init__(self, lib, buf, name: str, dims: list[int], fmt: int, scale: float, zero_point: int):
        self._lib = lib
        self._buf = buf
        self.name = name
        # VIPLite reports dimensions in [w, h, c, n] order; numpy wants the reverse.
        self.shape = tuple(reversed(dims))
        self.format = _FORMAT_NAME.get(fmt, str(fmt))
        self.scale = scale
        self.zero_point = zero_point

        dtype = _DTYPE.get(fmt)
        if dtype is None:
            raise NpuError(f"tensor {name!r}: unsupported VIPLite format {fmt}")
        nbytes = lib.vip_get_buffer_size(buf)
        ptr = lib.vip_map_buffer(buf)
        if not ptr:
            raise NpuError(f"tensor {name!r}: vip_map_buffer returned NULL")
        buf_type = (C.c_uint8 * nbytes).from_address(ptr)
        self.array = np.frombuffer(buf_type, dtype=dtype).reshape(self.shape)

    def flush(self) -> None:
        """Push CPU writes out to the NPU. Call after filling an input."""
        self._lib.vip_flush_buffer(self._buf, _OPER_FLUSH)

    def invalidate(self) -> None:
        """Drop stale CPU cache lines. Call before reading an output."""
        self._lib.vip_flush_buffer(self._buf, _OPER_INVALIDATE)

    def __repr__(self) -> str:
        return f"<Tensor {self.name!r} {self.shape} {self.format}>"


class VipNetwork:
    """One `.nb` network loaded onto the NPU.

    VIPLite is process-global: `vip_init` is refcounted here so several networks can coexist,
    and the whole thing must be created and used from one thread.
    """

    _lib = None
    _refs = 0

    def __init__(self, nb_path: str | Path, library_path: str | None = None):
        self._network = None
        lib = self._load_library(library_path)

        path = Path(nb_path)
        if not path.exists():
            raise NpuError(f"no such .nb model: {path}")

        if VipNetwork._refs == 0:
            self._check(lib.vip_init(), "vip_init")
        VipNetwork._refs += 1

        net = C.c_void_p()
        self._check(
            lib.vip_create_network(str(path).encode(), 0, _FROM_FILE, C.byref(net)),
            f"vip_create_network({path})",
        )
        self._network = net

        # Buffers can only be attached once the network is prepared.
        self._check(lib.vip_prepare_network(net), "vip_prepare_network")

        self.inputs = self._bind(_PROP_INPUT_COUNT, lib.vip_query_input, lib.vip_set_input)
        self.outputs = self._bind(_PROP_OUTPUT_COUNT, lib.vip_query_output, lib.vip_set_output)

    # ------------------------------------------------------------------ library

    @classmethod
    def _load_library(cls, library_path: str | None):
        if cls._lib is not None:
            return cls._lib
        prefix = f"{library_path.rstrip('/')}/" if library_path else ""
        try:
            # libVIPhal must be RTLD_GLOBAL: libNBGlinker resolves its viphal_* symbols
            # against the already-loaded image rather than by name at link time.
            C.CDLL(f"{prefix}libVIPhal.so", mode=C.RTLD_GLOBAL)
            lib = C.CDLL(f"{prefix}libNBGlinker.so", mode=C.RTLD_GLOBAL)
        except OSError as exc:
            raise NpuError(
                f"cannot load the VIPLite runtime ({exc}). Copy libVIPhal.so and "
                f"libNBGlinker.so out of Allwinner's model zoo and point `model.npu_libs` "
                f"at them, or set LD_LIBRARY_PATH. See README, 'Running on the NPU'."
            ) from exc
        lib.vip_map_buffer.restype = C.c_void_p
        lib.vip_get_version.restype = C.c_uint32
        lib.vip_get_buffer_size.restype = C.c_uint32
        cls._lib = lib
        return lib

    @property
    def version(self) -> str:
        v = self._lib.vip_get_version()
        return f"{(v >> 24) & 0xFF}.{(v >> 16) & 0xFF}.{(v >> 8) & 0xFF}.{v & 0xFF}"

    @staticmethod
    def _check(status: int, what: str) -> None:
        if status != VIP_SUCCESS:
            raise NpuError(f"{what} failed with VIPLite status {status}")

    # ------------------------------------------------------------------ binding

    def _bind(self, count_prop: int, query, setter) -> list[Tensor]:
        lib = self._lib
        count = C.c_uint32()
        lib.vip_query_network(self._network, count_prop, C.byref(count))

        tensors = []
        for i in range(count.value):
            # ctypes yields plain ints for scalar struct fields, so byref() cannot target
            # them: query into standalone cells, then copy into the create-params struct.
            fmt, ndim, qfmt = C.c_int32(), C.c_uint32(), C.c_int32()
            sizes = (C.c_uint32 * 6)()
            name = C.create_string_buffer(256)
            query(self._network, i, _BUF_DATA_FORMAT, C.byref(fmt))
            query(self._network, i, _BUF_NUM_DIMS, C.byref(ndim))
            query(self._network, i, _BUF_SIZES, sizes)
            query(self._network, i, _BUF_QUANT_FORMAT, C.byref(qfmt))
            query(self._network, i, _BUF_NAME, name)

            params = _CreateParams()
            params.data_format = fmt.value
            params.num_of_dims = ndim.value
            params.sizes = sizes
            params.quant_format = qfmt.value
            params.memory_type = 0

            scale, zero_point = 1.0, 0
            if qfmt.value == 2:  # VIP_BUFFER_QUANTIZE_TF_ASYMM
                sc, zp = C.c_float(), C.c_int32()
                query(self._network, i, _BUF_TF_SCALE, C.byref(sc))
                query(self._network, i, _BUF_TF_ZERO_POINT, C.byref(zp))
                params.quant_data.affine.scale = sc.value
                params.quant_data.affine.zeroPoint = zp.value
                scale, zero_point = sc.value, zp.value
            elif qfmt.value == 1:  # dynamic fixed point
                fp = C.c_int32()
                query(self._network, i, _BUF_FIXED_POINT_POS, C.byref(fp))
                params.quant_data.fixed_point_pos = fp.value

            buf = C.c_void_p()
            self._check(
                lib.vip_create_buffer(C.byref(params), C.sizeof(params), C.byref(buf)),
                f"vip_create_buffer(index {i})",
            )
            setter(self._network, i, buf)
            tensors.append(
                Tensor(
                    lib, buf, name.value.decode(errors="replace"),
                    [sizes[d] for d in range(ndim.value)], fmt.value, scale, zero_point,
                )
            )
        return tensors

    # ------------------------------------------------------------------ running

    def run(self) -> list[np.ndarray]:
        """Execute the network over whatever is currently in the input tensors."""
        for t in self.inputs:
            t.flush()
        self._check(self._lib.vip_run_network(self._network), "vip_run_network")
        for t in self.outputs:
            t.invalidate()
        return [t.array for t in self.outputs]

    def close(self) -> None:
        if self._network is None:
            return
        self._lib.vip_destroy_network(self._network)
        self._network = None
        VipNetwork._refs -= 1
        if VipNetwork._refs == 0:
            self._lib.vip_destroy()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
