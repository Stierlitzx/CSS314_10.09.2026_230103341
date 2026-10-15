"""Tasks 35--42: actual asynchronous CUDA and lazy-imported CuPy operations.

Host wrappers, not bracket-launch kernels. No CUDA simulator support and no CPU
fallback. Float32, C-contiguous arrays are required. Task35 doubles its input
(the task leaves the pointwise operation unspecified). Task36 accepts host
matrix(M,N), vectors(N,B), output(M,B); columns are staged independently.
Timing results describe this machine only, not guaranteed speedup or overlap.
Task40 captures two kernels, c=a+b then c=2*c; every replay overwrites c,
so even after multiple replays c=2*(a+b), not a persistent recurrence.
Task42 mutates a device vector: each iteration adds 1 then, after a real
grid-wide barrier, adds the iteration's first element to all other elements.
    It uses CuPy RawKernel cooperative launch and occupancy-based grid capping.
"""
import time

import numpy as np
from numba import cuda, config


def _hardware():
    if config.ENABLE_CUDASIM:
        raise RuntimeError("This operation needs real CUDA hardware, not CUDASIM")
    if not cuda.is_available():
        raise RuntimeError("A working CUDA driver and GPU are required")


def _cupy():
    _hardware()
    import cupy as cp
    return cp


def _host_float32(a, name):
    if not isinstance(a, np.ndarray) or a.dtype != np.float32 or not a.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous float32 NumPy array")


@cuda.jit
def _double(x, y, n):
    i = cuda.grid(1)
    if i < n:
        y[i] = 2 * x[i]


def pipeline_dual_stream(h_in, h_out, chunk_size):
    """Two reusable pinned/device staging slots; return h_out with x*2.

    Each slot waits only before reuse, so H2D/kernel/D2H of the other slot
    may overlap. All transfers are complete on return. Arbitrary tail sizes.
    """
    _hardware()
    _host_float32(h_in, "h_in")
    _host_float32(h_out, "h_out")
    if h_in.ndim != 1 or h_out.shape != h_in.shape or chunk_size <= 0:
        raise ValueError("equal 1D shapes and a positive chunk_size are required")
    streams = [cuda.stream(), cuda.stream()]
    ins = [cuda.pinned_array(chunk_size, np.float32) for _ in streams]
    outs = [cuda.pinned_array(chunk_size, np.float32) for _ in streams]
    dx = [cuda.device_array(chunk_size, np.float32) for _ in streams]
    dy = [cuda.device_array(chunk_size, np.float32) for _ in streams]
    pending = [None, None]
    for count, start in enumerate(range(0, h_in.size, chunk_size)):
        slot = count % 2
        streams[slot].synchronize()
        if pending[slot] is not None:
            old, n = pending[slot]
            h_out[old:old + n] = outs[slot][:n]
        n = min(chunk_size, h_in.size - start)
        ins[slot][:n] = h_in[start:start + n]
        dx[slot][:n].copy_to_device(ins[slot][:n], stream=streams[slot])
        _double[(n + 255) // 256, 256, streams[slot]](dx[slot], dy[slot], n)
        dy[slot][:n].copy_to_host(outs[slot][:n], stream=streams[slot])
        pending[slot] = start, n
    for slot in range(2):
        streams[slot].synchronize()
        if pending[slot] is not None:
            start, n = pending[slot]
            h_out[start:start + n] = outs[slot][:n]
    return h_out


@cuda.jit
def _gemv(a, x, y):
    row = cuda.grid(1)
    if row < a.shape[0]:
        value = 0.0
        for j in range(a.shape[1]):
            value += a[row, j] * x[j]
        y[row] = value


def multistream_gemv(matrix, vectors, out_vectors, num_streams):
    """Async pinned staging, round-robin batch columns; result matrix@vectors."""
    _hardware()
    for a, name in ((matrix, "matrix"), (vectors, "vectors"), (out_vectors, "out_vectors")):
        _host_float32(a, name)
    if (matrix.ndim != 2 or vectors.ndim != 2 or num_streams < 1 or
            vectors.shape[0] != matrix.shape[1] or
            out_vectors.shape != (matrix.shape[0], vectors.shape[1]) or matrix.shape[0] < 1):
        raise ValueError("expected matrix(M,N), vectors(N,B), output(M,B), positive streams/M")
    da = cuda.to_device(matrix)
    cuda.synchronize()  # matrix must be visible to all non-default streams
    streams = [cuda.stream() for _ in range(num_streams)]
    hx = [cuda.pinned_array(matrix.shape[1], np.float32) for _ in streams]
    hy = [cuda.pinned_array(matrix.shape[0], np.float32) for _ in streams]
    dx = [cuda.device_array(matrix.shape[1], np.float32) for _ in streams]
    dy = [cuda.device_array(matrix.shape[0], np.float32) for _ in streams]
    pending = [None] * num_streams
    for b in range(vectors.shape[1]):
        s = b % num_streams
        streams[s].synchronize()
        if pending[s] is not None:
            out_vectors[:, pending[s]] = hy[s]
        hx[s][:] = vectors[:, b]
        dx[s].copy_to_device(hx[s], stream=streams[s])
        _gemv[(matrix.shape[0] + 127) // 128, 128, streams[s]](da, dx[s], dy[s])
        dy[s].copy_to_host(hy[s], stream=streams[s])
        pending[s] = b
    for s in range(num_streams):
        streams[s].synchronize()
        if pending[s] is not None:
            out_vectors[:, pending[s]] = hy[s]
    return out_vectors


def _device_float32(cp, arrays):
    if any(not isinstance(a, cp.ndarray) or a.dtype != cp.float32 or
           not a.flags.c_contiguous for a in arrays):
        raise ValueError("expected C-contiguous float32 CuPy arrays")
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("array shapes must match")


def cupy_rawkernel_sinc(x, out):
    cp = _cupy()
    _device_float32(cp, (x, out))
    kernel = cp.RawKernel(r'''
    extern "C" __global__ void sinc_kernel(const float* x, float* y, long long n) {
        long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) { float z=x[i]; y[i] = z==0.0f ? 1.0f : sinpif(z)/(3.14159265358979323846f*z); }
    }''', 'sinc_kernel')
    if x.size:
        kernel(((x.size + 255) // 256,), (256,), (x, out, np.int64(x.size)))
    return out


def cupy_rawmodule_arithmetic(a, b, res_add, res_sub, res_mul):
    cp = _cupy()
    _device_float32(cp, (a, b, res_add, res_sub, res_mul))
    module = cp.RawModule(code=r'''
    extern "C" __global__ void add(const float* a,const float* b,float* c,long long n) {
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)c[i]=a[i]+b[i]; }
    extern "C" __global__ void sub(const float* a,const float* b,float* c,long long n) {
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)c[i]=a[i]-b[i]; }
    extern "C" __global__ void mul(const float* a,const float* b,float* c,long long n) {
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)c[i]=a[i]*b[i]; }
    ''')
    if a.size:
        for name, out in (("add", res_add), ("sub", res_sub), ("mul", res_mul)):
            module.get_function(name)(((a.size + 255) // 256,), (256,),
                                      (a, b, out, np.int64(a.size)))
    return res_add, res_sub, res_mul


def cuda_event_profiler(kernel_func, args, num_warmup, num_repeats):
    """Profile a configured callable on CUDA's default stream.

    e.g. kernel[grid,block] is already configured; custom-stream callables
    must synchronize into the default stream themselves. Returns ms samples,
    population variance (ms^2), mean/min/max. No wall-time proxy for GPU time.
    """
    _hardware()
    if num_warmup < 0 or num_repeats < 1:
        raise ValueError("num_warmup>=0 and num_repeats>=1 required")
    for _ in range(num_warmup):
        kernel_func(*args)
    cuda.synchronize()
    start, stop = cuda.event(timing=True), cuda.event(timing=True)
    samples = []
    for _ in range(num_repeats):
        start.record()
        kernel_func(*args)
        stop.record()
        stop.synchronize()
        samples.append(float(cuda.event_elapsed_time(start, stop)))
    return {"samples_ms": samples, "mean_ms": float(np.mean(samples)),
            "variance_ms2": float(np.var(samples)), "min_ms": min(samples),
            "max_ms": max(samples)}


def cuda_graph_capture_exec(a, b, c, num_iterations):
    cp = _cupy()
    _device_float32(cp, (a, b, c))
    if num_iterations < 1 or a.size == 0:
        raise ValueError("positive iteration count and nonempty arrays required")
    if cp.may_share_memory(c, a) or cp.may_share_memory(c, b):
        raise ValueError("graph output must not alias inputs")
    add = cp.RawKernel(r'''
    extern "C" __global__ void graph_add(const float* a,const float* b,float* c,long long n){
      long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}
    ''', 'graph_add')
    scale = cp.RawKernel(r'''
    extern "C" __global__ void graph_scale(float* c,long long n){
      long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]*=2.0f;}
    ''', 'graph_scale')
    add.compile()
    scale.compile()
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    grid = ((a.size + 255) // 256,)
    with stream:
        stream.begin_capture()
        add(grid, (256,), (a, b, c, np.int64(a.size)))
        scale(grid, (256,), (c, np.int64(a.size)))
        graph = stream.end_capture()
        for _ in range(num_iterations):
            graph.launch(stream)
    stream.synchronize()
    return c


def benchmark_unified_memory(size_bytes):
    """Managed float32 memory: CPU initialization -> GPU increment -> CPU check.

    Returns measured GPU first-touch and second-access ms, CPU touch seconds,
    and prefetch+kernel ms when concurrent managed access permits prefetch.
    No prefetched measurement is fabricated on unsupported platforms.
    """
    cp = _cupy()
    if size_bytes < 4 or size_bytes % 4:
        raise ValueError("size_bytes must be a positive multiple of four")
    import ctypes
    memory = cp.cuda.memory.ManagedMemory(size_bytes)
    pointer = cp.cuda.MemoryPointer(memory, 0)
    device = cp.ndarray((size_bytes // 4,), dtype=cp.float32, memptr=pointer)
    host = np.ctypeslib.as_array((ctypes.c_float * device.size).from_address(memory.ptr))
    stamp = time.perf_counter()
    host.fill(1)
    cpu_touch_seconds = time.perf_counter() - stamp
    kernel = cp.RawKernel(r'''
    extern "C" __global__ void increment(float* x,long long n){
      long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=1.0f;}
    ''', 'increment')
    kernel.compile()
    stream = cp.cuda.get_current_stream()
    def measure(prefetch=False):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record(stream)
        if prefetch:
            cp.cuda.runtime.memPrefetchAsync(memory.ptr, size_bytes,
                                            cp.cuda.Device().id, stream.ptr)
        kernel(((device.size + 255) // 256,), (256,), (device, np.int64(device.size)))
        end.record(stream)
        end.synchronize()
        return float(cp.cuda.get_elapsed_time(start, end))
    first = measure()
    second = measure()
    valid = bool(np.all(host == 3))
    prefetch_ms = None
    reason = "device lacks ConcurrentManagedAccess (prefetch not attempted)"
    if cp.cuda.Device().attributes.get("ConcurrentManagedAccess", 0):
        cp.cuda.runtime.memPrefetchAsync(memory.ptr, size_bytes, -1, stream.ptr)
        stream.synchronize()
        host.fill(1)
        prefetch_ms = measure(True)
        valid = valid and bool(np.all(host == 2))
        reason = None
    return {"size_bytes": size_bytes, "first_touch_ms": first,
            "second_access_ms": second, "cpu_touch_seconds": cpu_touch_seconds,
            "prefetch_ms": prefetch_ms, "prefetch_unavailable_reason": reason,
            "verified": valid}


def grid_cooperative_barrier(arr, iterations):
    """Occupancy-safe cooperative launch; returns arr after synchronization.

    For initial a0: out[0]=a0[0]+k, out[i]=a0[i]+k+k*a0[0]+k*(k+1)/2
    for i>0. Requires cooperative-launch capable hardware/driver.
    """
    cp = _cupy()
    _device_float32(cp, (arr,))
    if arr.ndim != 1 or arr.size < 1 or iterations < 0:
        raise ValueError("nonempty device vector and iterations>=0 required")
    if not cp.cuda.Device().attributes.get("CooperativeLaunch", 0):
        raise RuntimeError("Device does not support cooperative launch")
    kernel = cp.RawKernel(r'''
    #include <cooperative_groups.h>
    namespace cg = cooperative_groups;
    extern "C" __global__ void cooperative_barrier(float* arr,long long n,int iters){
        cg::grid_group grid = cg::this_grid();
        long long tid=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        long long stride=(long long)blockDim.x*gridDim.x;
        for(int k=0;k<iters;++k){
            for(long long i=tid;i<n;i+=stride) arr[i]+=1.0f;
            grid.sync();
            for(long long i=tid;i<n;i+=stride) if(i!=0)arr[i]+=arr[0];
            grid.sync();
        }
    }''', 'cooperative_barrier', options=('--std=c++17',),
                          enable_cooperative_groups=True)
    # CuPy queries occupancy and caps an oversized 1D cooperative grid.
    # Grid-stride loops preserve coverage if that cap reduces the grid.
    kernel(((arr.size + 127) // 128,), (128,),
           (arr, np.int64(arr.size), np.int32(iterations)))
    cp.cuda.get_current_stream().synchronize()
    return arr