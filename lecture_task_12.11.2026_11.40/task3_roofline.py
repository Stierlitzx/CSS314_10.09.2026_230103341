import argparse
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
import time

from harness import emit_verify_tag

TILES = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def copy_worker(size):
    source = bytearray(size)
    target = bytearray(size)
    target[:] = source


def fma_worker(iterations):
    values = [1.001] * 16
    for _ in range(iterations // 16):
        for index in range(16):
            values[index] = values[index] * 1.0000001 + 0.0000001
    return values[0]


def gemm_worker(a_name, b_name, c_name, n, start, end, variant, block):
    a_shm = shared_memory.SharedMemory(name=a_name)
    b_shm = shared_memory.SharedMemory(name=b_name)
    c_shm = shared_memory.SharedMemory(name=c_name)
    try:
        a = (ctypes.c_double * (n * n)).from_buffer(a_shm.buf)
        b = (ctypes.c_double * (n * n)).from_buffer(b_shm.buf)
        c = (ctypes.c_double * (n * n)).from_buffer(c_shm.buf)
        if variant == 1:
            for i in range(start, end):
                for j in range(n):
                    total = 0.0
                    for k in range(n):
                        total += a[i * n + k] * b[k * n + j]
                    c[i * n + j] = total
        elif variant == 2:
            for i in range(start, end):
                row = i * n
                for k in range(n):
                    aik = a[row + k]
                    brow = k * n
                    for j in range(n):
                        c[row + j] += aik * b[brow + j]
        else:
            for ii in range(start, end, block):
                for kk in range(0, n, block):
                    for jj in range(0, n, block):
                        for i in range(ii, min(ii + block, end)):
                            row = i * n
                            for k in range(kk, min(kk + block, n)):
                                aik = a[row + k]
                                brow = k * n
                                for j in range(jj, min(jj + block, n)):
                                    c[row + j] += aik * b[brow + j]
        del a, b, c
    finally:
        a_shm.close()
        b_shm.close()
        c_shm.close()


def gemm(n, workers, variant, block=16):
    size = n * n * 8
    a_shm = shared_memory.SharedMemory(create=True, size=size)
    b_shm = shared_memory.SharedMemory(create=True, size=size)
    c_shm = shared_memory.SharedMemory(create=True, size=size)
    try:
        a = (ctypes.c_double * (n * n)).from_buffer(a_shm.buf)
        b = (ctypes.c_double * (n * n)).from_buffer(b_shm.buf)
        c = (ctypes.c_double * (n * n)).from_buffer(c_shm.buf)
        for index in range(n * n):
            a[index] = ((index * 17) % 101) / 101.0
            b[index] = ((index * 13 + 7) % 97) / 97.0
            c[index] = 0.0
        del a, b, c
        step = (n + workers - 1) // workers
        children = []
        start = time.perf_counter_ns()
        for worker in range(workers):
            lo = worker * step
            hi = min(n, (worker + 1) * step)
            if lo < hi:
                children.append(mp.Process(target=gemm_worker,
                    args=(a_shm.name, b_shm.name, c_shm.name, n, lo, hi, variant, block)))
        for child in children:
            child.start()
        for child in children:
            child.join()
            if child.exitcode != 0:
                raise RuntimeError("GEMM worker failed")
        elapsed = (time.perf_counter_ns() - start) / 1e6
        result = list((ctypes.c_double * (n * n)).from_buffer(c_shm.buf))
        return elapsed, result
    finally:
        a_shm.close()
        b_shm.close()
        c_shm.close()
        a_shm.unlink()
        b_shm.unlink()
        c_shm.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    workers = min(physical_count(), 4) if args.quick else physical_count()
    size = 48 if args.quick else 384
    copy_size = 1 << 20 if args.quick else 96 * 10**6
    start = time.perf_counter_ns()
    with mp.Pool(workers) as pool:
        pool.map(copy_worker, [copy_size] * workers)
    bandwidth = 2 * copy_size * workers / ((time.perf_counter_ns() - start) / 1e9) / 1e9
    fma_iterations = 10_000 if args.quick else 80_000_000
    start = time.perf_counter_ns()
    with mp.Pool(workers) as pool:
        pool.map(fma_worker, [fma_iterations] * workers)
    compute = 16 * (fma_iterations // 16) * 2 * workers / ((time.perf_counter_ns() - start) / 1e9) / 1e9
    print(f"workers={workers} bandwidth_GBps={bandwidth:.4f} compute_GFLOPS={compute:.4f}")
    print(f"ridge_point_FLOP_per_byte={compute / bandwidth:.6f}")
    variant1_ms, baseline = gemm(size, workers, 1)
    variant2_ms, reordered = gemm(size, workers, 2)
    print(f"gemm variant=1 N={size} median_ms={variant1_ms:.3f} GFLOPS={2 * size**3 / variant1_ms / 1e6:.6f}")
    print(f"gemm variant=2 N={size} median_ms={variant2_ms:.3f} GFLOPS={2 * size**3 / variant2_ms / 1e6:.6f}")
    print(f"gemm max_error_variant2={max(abs(x - y) for x, y in zip(baseline, reordered)):.3e}")
    best = None
    for tile in TILES:
        elapsed, blocked = gemm(size, workers, 3, tile)
        error = max(abs(x - y) for x, y in zip(reordered, blocked))
        gflops = 2 * size**3 / elapsed / 1e6
        print(f"gemm variant=3 BS={tile} median_ms={elapsed:.3f} GFLOPS={gflops:.6f} max_error={error:.3e}")
        if best is None or gflops > best[1]:
            best = (tile, gflops)
    print(f"best_tile={best[0]} best_GFLOPS={best[1]:.6f}")
    print(emit_verify_tag("PHASE3", "230103341",
                          2 * size**3 / variant1_ms / 1e6, best[1]))


def physical_count():
    return max(1, (mp.cpu_count() or 1) // 2)


if __name__ == "__main__":
    mp.freeze_support()
    main()
