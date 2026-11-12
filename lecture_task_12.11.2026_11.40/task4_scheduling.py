import argparse
import math
import multiprocessing as mp
import time

from harness import emit_verify_tag

N = 8_000
CHUNKS = (1, 8, 64, 512)


def item_cost(index):
    return 1 + (index // 40) ** 2


def do_items(indices, timings, slot):
    start = time.perf_counter_ns()
    total = 0.0
    for index in indices:
        for k in range(item_cost(index)):
            total += math.sin(index * k + 1) * math.cos(k + 2) * math.sqrt(k + 3)
    timings[slot] = (time.perf_counter_ns() - start) / 1e6
    return total


def static_worker(start, end, timings, slot):
    do_items(range(start, end), timings, slot)


def cyclic_worker(worker, workers, n, timings):
    do_items(range(worker, n, workers), timings, worker)


def dynamic_worker(counter, lock, chunk, n, timings, slot):
    start = time.perf_counter_ns()
    total = 0.0
    while True:
        with lock:
            begin = counter.value
            counter.value = min(n, begin + chunk)
        if begin >= n:
            break
        for index in range(begin, min(n, begin + chunk)):
            for k in range(item_cost(index)):
                total += math.sin(index * k + 1) * math.cos(k + 2) * math.sqrt(k + 3)
    timings[slot] = (time.perf_counter_ns() - start) / 1e6


def run(strategy, workers, chunk=None, n=N):
    timings = mp.Array("d", workers)
    children = []
    start = time.perf_counter_ns()
    if strategy == "block":
        step = (n + workers - 1) // workers
        for slot in range(workers):
            children.append(mp.Process(target=static_worker,
                                       args=(slot * step, min(n, (slot + 1) * step), timings, slot)))
    elif strategy == "cyclic":
        for slot in range(workers):
            children.append(mp.Process(target=cyclic_worker, args=(slot, workers, n, timings)))
    else:
        counter = mp.Value("i", 0)
        lock = mp.Lock()
        for slot in range(workers):
            children.append(mp.Process(target=dynamic_worker,
                                       args=(counter, lock, chunk, n, timings, slot)))
    for child in children:
        child.start()
    for child in children:
        child.join()
        if child.exitcode != 0:
            raise RuntimeError(f"scheduling worker failed: {child.exitcode}")
    wall = (time.perf_counter_ns() - start) / 1e6
    active = list(timings)
    imbalance = max(active) / (sum(active) / workers)
    return wall, imbalance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    workers = min(physical_count(), 4) if args.quick else physical_count()
    n = 800 if args.quick else N
    wall_block, _ = run("block", workers, n=n)
    wall_cyclic, _ = run("cyclic", workers, n=n)
    print(f"block wall_ms={wall_block:.3f}")
    print(f"cyclic wall_ms={wall_cyclic:.3f}")
    dynamic = {}
    for chunk in CHUNKS:
        dynamic[chunk] = run("dynamic", workers, chunk, n)
        print(f"dynamic chunk={chunk:4d} wall_ms={dynamic[chunk][0]:.3f} "
              f"lambda={dynamic[chunk][1]:.3f}")
    best_chunk = min(dynamic, key=lambda value: dynamic[value][0])
    print(emit_verify_tag("PHASE4", "230103341", wall_block, dynamic[best_chunk][0]))


def physical_count():
    return max(1, (mp.cpu_count() or 1) // 2)


if __name__ == "__main__":
    mp.freeze_support()
    main()
