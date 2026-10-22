import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;

/** Task 4: static contiguous blocks versus an atomic dynamic work dispenser. */
public class Task4 {
    static final int ITEMS = 1_000;
    static final int WORKERS = 4;

    record Measurement(long wallNs, long[] finishNs, int[] itemCounts, long[] costsNs) {
        long firstNs() { return Arrays.stream(finishNs).min().orElseThrow(); }
        long lastNs() { return Arrays.stream(finishNs).max().orElseThrow(); }
        long deltaNs() { return lastNs() - firstNs(); }
    }

    static Measurement measure(boolean dynamic, int chunk, boolean sleep) throws Exception {
        if (chunk < 1) throw new IllegalArgumentException("Chunk must be positive");
        AtomicInteger next = new AtomicInteger();
        AtomicIntegerArray visits = new AtomicIntegerArray(ITEMS);
        long[] finished = new long[WORKERS];
        int[] counts = new int[WORKERS];
        long[] costs = new long[WORKERS];
        long[] epoch = new long[1];
        try (ThreadPoolExecutor pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(WORKERS)) {
            pool.prestartAllCoreThreads();
            CountDownLatch ready = new CountDownLatch(WORKERS);
            CountDownLatch start = new CountDownLatch(1);
            List<Future<?>> futures = new ArrayList<>();
            for (int worker = 0; worker < WORKERS; worker++) {
                int id = worker;
                futures.add(pool.submit(() -> {
                    ready.countDown();
                    start.await();
                    int from = dynamic ? next.getAndAdd(chunk) : id * (ITEMS / WORKERS);
                    while (from < ITEMS) {
                        int end = dynamic ? Math.min(from + chunk, ITEMS) : (id + 1) * (ITEMS / WORKERS);
                        for (int item = from; item < end; item++) {
                            long cost = 100_000L * (item % 20);
                            if (sleep && cost > 0) Thread.sleep(cost / 1_000_000, (int) (cost % 1_000_000));
                            costs[id] += cost;
                            counts[id]++;
                            visits.incrementAndGet(item);
                        }
                        if (!dynamic) break;
                        from = next.getAndAdd(chunk);
                    }
                    finished[id] = System.nanoTime() - epoch[0];
                    return null;
                }));
            }
            ready.await();
            epoch[0] = System.nanoTime();
            start.countDown();
            for (Future<?> future : futures) future.get();
            long wall = System.nanoTime() - epoch[0];
            for (int i = 0; i < ITEMS; i++) {
                if (visits.get(i) != 1) throw new AssertionError("Item " + i + " was not processed once");
            }
            return new Measurement(wall, finished, counts, costs);
        }
    }

    public static void main(String[] args) throws Exception {
        BenchmarkSupport.header("TASK 4 - Scheduling under non-uniform item costs");
        System.out.println("Items: 0..999; workers: 4; requested sleep = 0.0001*(i%20) seconds.");
        System.out.println("Static: four contiguous blocks of 250; dynamic: atomic queue, chunks 1 and 5.");
        System.out.println("Dynamic queue scheduling is used, not a claim of a per-worker stealing deque.");
        System.out.println("Pool startup excluded; synchronized release through final future join timed.");
        System.out.println("Sleep accuracy depends on OS timers; every item is verified exactly once.");
        System.out.println("The periodic costs are already nearly balanced across static blocks (232.5/242.5 ms).");
        // Warm the scheduler and verification paths without adding a second sleep workload.
        measure(false, 1, false);
        measure(true, 1, false);
        measure(true, 5, false);
        long[][] walls = new long[3][3];
        long[][] deltas = new long[3][3];
        String[] labels = {"Static", "Dynamic(1)", "Dynamic(5)"};
        for (int trial = 0; trial < 3; trial++) {
            for (int offset = 0; offset < 3; offset++) {
                int strategy = (trial + offset) % 3;
                Measurement result = measure(strategy != 0, strategy == 2 ? 5 : 1, true);
                walls[strategy][trial] = result.wallNs();
                deltas[strategy][trial] = result.deltaNs();
                System.out.printf("trial=%d %-10s wall=%.6f ms (%d ns), first=%.6f ms, last=%.6f ms, delta=%.6f ms%n",
                        trial + 1, labels[strategy], BenchmarkSupport.milliseconds(result.wallNs()),
                        result.wallNs(), BenchmarkSupport.milliseconds(result.firstNs()),
                        BenchmarkSupport.milliseconds(result.lastNs()), BenchmarkSupport.milliseconds(result.deltaNs()));
                System.out.println("  Worker completion offsets (ns): " + Arrays.toString(result.finishNs()));
                System.out.println("  Items per worker: " + Arrays.toString(result.itemCounts()));
                System.out.println("  Requested costs per worker (ns): " + Arrays.toString(result.costsNs()));
            }
        }
        System.out.printf("%-12s | %16s | %19s%n", "Strategy", "Median wall (ms)", "Median delta (ms)");
        for (int i = 0; i < 3; i++) {
            System.out.printf("%-12s | %16.6f | %19.6f%n", labels[i],
                    BenchmarkSupport.milliseconds(BenchmarkSupport.median(walls[i])),
                    BenchmarkSupport.milliseconds(BenchmarkSupport.median(deltas[i])));
        }
    }
}