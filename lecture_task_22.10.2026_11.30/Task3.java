import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

/** Task 3: identical opaque increments, adjacent versus 64-byte-spaced counters. */
public class Task3 {
    static final long ITERATIONS = 50_000_000L;
    static final int PADDED_STRIDE = 64 / Long.BYTES;
    static final VarHandle ELEMENT = MethodHandles.arrayElementVarHandle(long[].class);

    record Measurement(long nanoseconds, long[] counts) { }

    static Measurement measure(int workers, long iterations, int stride) throws Exception {
        if (workers < 1 || iterations < 0 || stride < 1) throw new IllegalArgumentException();
        long[] sharedCounts = new long[workers * stride];
        try (ThreadPoolExecutor pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(workers)) {
            pool.prestartAllCoreThreads();
            CountDownLatch ready = new CountDownLatch(workers);
            CountDownLatch start = new CountDownLatch(1);
            List<Future<?>> futures = new ArrayList<>();
            for (int worker = 0; worker < workers; worker++) {
                int index = worker * stride;
                futures.add(pool.submit(() -> {
                    ready.countDown();
                    start.await();
                    for (long j = 0; j < iterations; j++) {
                        // One writer per element: no atomic RMW or lock is needed.
                        // Opaque access prevents a plain loop being folded to one store.
                        long current = (long) ELEMENT.getOpaque(sharedCounts, index);
                        ELEMENT.setOpaque(sharedCounts, index, current + 1L);
                    }
                    return null;
                }));
            }
            ready.await();
            long begin = System.nanoTime();
            start.countDown();
            for (Future<?> future : futures) future.get();
            long elapsed = System.nanoTime() - begin;
            long[] counts = new long[workers];
            for (int worker = 0; worker < workers; worker++) {
                counts[worker] = sharedCounts[worker * stride];
                if (counts[worker] != iterations) throw new AssertionError("Incorrect counter");
            }
            return new Measurement(elapsed, counts);
        }
    }

    public static void main(String[] args) throws Exception {
        BenchmarkSupport.header("TASK 3 - False sharing profile");
        int workers = BenchmarkSupport.physicalCores();
        System.out.println("Physical cores / platform worker threads: " + workers);
        System.out.printf("Increments per worker: %,d%n", ITERATIONS);
        System.out.println("Unpadded: shared_counts length = N; padded: counter stride = 64 bytes.");
        System.out.println("Identical VarHandle opaque read/write loops; counts verified after join.");
        System.out.println("Synchronized start; pool startup excluded; release + loops + joins included.");
        System.out.println("OS selects CPU placement; no affinity pinning. Cache-line size assumed 64 bytes.");
        measure(workers, 200_000, 1);
        measure(workers, 200_000, PADDED_STRIDE);
        long[] unpadded = new long[3];
        long[] padded = new long[3];
        for (int trial = 0; trial < 3; trial++) {
            Measurement sharedResult;
            Measurement paddedResult;
            if (trial % 2 == 0) {
                sharedResult = measure(workers, ITERATIONS, 1);
                paddedResult = measure(workers, ITERATIONS, PADDED_STRIDE);
            } else {
                paddedResult = measure(workers, ITERATIONS, PADDED_STRIDE);
                sharedResult = measure(workers, ITERATIONS, 1);
            }
            unpadded[trial] = sharedResult.nanoseconds();
            padded[trial] = paddedResult.nanoseconds();
            System.out.printf("trial=%d Unpadded Shared: %.6f ms (%d ns); Padded: %.6f ms (%d ns)%n",
                    trial + 1, BenchmarkSupport.milliseconds(unpadded[trial]), unpadded[trial],
                    BenchmarkSupport.milliseconds(padded[trial]), padded[trial]);
            System.out.println("Verified counts (both variants): " + Arrays.toString(sharedResult.counts()));
        }
        long sharedMedian = BenchmarkSupport.median(unpadded);
        long paddedMedian = BenchmarkSupport.median(padded);
        System.out.printf("Median Unpadded Shared: %.6f ms%n", BenchmarkSupport.milliseconds(sharedMedian));
        System.out.printf("Median Padded: %.6f ms%n", BenchmarkSupport.milliseconds(paddedMedian));
        System.out.printf("Execution ratio (Unpadded Shared / Padded): %.4fx%n",
                (double) sharedMedian / paddedMedian);
    }
}