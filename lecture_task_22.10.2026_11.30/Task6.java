import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** Task 6: deliberately racy plain increments versus correct atomic increments. */
public class Task6 {
    static final int WORKERS = 4;
    static final int ITERATIONS = 2_000_000;
    static final int TRIALS = 5;

    static final class PlainCounter {
        int value = 0; // Deliberately neither volatile nor protected by a lock.
    }

    record Measurement(long nanoseconds, int actual, int expected) {
        double lostPercent() {
            return expected == 0 ? 0.0 : 100.0 * ((long) expected - actual) / expected;
        }
    }

    static Measurement measure(int workers, int iterations, boolean atomic) throws Exception {
        if (workers < 1 || iterations < 0) throw new IllegalArgumentException("Invalid workload");
        int expected = Math.multiplyExact(workers, iterations);
        PlainCounter plainCounter = new PlainCounter();
        AtomicInteger atomicCounter = new AtomicInteger(0);
        try (ThreadPoolExecutor pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(workers)) {
            pool.prestartAllCoreThreads();
            CountDownLatch ready = new CountDownLatch(workers);
            CountDownLatch start = new CountDownLatch(1);
            List<Future<?>> futures = new ArrayList<>(workers);
            for (int worker = 0; worker < workers; worker++) {
                futures.add(pool.submit(() -> {
                    ready.countDown();
                    start.await();
                    if (atomic) {
                        for (int i = 0; i < iterations; i++) atomicCounter.incrementAndGet();
                    } else {
                        // No coordination, atomic operations, sleeps or yields inside this loop.
                        for (int i = 0; i < iterations; i++) plainCounter.value++;
                    }
                    return null;
                }));
            }
            try {
                if (!ready.await(30, TimeUnit.SECONDS)) {
                    throw new IllegalStateException("Workers did not become ready");
                }
            } catch (Exception failure) {
                start.countDown(); // Do not strand waiting tasks during pool shutdown.
                throw failure;
            }
            long begin = System.nanoTime();
            start.countDown();
            for (Future<?> future : futures) future.get();
            long elapsed = System.nanoTime() - begin;
            int actual = atomic ? atomicCounter.get() : plainCounter.value;
            if (atomic && actual != expected) throw new AssertionError("Atomic counter lost updates");
            return new Measurement(elapsed, actual, expected);
        }
    }

    static void printResult(String label, Measurement result) {
        System.out.printf("  %-16s Expected: %,d | Actual: %,d | Updates lost: %.6f%% | Wall time: %.6f ms (%d ns)%n",
                label, result.expected(), result.actual(), result.lostPercent(),
                BenchmarkSupport.milliseconds(result.nanoseconds()), result.nanoseconds());
    }

    public static void main(String[] args) throws Exception {
        BenchmarkSupport.header("TASK 6 - Synchronization cost and race condition forensics");
        System.out.printf("Platform threads: %d; increment attempts per thread: %,d%n", WORKERS, ITERATIONS);
        System.out.printf("Expected final value per run: %,d%n", Math.multiplyExact(WORKERS, ITERATIONS));
        System.out.println("Each variant starts with a fresh shared counter = 0.");
        System.out.println("Unsafe: plain int counter++; correct: AtomicInteger.incrementAndGet().");
        System.out.println("Start/finish coordination only outside increment loops; pool startup excluded.");
        System.out.println("Wall time includes common start release, increment loops and future joins.");
        System.out.println("Five paired trials after warmup; variant execution order alternates.");
        for (int warmup = 0; warmup < 4; warmup++) {
            measure(WORKERS, 100_000, false);
            measure(WORKERS, 100_000, true);
        }
        long[] unsafeTimes = new long[TRIALS];
        long[] atomicTimes = new long[TRIALS];
        long[] penalties = new long[TRIALS];
        for (int trial = 0; trial < TRIALS; trial++) {
            Measurement unsafeResult;
            Measurement atomicResult;
            if (trial % 2 == 0) {
                unsafeResult = measure(WORKERS, ITERATIONS, false);
                atomicResult = measure(WORKERS, ITERATIONS, true);
            } else {
                atomicResult = measure(WORKERS, ITERATIONS, true);
                unsafeResult = measure(WORKERS, ITERATIONS, false);
            }
            unsafeTimes[trial] = unsafeResult.nanoseconds();
            atomicTimes[trial] = atomicResult.nanoseconds();
            penalties[trial] = atomicTimes[trial] - unsafeTimes[trial];
            System.out.printf("Trial %d (%s first)%n", trial + 1, trial % 2 == 0 ? "unsafe" : "atomic");
            printResult("Unsynchronized", unsafeResult);
            printResult("AtomicInteger", atomicResult);
            System.out.printf("  Synchronization time penalty (atomic - unsafe): %+.6f ms (%+d ns)%n",
                    BenchmarkSupport.milliseconds(penalties[trial]), penalties[trial]);
        }
        long unsafeMedian = BenchmarkSupport.median(unsafeTimes);
        long atomicMedian = BenchmarkSupport.median(atomicTimes);
        System.out.printf("Median unsynchronized wall time: %.6f ms%n", BenchmarkSupport.milliseconds(unsafeMedian));
        System.out.printf("Median atomic wall time: %.6f ms%n", BenchmarkSupport.milliseconds(atomicMedian));
        System.out.printf("Synchronization time penalty (difference of medians): %+.6f ms (%+d ns)%n",
                BenchmarkSupport.milliseconds(atomicMedian - unsafeMedian), atomicMedian - unsafeMedian);
        if (unsafeMedian > 0) {
            System.out.printf("Atomic / unsafe median wall-time ratio: %.4fx%n", (double) atomicMedian / unsafeMedian);
            System.out.printf("Synchronization overhead relative to unsafe median: %+.4f%%%n",
                    100.0 * (atomicMedian - unsafeMedian) / unsafeMedian);
        } else {
            System.out.println("Ratio/percentage unavailable: unsafe median below clock resolution.");
        }
        System.out.printf("Median paired synchronization penalty: %+.6f ms%n",
                BenchmarkSupport.milliseconds(BenchmarkSupport.median(penalties)));
        System.out.println("Raw unsynchronized wall times (ns): " + java.util.Arrays.toString(unsafeTimes));
        System.out.println("Raw atomic wall times (ns): " + java.util.Arrays.toString(atomicTimes));
        System.out.println("AtomicInteger is an atomic primitive, not an explicit mutex/monitor lock.");
        System.out.println("Unsafe outcomes are nondeterministic: zero observed loss would not prove safety.");
        System.out.println("Plain racy loops may be optimized/coalesced by the JVM; no per-increment memory access guarantee.");
        System.out.println("The timing difference is the cost of these implementations, NOT isolated lock latency.");
        System.out.println("Lost updates (%) = 100*(expected - actual)/expected; no retries to manufacture loss.");
    }
}