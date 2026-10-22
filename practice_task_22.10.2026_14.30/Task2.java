import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.Callable;
import java.util.concurrent.ThreadPoolExecutor;

/** Identical PRNG algorithm: synchronized shared state versus seeded per-worker state. */
public class Task2 {
    static final int WORKERS = 4;
    static final int SAMPLES = 40_000_000;
    static final int TRIALS = 3;

    record Measurement(long ns, long hits, int samples) {
        double pi() { return 4.0 * hits / samples; }
    }

    static Measurement measure(int samples, int workers, boolean shared, ThreadPoolExecutor pool) throws Exception {
        if (samples < 1 || workers < 1) throw new IllegalArgumentException("Positive sample/worker counts required");
        SplittableRandom sharedRandom = new SplittableRandom(PracticeSupport.STUDENT_ID);
        Object lock = new Object();
        List<Callable<Long>> jobs = new ArrayList<>();
        for (int worker = 0; worker < workers; worker++) {
            int index = worker;
            int count = PracticeSupport.boundary(worker + 1, samples, workers)
                    - PracticeSupport.boundary(worker, samples, workers);
            jobs.add(() -> {
                SplittableRandom independent = new SplittableRandom(PracticeSupport.STUDENT_ID * 10 + index);
                long hits = 0;
                for (int point = 0; point < count; point++) {
                    double x;
                    double y;
                    if (shared) {
                        // One synchronization per POINT, both coordinates drawn under the same lock.
                        synchronized (lock) {
                            x = sharedRandom.nextDouble();
                            y = sharedRandom.nextDouble();
                        }
                    } else {
                        x = independent.nextDouble();
                        y = independent.nextDouble();
                    }
                    if (x * x + y * y <= 1.0) hits++;
                }
                return hits;
            });
        }
        long begin = System.nanoTime();
        long hits = 0;
        for (long partial : PracticeSupport.invoke(pool, jobs)) hits += partial;
        long elapsed = System.nanoTime() - begin;
        if (hits < 0 || hits > samples) throw new AssertionError("Invalid hit count");
        return new Measurement(elapsed, hits, samples);
    }

    static void print(String label, Measurement result) {
        System.out.printf("%s: pi=%.12f | absolute error=%.12f | wall=%.6f ms (%d ns) | hits=%d/%d%n",
                label, result.pi(), Math.abs(result.pi() - Math.PI), PracticeSupport.ms(result.ns()),
                result.ns(), result.hits(), result.samples());
    }

    public static void main(String[] args) throws Exception {
        PracticeSupport.header("PRACTICE TASK 2 - PRNG state bottleneck");
        System.out.printf("Threads: %d; sample points per run: %,d%n", WORKERS, SAMPLES);
        System.out.println("Both variants use SplittableRandom; fresh seed/state for every run.");
        System.out.printf("Shared seed: %d; independent seed[i] = Student_ID*10+i: [%d, %d, %d, %d]%n",
                PracticeSupport.STUDENT_ID, PracticeSupport.STUDENT_ID * 10, PracticeSupport.STUDENT_ID * 10 + 1,
                PracticeSupport.STUDENT_ID * 10 + 2, PracticeSupport.STUDENT_ID * 10 + 3);
        System.out.println("Shared PRNG locked once per point for two draws; local hit counts reduced after joins.");
        System.out.println("Timing: dispatch + worker-local PRNG initialization + sampling + joins/reduction; pool/shared-state/job setup excluded.");
        long[] shared = new long[TRIALS];
        long[] local = new long[TRIALS];
        try (ThreadPoolExecutor pool = PracticeSupport.pool(WORKERS)) {
            for (int warmup = 0; warmup < 3; warmup++) {
                measure(100_000, WORKERS, true, pool);
                measure(100_000, WORKERS, false, pool);
            }
            for (int trial = 0; trial < TRIALS; trial++) {
                Measurement a;
                Measurement b;
                if (trial % 2 == 0) {
                    a = measure(SAMPLES, WORKERS, true, pool);
                    b = measure(SAMPLES, WORKERS, false, pool);
                } else {
                    b = measure(SAMPLES, WORKERS, false, pool);
                    a = measure(SAMPLES, WORKERS, true, pool);
                }
                shared[trial] = a.ns();
                local[trial] = b.ns();
                System.out.println("Trial " + (trial + 1));
                print("Shared PRNG", a);
                print("Independent PRNG", b);
                System.out.printf("Trial state-isolation speedup: %.4fx%n", (double) a.ns() / b.ns());
            }
        }
        System.out.printf("Shared PRNG median wall time: %.6f ms%n", PracticeSupport.ms(PracticeSupport.median(shared)));
        System.out.printf("Independent PRNG median wall time: %.6f ms%n", PracticeSupport.ms(PracticeSupport.median(local)));
        System.out.printf("Speedup from PRNG state isolation (ratio of medians): %.4fx%n",
                (double) PracticeSupport.median(shared) / PracticeSupport.median(local));
        System.out.println("Raw shared wall times (ns): " + Arrays.toString(shared));
        System.out.println("Raw independent wall times (ns): " + Arrays.toString(local));
        System.out.println("Different seeded streams need not produce equal pi estimates; no statistical-accuracy threshold is asserted.");
        System.out.println("This compares state ownership/locking, not isolated monitor latency; all samples are actually evaluated.");
    }
}