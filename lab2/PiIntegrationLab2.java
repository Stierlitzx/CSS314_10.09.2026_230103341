import java.util.concurrent.ForkJoinPool;
import java.util.stream.LongStream;

/**
 * Lab 2: Numerical Integration (Pi Approximation) & Parallel Reductions.
 *
 * Modes:
 *   race     - Task 2.1: naive unsynchronized shared accumulator, P in {1,2,4,8}.
 *   critical - Task 2.2: synchronized critical section at N = 1,000,000 steps.
 *   scale    - Tasks 2.3/2.4: parallel reduction strong scaling, P in {1,2,4,8,16},
 *              5 trials; computes speedup S(P) and efficiency E(P).
 *   all      - run everything.
 */
public class PiIntegrationLab2 {

    static final long N = 100_000_000L;
    static final double STEP = 1.0 / N;
    static final long N_CRITICAL = 1_000_000L;

    // ---------------- Variant A: naive unsynchronized race ----------------
    static double runNaiveRace(long n, int threads) throws InterruptedException {
        double[] sharedSum = new double[1];
        Thread[] pool = new Thread[threads];
        long chunkSize = n / threads;
        for (int t = 0; t < threads; t++) {
            final long start = t * chunkSize;
            final long end = (t == threads - 1) ? n : start + chunkSize;
            pool[t] = new Thread(() -> {
                for (long i = start; i < end; i++) {
                    double x = (i + 0.5) * (1.0 / n);
                    sharedSum[0] += 4.0 / (1.0 + x * x); // unprotected shared write
                }
            });
            pool[t].start();
        }
        for (Thread t : pool) t.join();
        return sharedSum[0] * (1.0 / n);
    }

    // ---------------- Variant B: synchronized critical section ----------------
    static double runCriticalSection(long n, int threads) throws InterruptedException {
        final Object lock = new Object();
        double[] sharedSum = new double[1];
        Thread[] pool = new Thread[threads];
        long chunkSize = n / threads;
        for (int t = 0; t < threads; t++) {
            final long start = t * chunkSize;
            final long end = (t == threads - 1) ? n : start + chunkSize;
            pool[t] = new Thread(() -> {
                for (long i = start; i < end; i++) {
                    double x = (i + 0.5) * (1.0 / n);
                    double term = 4.0 / (1.0 + x * x);
                    synchronized (lock) { // emulates #pragma omp critical
                        sharedSum[0] += term;
                    }
                }
            });
            pool[t].start();
        }
        for (Thread t : pool) t.join();
        return sharedSum[0] * (1.0 / n);
    }

    /** Serial baseline (single-thread equivalent of the reduction). */
    static double runSerial(long n) {
        double total = 0.0;
        for (long i = 0; i < n; i++) {
            double x = (i + 0.5) * (1.0 / n);
            total += 4.0 / (1.0 + x * x);
        }
        return total * (1.0 / n);
    }

    // ---------------- Variant C: parallel reduction ----------------
    /** Emulates #pragma omp parallel for reduction(+:sum) on a pool of P threads. */
    static double runParallelReduction(ForkJoinPool pool) throws Exception {
        double sum = pool.submit(() ->
                LongStream.range(0, N).parallel()
                        .mapToDouble(i -> {
                            double x = (i + 0.5) * STEP;
                            return 4.0 / (1.0 + x * x);
                        })
                        .sum() // tree reduction over thread-private partial sums
        ).get();
        return sum * STEP;
    }

    // ---------------- Tasks ----------------

    static void task21() throws Exception {
        System.out.println("# Task 2.1: Race Condition Quantification (N = " + N + ")");
        System.out.println("threads,pi,abs_error,time_ms");
        for (int p : new int[]{1, 2, 4, 8}) {
            long t0 = System.nanoTime();
            double pi = runNaiveRace(N, p);
            double ms = (System.nanoTime() - t0) / 1e6;
            System.out.printf("%d,%.12f,%.6e,%.1f%n", p, pi, Math.abs(pi - Math.PI), ms);
        }
        System.out.println();
    }

    static void task22() throws Exception {
        System.out.println("# Task 2.2: Critical Section Overhead (N = " + N_CRITICAL + ")");
        runSerial(N_CRITICAL); // warm-up
        long t0 = System.nanoTime();
        double piSerial = runSerial(N_CRITICAL);
        double msSerial = (System.nanoTime() - t0) / 1e6;
        System.out.printf("# serial baseline: %.1f ms (pi=%.12f)%n", msSerial, piSerial);
        System.out.println("threads,pi,abs_error,time_ms,overhead_pct");
        for (int p : new int[]{1, 2, 4, 8}) {
            runCriticalSection(N_CRITICAL, p); // warm-up
            long t1 = System.nanoTime();
            double pi = runCriticalSection(N_CRITICAL, p);
            double ms = (System.nanoTime() - t1) / 1e6;
            double overhead = (ms - msSerial) / msSerial * 100.0;
            System.out.printf("%d,%.12f,%.6e,%.1f,%.1f%n", p, pi, Math.abs(pi - Math.PI), ms, overhead);
        }
        System.out.println();
    }

    static void task23() throws Exception {
        System.out.println("# Task 2.3/2.4: Strong Scaling of Parallel Reduction (N = " + N + ")");
        int trials = 5;
        ForkJoinPool warm = new ForkJoinPool(2);
        runParallelReduction(warm); // JIT warm-up
        warm.shutdown();

        double t1 = -1;
        System.out.println("threads,avg_time_ms,speedup,efficiency,pi,abs_error");
        for (int p : new int[]{1, 2, 4, 8, 16}) {
            ForkJoinPool pool = new ForkJoinPool(p);
            double total = 0, pi = 0;
            for (int t = 0; t < trials; t++) {
                long t0 = System.nanoTime();
                pi = runParallelReduction(pool);
                total += (System.nanoTime() - t0) / 1e6;
            }
            pool.shutdown();
            double avgMs = total / trials;
            if (p == 1) t1 = avgMs;
            double speedup = t1 / avgMs;
            double efficiency = speedup / p;
            System.out.printf("%d,%.2f,%.3f,%.3f,%.12f,%.6e%n",
                    p, avgMs, speedup, efficiency, pi, Math.abs(pi - Math.PI));
        }
        System.out.println();
    }

    public static void main(String[] args) throws Exception {
        String mode = args.length > 0 ? args[0] : "all";
        switch (mode) {
            case "race" -> task21();
            case "critical" -> task22();
            case "scale" -> task23();
            case "all" -> {
                task21();
                task22();
                task23();
            }
            default -> System.err.println("Unknown mode: " + mode);
        }
    }
}
