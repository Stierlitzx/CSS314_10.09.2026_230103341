import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

/** Task 5: measured, sampled crossover for y[i] = a*x[i] + b. */
public class Task5 {
    static final int[] SIZES = {100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000};
    static final int TRIALS = 15;
    static final double A = 2.5;
    static final double B = 1.25;

    static void vector(double[] x, double[] y, int from, int to) {
        for (int i = from; i < to; i++) y[i] = A * x[i] + B;
    }

    static long serial(double[] x, double[] y) {
        long start = System.nanoTime();
        vector(x, y, 0, x.length);
        return System.nanoTime() - start;
    }

    static long parallel(double[] x, double[] y, int workers, ThreadPoolExecutor pool) throws Exception {
        long start = System.nanoTime();
        List<Future<?>> futures = new ArrayList<>(workers);
        for (int worker = 0; worker < workers; worker++) {
            int from = BenchmarkSupport.boundary(worker, x.length, workers);
            int to = BenchmarkSupport.boundary(worker + 1, x.length, workers);
            futures.add(pool.submit(() -> vector(x, y, from, to)));
        }
        for (Future<?> future : futures) future.get();
        return System.nanoTime() - start;
    }

    static List<String> crossoverRanges(int[] sizes, long[] serialNs, long[] parallelNs) {
        if (sizes.length != serialNs.length || sizes.length != parallelNs.length) {
            throw new IllegalArgumentException("Mismatched measurement arrays");
        }
        List<String> ranges = new ArrayList<>();
        for (int i = 1; i < sizes.length; i++) {
            if (parallelNs[i - 1] > serialNs[i - 1] && parallelNs[i] < serialNs[i]) {
                ranges.add(String.format("%d < N <= %d", sizes[i - 1], sizes[i]));
            }
        }
        return ranges;
    }

    public static void main(String[] args) throws Exception {
        BenchmarkSupport.header("TASK 5 - Parallel overhead crossover");
        int workers = Runtime.getRuntime().availableProcessors();
        System.out.println("Operation: y[i] = 2.5*x[i] + 1.25, float64, seeded input.");
        System.out.println("Parallel worker count (all logical cores): " + workers);
        System.out.println("Persistent warmed pool; dispatch + operation + joins included in parallel timing.");
        System.out.println("Allocation, generation, first-touch, warmup and validation excluded from both timings.");
        System.out.println("15 paired trials; alternating serial/parallel order; median wall-clock times.");
        long[] serialMedians = new long[SIZES.length];
        long[] parallelMedians = new long[SIZES.length];
        long[][] serialRaw = new long[SIZES.length][TRIALS];
        long[][] parallelRaw = new long[SIZES.length][TRIALS];
        try (ThreadPoolExecutor pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(workers)) {
            BenchmarkSupport.startPool(pool, workers);
            double[] warmX = BenchmarkSupport.data(100_000);
            double[] warmY = new double[warmX.length];
            for (int i = 0; i < 200; i++) {
                serial(warmX, warmY);
                parallel(warmX, warmY, workers, pool);
                BenchmarkSupport.blackhole = warmY[i];
            }
            for (int index = 0; index < SIZES.length; index++) {
                int size = SIZES[index];
                double[] x = BenchmarkSupport.data(size);
                double[] serialY = new double[size];
                double[] parallelY = new double[size];
                Arrays.fill(serialY, Double.NaN);
                Arrays.fill(parallelY, Double.NaN);
                for (int warmup = 0; warmup < 6; warmup++) {
                    serial(x, serialY);
                    parallel(x, parallelY, workers, pool);
                }
                for (int trial = 0; trial < TRIALS; trial++) {
                    if (trial % 2 == 0) {
                        serialRaw[index][trial] = serial(x, serialY);
                        parallelRaw[index][trial] = parallel(x, parallelY, workers, pool);
                    } else {
                        parallelRaw[index][trial] = parallel(x, parallelY, workers, pool);
                        serialRaw[index][trial] = serial(x, serialY);
                    }
                    BenchmarkSupport.blackhole = serialY[size / 2] + parallelY[size / 2];
                }
                for (int i = 0; i < size; i++) {
                    double expected = A * x[i] + B;
                    BenchmarkSupport.checkClose(serialY[i], expected);
                    BenchmarkSupport.checkClose(parallelY[i], expected);
                }
                serialMedians[index] = BenchmarkSupport.median(serialRaw[index]);
                parallelMedians[index] = BenchmarkSupport.median(parallelRaw[index]);
            }
        }
        System.out.printf("%12s | %14s | %14s | %9s | %s%n",
                "N", "Serial (ms)", "Parallel (ms)", "Speedup", "Parallel status");
        for (int i = 0; i < SIZES.length; i++) {
            String status = parallelMedians[i] < serialMedians[i] ? "FASTER"
                    : parallelMedians[i] > serialMedians[i] ? "SLOWER" : "TIED";
            System.out.printf("%,12d | %14.6f | %14.6f | %8.4fx | %s%n", SIZES[i],
                    BenchmarkSupport.milliseconds(serialMedians[i]),
                    BenchmarkSupport.milliseconds(parallelMedians[i]),
                    (double) serialMedians[i] / parallelMedians[i], status);
        }
        List<String> ranges = crossoverRanges(SIZES, serialMedians, parallelMedians);
        if (ranges.isEmpty()) {
            System.out.println("No SLOWER -> FASTER crossover observed within the tested sizes.");
        } else {
            for (String range : ranges) System.out.println("Observed SLOWER -> FASTER bracket: " + range);
        }
        for (int i = 1; i < SIZES.length; i++) {
            if (parallelMedians[i - 1] < serialMedians[i - 1] && parallelMedians[i] > serialMedians[i]) {
                System.out.printf("Observed reversal FASTER -> SLOWER: %d < N <= %d%n", SIZES[i - 1], SIZES[i]);
            }
        }
        System.out.println("A bracket is an exact interval between sampled sizes, NOT an exact universal tipping N.");
        System.out.println("Cache state, bandwidth, JVM compilation and OS load can move or reverse the crossover.");
        System.out.println("Raw wall times (ns):");
        for (int i = 0; i < SIZES.length; i++) {
            System.out.printf("N=%d serial=%s%n", SIZES[i], Arrays.toString(serialRaw[i]));
            System.out.printf("N=%d parallel=%s%n", SIZES[i], Arrays.toString(parallelRaw[i]));
        }
        System.out.println("Validation: all serial and parallel output elements match y=a*x+b.");
    }
}