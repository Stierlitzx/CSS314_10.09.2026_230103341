import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

/** Task 2: strong scaling of the Task 1 summation, not its two-pass statistics. */
public class Task2 {
    static final int TRIALS = 7;

    record Measurement(long nanoseconds, double sum) { }

    static Measurement parallelSum(double[] values, int workers, ThreadPoolExecutor pool)
            throws Exception {
        long start = System.nanoTime();
        List<Future<Double>> partials = new ArrayList<>(workers);
        for (int worker = 0; worker < workers; worker++) {
            int from = BenchmarkSupport.boundary(worker, values.length, workers);
            int to = BenchmarkSupport.boundary(worker + 1, values.length, workers);
            partials.add(pool.submit(() -> BenchmarkSupport.sum(values, from, to)));
        }
        double sum = 0.0;
        for (Future<Double> partial : partials) sum += partial.get();
        return new Measurement(System.nanoTime() - start, sum);
    }

    static double amdahl(int workers) {
        return 1.0 / (0.05 + 0.95 / workers);
    }

    public static void main(String[] args) throws Exception {
        BenchmarkSupport.header("TASK 2 - Multi-core summation versus Amdahl's Law");
        double[] values = BenchmarkSupport.data(BenchmarkSupport.ARRAY_SIZE);
        for (int i = 0; i < 5; i++) {
            BenchmarkSupport.blackhole = BenchmarkSupport.sum(values, 0, values.length);
        }
        double expected = BenchmarkSupport.sum(values, 0, values.length);
        int cores = Runtime.getRuntime().availableProcessors();
        int[] workerCounts = {1, 2, 4, cores, cores * 2};
        long[][] samples = new long[workerCounts.length][TRIALS];
        long[] medians = new long[workerCounts.length];
        for (int index = 0; index < workerCounts.length; index++) {
            int workers = workerCounts[index];
            try (ThreadPoolExecutor pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(workers)) {
                BenchmarkSupport.startPool(pool, workers);
                for (int warmup = 0; warmup < 4; warmup++) parallelSum(values, workers, pool);
                for (int trial = 0; trial < TRIALS; trial++) {
                    Measurement result = parallelSum(values, workers, pool);
                    samples[index][trial] = result.nanoseconds();
                    BenchmarkSupport.checkClose(result.sum(), expected);
                    BenchmarkSupport.blackhole = result.sum();
                }
            }
            medians[index] = BenchmarkSupport.median(samples[index]);
        }
        System.out.printf("Elements: %,d float64, seed: %d%n", values.length, BenchmarkSupport.STUDENT_ID);
        System.out.println("Timing: warmed persistent pool; submission + sum + merge + wait included.");
        System.out.println("Generation, pool startup, warmup and result checks excluded. Median of 7 trials.");
        System.out.printf("%-7s | %16s | %10s | %19s%n",
                "Cores", "Wall Time (ms)", "Speedup", "Amdahl Upper Bound");
        for (int index = 0; index < workerCounts.length; index++) {
            System.out.printf("%-7d | %16.6f | %9.4fx | %18.4fx%n", workerCounts[index],
                    BenchmarkSupport.milliseconds(medians[index]), (double) medians[0] / medians[index],
                    amdahl(workerCounts[index]));
        }
        System.out.println("Raw wall times (ns):");
        for (int index = 0; index < workerCounts.length; index++) {
            System.out.printf("workers=%d %s%n", workerCounts[index], java.util.Arrays.toString(samples[index]));
        }
        System.out.printf("Verified sum (floating-point tolerance 1e-11 relative): %.15f%n", expected);
        System.out.println("T(1) is one pool worker doing SUM ONLY; Task 1 also computes standard deviation.");
        System.out.println("Amdahl model: 1/(0.05 + 0.95/p); idealized P=0.95, not a fitted hardware limit.");
        System.out.println("Cores column means worker count; 2*logical cores is deliberate oversubscription.");
    }
}