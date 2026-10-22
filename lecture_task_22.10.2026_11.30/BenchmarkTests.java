import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

/** Fast regression checks; run with java -ea BenchmarkTests after compilation. */
public class BenchmarkTests {
    static int checks;

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
        checks++;
    }

    public static void main(String[] args) throws Exception {
        BenchmarkSupport.Statistics stats = BenchmarkSupport.statistics(new double[]{1, 2, 3, 4});
        BenchmarkSupport.checkClose(stats.sum(), 10);
        BenchmarkSupport.checkClose(stats.standardDeviation(), Math.sqrt(1.25));
        check(BenchmarkSupport.statistics(new double[]{7}).standardDeviation() == 0, "Singleton stddev");
        check(Arrays.equals(BenchmarkSupport.data(101), BenchmarkSupport.data(101)), "Seed repeatability");
        check(BenchmarkSupport.median(new long[]{8, 3, 1, 9, 4}) == 4, "Median");
        BenchmarkSupport.checkClose(Task2.amdahl(1), 1);
        BenchmarkSupport.checkClose(Task2.amdahl(4), 1 / (0.05 + 0.95 / 4));
        for (int size : new int[]{1, 3, 101, 1_003}) {
            double[] x = BenchmarkSupport.data(size);
            for (int workers : new int[]{1, 2, 4, 8}) {
                try (ThreadPoolExecutor pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(workers)) {
                    BenchmarkSupport.startPool(pool, workers);
                    BenchmarkSupport.checkClose(Task2.parallelSum(x, workers, pool).sum(),
                            BenchmarkSupport.sum(x, 0, size));
                    double[] y = new double[size];
                    Arrays.fill(y, Double.NaN);
                    Task5.parallel(x, y, workers, pool);
                    for (int i = 0; i < size; i++) BenchmarkSupport.checkClose(y[i], Task5.A * x[i] + Task5.B);
                    checks++;
                }
            }
        }
        for (int stride : new int[]{1, Task3.PADDED_STRIDE}) {
            Task3.Measurement result = Task3.measure(4, 10_001, stride);
            check(Arrays.stream(result.counts()).allMatch(value -> value == 10_001), "Counter correctness");
        }
        for (int strategy = 0; strategy < 3; strategy++) {
            Task4.Measurement result = Task4.measure(strategy != 0, strategy == 2 ? 5 : 1, false);
            check(Arrays.stream(result.itemCounts()).sum() == 1_000, "Scheduler item count");
            check(Arrays.stream(result.costsNs()).sum() == 950_000_000L, "Scheduler total requested cost");
            check(result.deltaNs() >= 0 && result.lastNs() <= result.wallNs(), "Completion delta");
            if (strategy == 0) {
                check(Arrays.equals(result.itemCounts(), new int[]{250, 250, 250, 250}), "Static blocks");
                check(Arrays.equals(result.costsNs(), new long[]{232_500_000, 242_500_000,
                        232_500_000, 242_500_000}), "Static cost distribution");
            }
        }
        check(Task5.crossoverRanges(new int[]{100, 1_000}, new long[]{5, 50}, new long[]{10, 25})
                .equals(List.of("100 < N <= 1000")), "Crossover bracket");
        check(Task5.crossoverRanges(new int[]{100, 1_000}, new long[]{5, 50}, new long[]{10, 100})
                .isEmpty(), "No crossover");
        check(Task5.crossoverRanges(new int[]{1, 2, 3, 4}, new long[]{5, 50, 5, 50},
                new long[]{10, 25, 10, 25}).size() == 2, "Multiple crossings");
        check(Task5.crossoverRanges(new int[]{1, 2}, new long[]{5, 50}, new long[]{5, 25})
                .isEmpty(), "Tie is not slower");
        System.out.println("PASS: " + checks + " grouped checks; numerical assertions also passed.");
    }
}