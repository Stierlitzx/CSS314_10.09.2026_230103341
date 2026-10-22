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
        for (int workers : new int[]{1, Task6.WORKERS}) {
            for (int iterations : new int[]{0, 10_003}) {
                Task6.Measurement result = Task6.measure(workers, iterations, true);
                check(result.actual() == workers * iterations && result.lostPercent() == 0,
                        "Atomic counter exactness and reset");
            }
        }
        check(Task6.measure(1, 10_003, false).actual() == 10_003, "Single-writer plain counter");
        check(Task6.measure(Task6.WORKERS, 0, false).actual() == 0, "Empty plain workload");
        // Never assert that an unsynchronized run MUST lose updates: scheduling can mask races.
        BenchmarkSupport.checkClose(new Task6.Measurement(1, 6_000_000, 8_000_000).lostPercent(), 25);
        check(new Task6.Measurement(1, 8_000_000, 8_000_000).lostPercent() == 0, "No-loss formula");
        check(new Task6.Measurement(1, 0, 8_000_000).lostPercent() == 100, "Complete-loss formula");
        try {
            Task6.measure(0, 1, true);
            throw new AssertionError("Invalid worker count accepted");
        } catch (IllegalArgumentException expected) {
            checks++;
        }
        try {
            Task6.measure(4, Integer.MAX_VALUE, true);
            throw new AssertionError("Overflowing workload accepted");
        } catch (ArithmeticException expected) {
            checks++;
        }
        for (int size : new int[]{0, 1, 3, 101, 1_003}) {
            double[] input = BenchmarkSupport.data(size);
            for (int workers : new int[]{1, 2, 8}) {
                try (ThreadPoolExecutor pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(workers)) {
                    BenchmarkSupport.startPool(pool, workers);
                    for (Task7.Kernel kind : Task7.Kernel.values()) {
                        double[] output = new double[size];
                        Arrays.fill(output, Double.NaN);
                        Task7.measure(input, output, workers, pool, kind);
                        Task7.validate(input, output, kind, true);
                        check(Arrays.stream(output).allMatch(Double::isFinite), "Task 7 full partition coverage");
                    }
                }
            }
        }
        BenchmarkSupport.checkClose(Task7.speedup(120, 30), 4);
        BenchmarkSupport.checkClose(Task7.efficiency(120, 30, 8), 0.5);
        BenchmarkSupport.checkClose(Task7.usefulGBps(30_000_000, 10_000_000), 48);
        check(Task7.scalingStalled(100, 100, 12), "Copy plateau detection");
        check(Task7.scalingStalled(120, 100, 12), "Inclusive stall threshold");
        check(!Task7.scalingStalled(200, 100, 12), "Continued scaling detection");
        check(!Task7.scalingStalled(100, 100, 2), "No 2-to-max interval");
        double[] special = {-0.0, 0.0, 0.125, 0.5, 0.999999};
        double[] copied = new double[special.length];
        Task7.kernel(special, copied, 0, special.length, Task7.Kernel.COPY);
        Task7.validate(special, copied, Task7.Kernel.COPY, true);
        check(Double.doubleToRawLongBits(copied[0]) == Double.doubleToRawLongBits(-0.0), "Copy signed zero");
        try {
            Task7.validate(new double[]{0.5}, new double[]{Double.NaN}, Task7.Kernel.COMPUTE, true);
            throw new AssertionError("Missing compute output accepted");
        } catch (AssertionError expected) {
            if (expected.getMessage().equals("Missing compute output accepted")) throw expected;
            checks++;
        }
        System.out.println("PASS: " + checks + " grouped checks; numerical assertions also passed.");
    }
}