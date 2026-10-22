import java.nio.charset.StandardCharsets;
import java.time.ZonedDateTime;
import java.util.Arrays;
import java.util.Locale;
import java.util.SplittableRandom;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/** Shared, dependency-free support for the Java 21 lecture benchmarks. */
final class BenchmarkSupport {
    static final long STUDENT_ID = 230103341L;
    static final int ARRAY_SIZE = 20_000_000;
    static volatile double blackhole;

    private BenchmarkSupport() { }

    static void header(String task) {
        Locale.setDefault(Locale.ROOT);
        System.out.println(task);
        System.out.println("Student ID: " + STUDENT_ID);
        System.out.println("Recorded at: " + ZonedDateTime.now());
        System.out.printf("OS: %s %s (%s)%n", System.getProperty("os.name"),
                System.getProperty("os.version"), System.getProperty("os.arch"));
        System.out.println("CPU/platform: " + System.getenv().getOrDefault(
                "PROCESSOR_IDENTIFIER", System.getProperty("os.arch")));
        System.out.println("Java: " + System.getProperty("java.runtime.version"));
        System.out.println("Logical cores: " + Runtime.getRuntime().availableProcessors());
    }

    static double[] data(int size) {
        double[] values = new double[size];
        SplittableRandom random = new SplittableRandom(STUDENT_ID);
        for (int i = 0; i < size; i++) values[i] = random.nextDouble();
        return values;
    }

    static double sum(double[] values, int start, int end) {
        double total = 0.0;
        for (int i = start; i < end; i++) total += values[i];
        return total;
    }

    record Statistics(double sum, double standardDeviation) { }

    /** Two serial passes; population standard deviation (ddof=0). */
    static Statistics statistics(double[] values) {
        if (values.length == 0) throw new IllegalArgumentException("Empty array");
        double total = sum(values, 0, values.length);
        double mean = total / values.length;
        double squaredDifferences = 0.0;
        for (double value : values) {
            double difference = value - mean;
            squaredDifferences += difference * difference;
        }
        return new Statistics(total, Math.sqrt(squaredDifferences / values.length));
    }

    static int boundary(int part, int size, int workers) {
        return (int) ((long) part * size / workers);
    }

    static double milliseconds(long nanoseconds) {
        return nanoseconds / 1_000_000.0;
    }

    static long median(long[] samples) {
        long[] sorted = samples.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    static void checkClose(double actual, double expected) {
        double tolerance = Math.max(1.0, Math.abs(expected)) * 1e-11;
        if (!Double.isFinite(actual) || Math.abs(actual - expected) > tolerance) {
            throw new AssertionError("Expected " + expected + ", got " + actual);
        }
    }

    /** Prestart and touch every worker, outside measured dispatch intervals. */
    static void startPool(ThreadPoolExecutor pool, int workers) throws Exception {
        pool.prestartAllCoreThreads();
        CountDownLatch ready = new CountDownLatch(workers);
        CountDownLatch release = new CountDownLatch(1);
        for (int i = 0; i < workers; i++) {
            pool.submit(() -> {
                ready.countDown();
                release.await();
                return null;
            });
        }
        if (!ready.await(30, TimeUnit.SECONDS)) {
            release.countDown();
            throw new IllegalStateException("Workers did not start");
        }
        release.countDown();
    }

    /** Never infer physical cores from the logical count (SMT is not always 2). */
    static int physicalCores() throws Exception {
        String override = System.getProperty("physical.cores");
        if (override != null) {
            int cores = Integer.parseInt(override);
            if (cores < 1) throw new IllegalArgumentException("physical.cores must be positive");
            return cores;
        }
        if (!System.getProperty("os.name").startsWith("Windows")) {
            throw new IllegalStateException("Supply verified physical count: -Dphysical.cores=N");
        }
        Process process = new ProcessBuilder("powershell.exe", "-NoProfile", "-NonInteractive",
                "-Command", "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum")
                .redirectErrorStream(true).start();
        if (!process.waitFor(30, TimeUnit.SECONDS)) {
            process.destroyForcibly();
            throw new IllegalStateException("Physical-core detection timed out");
        }
        String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
        if (process.exitValue() != 0) throw new IllegalStateException(output);
        int cores = Integer.parseInt(output);
        if (cores < 1) throw new IllegalStateException("Invalid physical-core count: " + output);
        return cores;
    }
}