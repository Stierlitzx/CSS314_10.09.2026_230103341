import java.nio.charset.StandardCharsets;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.SplittableRandom;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/** Independent support for the 22 October 14:30 practice (Java 21, no dependencies). */
final class PracticeSupport {
    static final long STUDENT_ID = 230103341L;
    static volatile double blackhole;

    private PracticeSupport() { }

    static void header(String title) {
        Locale.setDefault(Locale.ROOT);
        System.out.println(title);
        System.out.println("Session: Practice, 2026-10-22 14:30 (label, not execution timestamp)");
        System.out.println("Student ID: " + STUDENT_ID);
        System.out.println("Recorded at: " + ZonedDateTime.now());
        System.out.printf("OS/platform: %s %s / %s%n", System.getProperty("os.name"),
                System.getProperty("os.version"), System.getProperty("os.arch"));
        System.out.println("CPU/platform: " + System.getenv().getOrDefault("PROCESSOR_IDENTIFIER", "unknown"));
        System.out.println("Logical cores: " + Runtime.getRuntime().availableProcessors());
        System.out.println("Java: " + System.getProperty("java.runtime.version"));
    }

    static double[] data(int size) {
        double[] values = new double[size];
        SplittableRandom random = new SplittableRandom(STUDENT_ID);
        for (int i = 0; i < size; i++) values[i] = random.nextDouble();
        return values;
    }

    static int boundary(int index, int size, int workers) {
        return (int) ((long) index * size / workers);
    }

    static double ms(long ns) { return ns / 1_000_000.0; }

    static long median(long[] samples) {
        long[] sorted = samples.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    static void closeEnough(double actual, double expected) {
        if (!Double.isFinite(actual) || Math.abs(actual - expected) > Math.max(1, Math.abs(expected)) * 1e-10) {
            throw new AssertionError("Expected " + expected + ", got " + actual);
        }
    }

    static ThreadPoolExecutor pool(int workers) throws Exception {
        if (workers < 1) throw new IllegalArgumentException("Positive worker count required");
        ThreadPoolExecutor pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(workers);
        try {
            pool.prestartAllCoreThreads();
            CountDownLatch ready = new CountDownLatch(workers);
            CountDownLatch release = new CountDownLatch(1);
            try {
                for (int i = 0; i < workers; i++) {
                    pool.submit(() -> {
                        ready.countDown();
                        release.await();
                        return null;
                    });
                }
                if (!ready.await(30, TimeUnit.SECONDS)) throw new IllegalStateException("Pool startup timed out");
            } finally {
                release.countDown();
            }
            return pool;
        } catch (Exception error) {
            pool.shutdownNow();
            throw error;
        }
    }

    static <T> List<T> invoke(ThreadPoolExecutor pool, List<Callable<T>> jobs) throws Exception {
        List<Future<T>> futures = new ArrayList<>(jobs.size());
        for (Callable<T> job : jobs) futures.add(pool.submit(job));
        List<T> values = new ArrayList<>(jobs.size());
        for (Future<T> future : futures) values.add(future.get());
        return values;
    }

    static int physicalCores() throws Exception {
        String override = System.getProperty("physical.cores");
        if (override != null) {
            int count = Integer.parseInt(override);
            if (count < 1) throw new IllegalArgumentException("physical.cores must be positive");
            return count;
        }
        if (!System.getProperty("os.name").startsWith("Windows")) {
            throw new IllegalStateException("Supply verified physical count with -Dphysical.cores=N");
        }
        Process process = new ProcessBuilder("powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum")
                .redirectErrorStream(true).start();
        if (!process.waitFor(30, TimeUnit.SECONDS)) {
            process.destroyForcibly();
            throw new IllegalStateException("Physical-core detection timed out");
        }
        String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
        if (process.exitValue() != 0) throw new IllegalStateException(output);
        int count = Integer.parseInt(output);
        if (count < 1) throw new IllegalStateException("Invalid physical-core count");
        return count;
    }
}