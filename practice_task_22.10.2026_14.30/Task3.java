import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.Callable;
import java.util.concurrent.ThreadPoolExecutor;

/** Work-efficient inclusive chunked scan: local scan, scalar chunk reduction, offsets. */
public class Task3 {
    static final int ELEMENTS = 2_000_000;
    static final int TRIALS = 9;

    static int[] data(int size) {
        int[] values = new int[size];
        SplittableRandom random = new SplittableRandom(PracticeSupport.STUDENT_ID);
        for (int i = 0; i < size; i++) values[i] = random.nextInt(1, 11);
        return values;
    }

    static long serial(int[] input, long[] output) {
        if (input.length != output.length) throw new IllegalArgumentException("Array length mismatch");
        long begin = System.nanoTime();
        long total = 0;
        for (int i = 0; i < input.length; i++) output[i] = total += input[i];
        return System.nanoTime() - begin;
    }

    static long parallel(int[] input, long[] output, int workers, ThreadPoolExecutor pool) throws Exception {
        if (input.length != output.length || workers < 1) throw new IllegalArgumentException("Invalid scan arguments");
        long begin = System.nanoTime();
        List<Callable<Long>> jobs = new ArrayList<>();
        for (int worker = 0; worker < workers; worker++) {
            int from = PracticeSupport.boundary(worker, input.length, workers);
            int to = PracticeSupport.boundary(worker + 1, input.length, workers);
            jobs.add(() -> {
                long local = 0;
                for (int i = from; i < to; i++) output[i] = local += input[i];
                return local;
            });
        }
        List<Long> totals = PracticeSupport.invoke(pool, jobs); // Barrier after local scans.
        long offset = 0;
        List<Callable<Long>> updates = new ArrayList<>();
        for (int worker = 0; worker < workers; worker++) {
            int from = PracticeSupport.boundary(worker, input.length, workers);
            int to = PracticeSupport.boundary(worker + 1, input.length, workers);
            long add = offset;
            offset += totals.get(worker);
            updates.add(() -> {
                for (int i = from; i < to; i++) output[i] += add;
                return 0L;
            });
        }
        PracticeSupport.invoke(pool, updates); // Barrier before output may be consumed.
        return System.nanoTime() - begin;
    }

    static void verify(long[] serial, long[] parallel) {
        if (serial.length != parallel.length) throw new AssertionError("Output lengths differ");
        if (serial.length > 0 && serial[serial.length - 1] != parallel[parallel.length - 1]) {
            throw new AssertionError("Final prefix totals differ");
        }
        // Do not rely on JVM -ea flags for the required verification.
        if (!Arrays.equals(serial, parallel)) throw new AssertionError("An intermediate prefix differs");
    }

    public static void main(String[] args) {
        try {
            run();
        } catch (Exception | AssertionError failure) {
            System.out.println("Verification Status: FAIL");
            failure.printStackTrace(System.err);
            System.exit(1);
        }
    }

    static void run() throws Exception {
        PracticeSupport.header("PRACTICE TASK 3 - Inclusive parallel prefix scan verification");
        int workers = Runtime.getRuntime().availableProcessors();
        int[] input = data(ELEMENTS);
        long[] serialOutput = new long[ELEMENTS];
        long[] parallelOutput = new long[ELEMENTS];
        System.out.printf("Input: %,d integers in [1,10], seed=%d; workers=%d; long prefix outputs.%n",
                ELEMENTS, PracticeSupport.STUDENT_ID, workers);
        System.out.println("Local inclusive scans -> serial exclusive scan of chunk totals -> parallel offset application.");
        System.out.println("O(N+p) work; two array passes, two dispatch barriers; not a logarithmic-step tree scan.");
        System.out.println("Nine alternating-order trials; allocation/generation/pool startup/validation excluded.");
        System.out.println("Parallel timing includes job creation, both passes, chunk-total reduction and future joins.");
        long[] serialTimes = new long[TRIALS];
        long[] parallelTimes = new long[TRIALS];
        try (ThreadPoolExecutor pool = PracticeSupport.pool(workers)) {
            for (int i = 0; i < 8; i++) {
                serial(input, serialOutput);
                parallel(input, parallelOutput, workers, pool);
            }
            for (int trial = 0; trial < TRIALS; trial++) {
                Arrays.fill(serialOutput, Long.MIN_VALUE);
                Arrays.fill(parallelOutput, Long.MIN_VALUE);
                if (trial % 2 == 0) {
                    serialTimes[trial] = serial(input, serialOutput);
                    parallelTimes[trial] = parallel(input, parallelOutput, workers, pool);
                } else {
                    parallelTimes[trial] = parallel(input, parallelOutput, workers, pool);
                    serialTimes[trial] = serial(input, serialOutput);
                }
                verify(serialOutput, parallelOutput);
            }
        }
        long a = PracticeSupport.median(serialTimes);
        long b = PracticeSupport.median(parallelTimes);
        System.out.println("Serial Time (ms) | Parallel Time (ms) | Speedup | Verification Status");
        System.out.printf("%.6f | %.6f | %.4fx | PASS%n", PracticeSupport.ms(a), PracticeSupport.ms(b), (double) a / b);
        System.out.printf("Final prefixes: serial=%d, parallel=%d (explicit equality check; mismatch exits 1).%n",
                serialOutput[ELEMENTS - 1], parallelOutput[ELEMENTS - 1]);
        System.out.println("All 2,000,000 prefixes verified on EVERY trial (not only the final element).");
        System.out.println("Raw serial wall times (ns): " + Arrays.toString(serialTimes));
        System.out.println("Raw parallel wall times (ns): " + Arrays.toString(parallelTimes));
    }
}