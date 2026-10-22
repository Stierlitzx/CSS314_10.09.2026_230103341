import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.concurrent.ThreadPoolExecutor;

/** Fast dependency-free correctness tests, not performance assertions. */
public class PracticeTests {
    static int checks;

    static void check(boolean condition, String label) {
        if (!condition) throw new AssertionError(label);
        checks++;
    }

    static long sequentialHits(int samples, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        long hits = 0;
        for (int i = 0; i < samples; i++) {
            double x = random.nextDouble();
            double y = random.nextDouble();
            if (x * x + y * y <= 1) hits++;
        }
        return hits;
    }

    public static void main(String[] args) throws Exception {
        for (int workers : new int[]{1, 2, 8}) {
            try (ThreadPoolExecutor pool = PracticeSupport.pool(workers)) {
                for (int[] shape : new int[][]{{0, 0}, {1, 1}, {3, 5}, {17, 19}}) {
                    double[][] matrix = Task1.matrix(shape[0], shape[1]);
                    double expected = 0;
                    for (double[] row : matrix) for (double value : row) expected += value;
                    PracticeSupport.closeEnough(Task1.measure(matrix, shape[1], workers, pool, true).sum(), expected);
                    PracticeSupport.closeEnough(Task1.measure(matrix, shape[1], workers, pool, false).sum(), expected);
                    checks++;
                }
                int samples = 10_003;
                Task2.Measurement shared = Task2.measure(samples, workers, true, pool);
                check(shared.hits() == sequentialHits(samples, PracticeSupport.STUDENT_ID), "Shared PRNG stream preserved");
                long independentHits = 0;
                for (int i = 0; i < workers; i++) {
                    int count = PracticeSupport.boundary(i + 1, samples, workers)
                            - PracticeSupport.boundary(i, samples, workers);
                    independentHits += sequentialHits(count, PracticeSupport.STUDENT_ID * 10 + i);
                }
                check(Task2.measure(samples, workers, false, pool).hits() == independentHits, "Independent seeds and partition");
                check(Task2.measure(samples, workers, false, pool).hits() == independentHits, "Fresh PRNG state per invocation");
                for (int size : new int[]{0, 1, 3, 101, 1_003}) {
                    int[] input = Task3.data(size);
                    check(Arrays.stream(input).allMatch(value -> value >= 1 && value <= 10), "Scan input bounds");
                    long[] serial = new long[size];
                    long[] parallel = new long[size];
                    Arrays.fill(parallel, Long.MIN_VALUE);
                    Task3.serial(input, serial);
                    Task3.parallel(input, parallel, workers, pool);
                    Task3.verify(serial, parallel);
                    long total = 0;
                    for (int i = 0; i < size; i++) {
                        total += input[i];
                        check(parallel[i] == total, "Every prefix exact");
                    }
                }
                byte[][] inputs = Task5.inputs(13);
                long expected = 0;
                for (byte[] input : inputs) expected += Task5.hash(input, 7);
                byte[] original = inputs[0].clone();
                check(Task5.measure(inputs, 7, pool).checksum() == expected, "Hash batch partition and reduction");
                check(Arrays.equals(inputs[0], original), "Hash task does not mutate shared input");
            }
        }
        try {
            Task3.verify(new long[]{1, 2}, new long[]{1, 3});
            throw new IllegalStateException("Final mismatch accepted");
        } catch (AssertionError expected) { checks++; }
        try {
            Task3.verify(new long[]{1, 3}, new long[]{2, 3});
            throw new IllegalStateException("Intermediate mismatch accepted");
        } catch (AssertionError expected) { checks++; }
        check(Arrays.equals(Task3.data(101), Task3.data(101)), "Integer seed repeatability");
        check(Arrays.equals(Task5.sizes(12), new int[]{1, 6, 12, 24, 48, 192, 256}), "Requested pool sizes");
        check(Task5.firstWorsening(new long[]{100, 50, 60, 40}) == 2, "First worsening even with reversal");
        check(Task5.firstWorsening(new long[]{100, 50, 50}) == -1, "Ties not worsening");
        check(Task5.firstWorsening(new long[]{100}) == -1, "No worsening case");
        check(PracticeSupport.median(new long[]{30, 10, 20}) == 20, "Median");
        Task4.selfTest();
        checks++;
        System.out.println("PASS: " + checks + " checks plus matrix/numerical assertions and real child-process IPC self-tests.");
    }
}