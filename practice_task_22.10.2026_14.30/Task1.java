import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.Callable;
import java.util.concurrent.ThreadPoolExecutor;

/** Row-contiguous versus row-strided traversal of a real double[][] matrix. */
public class Task1 {
    static final int DIMENSION = 8_000;
    static final int TRIALS = 5;

    record Measurement(long ns, double sum) { }

    static double[][] matrix(int rows, int columns) {
        double[][] matrix = new double[rows][columns];
        SplittableRandom random = new SplittableRandom(PracticeSupport.STUDENT_ID);
        for (double[] row : matrix) {
            for (int column = 0; column < columns; column++) row[column] = random.nextDouble();
        }
        return matrix;
    }

    static Measurement measure(double[][] matrix, int columns, int workers,
                               ThreadPoolExecutor pool, boolean rowMajor) throws Exception {
        long begin = System.nanoTime();
        List<Callable<Double>> jobs = new ArrayList<>();
        int dimension = rowMajor ? matrix.length : columns;
        for (int worker = 0; worker < workers; worker++) {
            int from = PracticeSupport.boundary(worker, dimension, workers);
            int to = PracticeSupport.boundary(worker + 1, dimension, workers);
            jobs.add(() -> {
                double sum = 0;
                if (rowMajor) {
                    for (int row = from; row < to; row++) {
                        double[] values = matrix[row];
                        for (int column = 0; column < columns; column++) sum += values[column];
                    }
                } else {
                    for (int column = from; column < to; column++) {
                        for (double[] row : matrix) sum += row[column];
                    }
                }
                return sum;
            });
        }
        double sum = 0;
        for (double partial : PracticeSupport.invoke(pool, jobs)) sum += partial;
        return new Measurement(System.nanoTime() - begin, sum);
    }

    public static void main(String[] args) throws Exception {
        PracticeSupport.header("PRACTICE TASK 1 - Row-major versus strided cache traversal");
        int workers = PracticeSupport.physicalCores();
        System.out.println("Physical-core workers: " + workers);
        System.out.printf("Matrix: %,d x %,d float64; seed: %d; payload: %,d bytes%n",
                DIMENSION, DIMENSION, PracticeSupport.STUDENT_ID, (long) DIMENSION * DIMENSION * Double.BYTES);
        System.out.println("Java double[][] has contiguous elements within each separately allocated row.");
        System.out.println("Mode A: slice rows, inner loop columns. Mode B: slice columns, inner loop rows.");
        System.out.println("Timing includes submission, traversal, joins and reduction; excludes allocation/generation and pool startup.");
        System.out.println("Five trials after warmup, alternating mode order; medians reported. No CPU affinity/cache flushing.");
        double[][] values = matrix(DIMENSION, DIMENSION);
        double expected = 0;
        for (double[] row : values) for (double value : row) expected += value;
        long[] rows = new long[TRIALS];
        long[] columns = new long[TRIALS];
        try (ThreadPoolExecutor pool = PracticeSupport.pool(workers)) {
            double[][] warm = matrix(101, 103);
            for (int i = 0; i < 12; i++) {
                measure(warm, 103, workers, pool, true);
                measure(warm, 103, workers, pool, false);
            }
            measure(values, DIMENSION, workers, pool, true);
            measure(values, DIMENSION, workers, pool, false);
            for (int trial = 0; trial < TRIALS; trial++) {
                Measurement a;
                Measurement b;
                if (trial % 2 == 0) {
                    a = measure(values, DIMENSION, workers, pool, true);
                    b = measure(values, DIMENSION, workers, pool, false);
                } else {
                    b = measure(values, DIMENSION, workers, pool, false);
                    a = measure(values, DIMENSION, workers, pool, true);
                }
                PracticeSupport.closeEnough(a.sum(), expected);
                PracticeSupport.closeEnough(b.sum(), expected);
                rows[trial] = a.ns();
                columns[trial] = b.ns();
                System.out.printf("trial=%d Mode A=%.6f ms (%d ns); Mode B=%.6f ms (%d ns); sums=%.12f / %.12f%n",
                        trial + 1, PracticeSupport.ms(a.ns()), a.ns(), PracticeSupport.ms(b.ns()), b.ns(), a.sum(), b.sum());
            }
        }
        long a = PracticeSupport.median(rows);
        long b = PracticeSupport.median(columns);
        System.out.printf("Mode A (Row-Major) median runtime: %.6f ms%n", PracticeSupport.ms(a));
        System.out.printf("Mode B (Column-Major / Strided) median runtime: %.6f ms%n", PracticeSupport.ms(b));
        System.out.printf("Cache penalty factor (Mode B / Mode A): %.4fx%n", (double) b / a);
        System.out.println("Raw Mode A wall times (ns): " + Arrays.toString(rows));
        System.out.println("Raw Mode B wall times (ns): " + Arrays.toString(columns));
        System.out.printf("Verification PASS: all sums match serial reference %.12f (relative tolerance 1e-10).%n", expected);
        System.out.println("Timing ratio is observed traversal cost, not a hardware-counter measurement of cache misses.");
    }
}