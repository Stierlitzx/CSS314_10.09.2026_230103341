import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

/**
 * Lab 5: Recursive Task-Based Parallelism (Parallel Merge Sort).
 *
 * Task 5.1: correctness verification (ascending-order assertion).
 * Task 5.2: sequential cutoff sweep K in {10,100,1000,10000,50000,100000}
 *           on 5,000,000 random integers.
 * Task 5.3: Work-Span analysis - measured T1 (sequential) vs. best parallel
 *           time, plus the theoretical model T1 = O(N log N), T_inf = O(N),
 *           P_theoretical = O(log N).
 */
public class ParallelMergeSortLab5 extends RecursiveAction {

    private final int[] array;
    private final int left, right;
    private final int cutoff;

    public ParallelMergeSortLab5(int[] array, int left, int right, int cutoff) {
        this.array = array;
        this.left = left;
        this.right = right;
        this.cutoff = cutoff;
    }

    @Override
    protected void compute() {
        // Sequential cutoff threshold (task granularity control)
        if (right - left + 1 <= cutoff) {
            Arrays.sort(array, left, right + 1);
            return;
        }
        int mid = left + (right - left) / 2;
        // Emulates: #pragma omp task  (left half)
        ParallelMergeSortLab5 leftTask = new ParallelMergeSortLab5(array, left, mid, cutoff);
        // Emulates: #pragma omp task  (right half)
        ParallelMergeSortLab5 rightTask = new ParallelMergeSortLab5(array, mid + 1, right, cutoff);
        // Emulates: spawn both + #pragma omp taskwait
        invokeAll(leftTask, rightTask);
        merge(mid);
    }

    private void merge(int mid) {
        int[] temp = Arrays.copyOfRange(array, left, right + 1);
        int i = 0, j = mid - left + 1, k = left;
        int midOffset = mid - left;
        int endOffset = right - left;
        while (i <= midOffset && j <= endOffset) {
            if (temp[i] <= temp[j]) array[k++] = temp[i++];
            else array[k++] = temp[j++];
        }
        while (i <= midOffset) array[k++] = temp[i++];
        while (j <= endOffset) array[k++] = temp[j++];
    }

    static boolean isSorted(int[] a) {
        for (int i = 1; i < a.length; i++) {
            if (a[i - 1] > a[i]) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        int size = 5_000_000;
        int[] data = new int[size];
        java.util.Random rng = new java.util.Random(42); // fixed seed = reproducible
        for (int i = 0; i < size; i++) data[i] = rng.nextInt(10_000_000);

        ForkJoinPool pool = new ForkJoinPool(); // defaults to all available cores

        // ---------- Task 5.1: correctness ----------
        int[] check = Arrays.copyOf(data, data.length);
        pool.invoke(new ParallelMergeSortLab5(check, 0, check.length - 1, 10_000));
        assert isSorted(check) : "Array is NOT sorted!";
        System.out.println("Task 5.1: output verified sorted in ascending order: " + isSorted(check));
        System.out.println();

        // ---------- Sequential baseline (Work T1 proxy) ----------
        // Warm-up
        int[] warm = Arrays.copyOf(data, data.length);
        Arrays.sort(warm);
        long t0 = System.nanoTime();
        int[] seq = Arrays.copyOf(data, data.length);
        Arrays.sort(seq);
        double seqMs = (System.nanoTime() - t0) / 1e6;
        System.out.printf("Sequential Arrays.sort baseline (T1): %.1f ms%n%n", seqMs);

        // ---------- Task 5.2: cutoff sweep ----------
        int[] cutoffs = {10, 100, 1_000, 10_000, 50_000, 100_000};
        int trials = 3;
        // Warm-up the ForkJoin machinery
        pool.invoke(new ParallelMergeSortLab5(Arrays.copyOf(data, data.length), 0, size - 1, 10_000));

        System.out.println("# Task 5.2: cutoff threshold sweep (N = " + size + ")");
        System.out.println("cutoff,avg_time_ms,min_time_ms,speedup_vs_sequential");
        double bestMs = Double.MAX_VALUE;
        for (int cut : cutoffs) {
            double total = 0, min = Double.MAX_VALUE;
            for (int t = 0; t < trials; t++) {
                int[] copy = Arrays.copyOf(data, data.length);
                long start = System.nanoTime();
                pool.invoke(new ParallelMergeSortLab5(copy, 0, copy.length - 1, cut));
                double ms = (System.nanoTime() - start) / 1e6;
                if (!isSorted(copy)) throw new AssertionError("sort failed at cutoff " + cut);
                total += ms;
                min = Math.min(min, ms);
            }
            double avg = total / trials;
            bestMs = Math.min(bestMs, avg);
            System.out.printf("%d,%.1f,%.1f,%.2f%n", cut, avg, min, seqMs / avg);
        }
        pool.shutdown();

        // ---------- Task 5.3: Work-Span model ----------
        System.out.println();
        System.out.println("# Task 5.3: Work-Span analysis");
        System.out.printf("Measured Work  T1 (sequential)      = %.1f ms%n", seqMs);
        System.out.printf("Measured best parallel time T_P     = %.1f ms%n", bestMs);
        System.out.printf("Empirical speedup                   = %.2fx%n", seqMs / bestMs);
        System.out.println("Theoretical:  Work  T1   = O(N log N)  (all comparisons + merges)");
        System.out.println("              Span  Tinf = O(N)        (root merge of N elements is sequential");
        System.out.println("                                   and dominates the critical path)");
        System.out.println("              Parallelism P_theoretical = T1/Tinf = O(log N) ~ "
                + (int) Math.round(Math.log(size) / Math.log(2)) + " for N = " + size);
    }
}
