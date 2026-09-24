import java.util.concurrent.atomic.AtomicInteger;

/**
 * Lab 3: Work-Sharing & Loop Scheduling Policies (Mandelbrot Fractal).
 *
 * Modes:
 *   matrix    - Tasks 3.1/3.2: 4x4 sweep of threads {2,4,8,16} x chunks {1,16,64,256}
 *               for the dynamic scheduler plus static baseline, 3 runs per cell.
 *   imbalance - Task 3.4: per-thread iteration counts and load imbalance metric
 *               for static vs. dynamic scheduling (P = 4 and P = 8).
 */
public class MandelbrotSchedulerLab3 {

    static final int WIDTH = 1920, HEIGHT = 1080, MAX_ITER = 1000;
    static final int[][] IMAGE = new int[HEIGHT][WIDTH];

    static int computePixel(int px, int py) {
        double x0 = (px - WIDTH / 2.0) * 4.0 / WIDTH;
        double y0 = (py - HEIGHT / 2.0) * 4.0 / HEIGHT;
        double x = 0.0, y = 0.0;
        int iter = 0;
        while (x * x + y * y <= 4.0 && iter < MAX_ITER) {
            double temp = x * x - y * y + x0;
            y = 2.0 * x * y + y0;
            x = temp;
            iter++;
        }
        return iter;
    }

    /** Renders rows [startRow, endRow). Returns total escape iterations (work metric). */
    static long renderRows(int startRow, int endRow) {
        long work = 0;
        for (int y = startRow; y < endRow; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int it = computePixel(x, y);
                IMAGE[y][x] = it;
                work += it;
            }
        }
        return work;
    }

    // ---------------- Static scheduling (schedule(static)) ----------------
    /** Contiguous block partition of rows; records per-thread work if workOut != null. */
    static long runStatic(int numThreads, long[] workOut) throws InterruptedException {
        Thread[] threads = new Thread[numThreads];
        int block = (HEIGHT + numThreads - 1) / numThreads;
        long t0 = System.nanoTime();
        for (int t = 0; t < numThreads; t++) {
            final int tid = t;
            final int start = t * block;
            final int end = Math.min(start + block, HEIGHT);
            threads[t] = new Thread(() -> {
                if (start < end) {
                    long w = renderRows(start, end);
                    if (workOut != null) workOut[tid] = w;
                }
            });
            threads[t].start();
        }
        for (Thread t : threads) t.join();
        return (System.nanoTime() - t0) / 1_000_000;
    }

    // ---------------- Dynamic scheduling (schedule(dynamic, chunk)) ----------------
    /** Atomic work dispenser; records per-thread work if workOut != null. */
    static long runDynamic(int numThreads, int chunkSize, long[] workOut) throws InterruptedException {
        AtomicInteger workQueue = new AtomicInteger(0); // dynamic row dispenser
        Thread[] threads = new Thread[numThreads];
        long t0 = System.nanoTime();
        for (int t = 0; t < numThreads; t++) {
            final int tid = t;
            threads[t] = new Thread(() -> {
                long w = 0;
                int startRow;
                while ((startRow = workQueue.getAndAdd(chunkSize)) < HEIGHT) {
                    int endRow = Math.min(startRow + chunkSize, HEIGHT);
                    w += renderRows(startRow, endRow);
                }
                if (workOut != null) workOut[tid] = w;
            });
            threads[t].start();
        }
        for (Thread t : threads) t.join();
        return (System.nanoTime() - t0) / 1_000_000;
    }

    // ---------------- Tasks ----------------

    /** Tasks 3.1/3.2: 4x4 parameter sweep, 3 runs per cell. */
    static void matrix() throws Exception {
        int[] threadCounts = {2, 4, 8, 16};
        int[] chunkSizes = {1, 16, 64, 256};
        int runs = 3;

        runDynamic(2, 64, null); // JIT warm-up

        System.out.println("# Static scheduling baseline (block = HEIGHT/P)");
        System.out.println("scheduler,threads,chunk,run,time_ms");
        for (int p : threadCounts) {
            for (int r = 0; r < runs; r++) {
                long ms = runStatic(p, null);
                System.out.printf("static,%d,0,%d,%d%n", p, r, ms);
            }
        }
        System.out.println();
        System.out.println("# Dynamic scheduling sweep");
        System.out.println("scheduler,threads,chunk,run,time_ms");
        for (int p : threadCounts) {
            for (int c : chunkSizes) {
                for (int r = 0; r < runs; r++) {
                    long ms = runDynamic(p, c, null);
                    System.out.printf("dynamic,%d,%d,%d,%d%n", p, c, r, ms);
                }
            }
        }
    }

    /** Task 3.4: per-thread work distribution and imbalance metric. */
    static void imbalance() throws Exception {
        System.out.println("# Task 3.4: Load imbalance = (Max - Min) / Average per-thread work");
        System.out.println("scheduler,threads,chunk,per_thread_work,imbalance");
        for (int p : new int[]{4, 8}) {
            long[] work = new long[p];
            runStatic(p, work);
            report("static", p, 0, work);
            for (int c : new int[]{1, 64}) {
                long[] wd = new long[p];
                runDynamic(p, c, wd);
                report("dynamic", p, c, wd);
            }
        }
    }

    static void report(String name, int p, int chunk, long[] work) {
        long max = Long.MIN_VALUE, min = Long.MAX_VALUE, sum = 0;
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < p; i++) {
            max = Math.max(max, work[i]);
            min = Math.min(min, work[i]);
            sum += work[i];
            sb.append(work[i]);
            if (i < p - 1) sb.append("; ");
        }
        sb.append("]");
        double avg = (double) sum / p;
        double imbalance = (max - min) / avg;
        System.out.printf("%s,%d,%d,\"%s\",%.4f%n", name, p, chunk, sb, imbalance);
    }

    public static void main(String[] args) throws Exception {
        String mode = args.length > 0 ? args[0] : "matrix";
        switch (mode) {
            case "matrix" -> matrix();
            case "imbalance" -> imbalance();
            default -> System.err.println("Unknown mode: " + mode);
        }
    }
}
