import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * Lab 1: The Fork-Join Model, Team Creation, and Thread Scoping.
 *
 * Modes:
 *   demo  - Task 1.1: spawn a team of 4 threads, each prints its logical rank
 *           and native OS thread id (run repeatedly to observe non-determinism).
 *   sweep - Task 1.2: measure fork/join wall-clock time for team sizes
 *           P in {1,2,4,8,16,32,64} (oversubscription sweep).
 *   load  - Task 1.3: CPU saturation - each thread computes 10,000,000
 *           floating-point square roots; throughput is measured for each P.
 */
public class ForkJoinLab1 {

    static final int WORK_PER_THREAD = 10_000_000;

    public static void main(String[] args) {
        String mode = args.length > 0 ? args[0] : "demo";
        switch (mode) {
            case "demo" -> demo(4);
            case "sweep" -> sweep();
            case "load" -> load();
            default -> System.err.println("Unknown mode: " + mode);
        }
    }

    /** Task 1.1 - fork-join lifecycle and thread identification. */
    static void demo(int targetThreads) {
        System.out.println("Master thread starting. Spawning thread team of size: " + targetThreads);
        // Emulates #pragma omp parallel num_threads(targetThreads)
        ForkJoinPool customPool = new ForkJoinPool(targetThreads);
        try {
            customPool.submit(() ->
                    IntStream.range(0, targetThreads).parallel().forEach(idx -> {
                        long osTid = Thread.currentThread().threadId();
                        String threadName = Thread.currentThread().getName();
                        boolean isMaster = idx == 0;
                        try {
                            Thread.sleep(idx % 3); // small stagger to expose scheduling order
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                        System.out.printf("[%s] Logical Rank: %d | Worker Thread: %s (OS ID: %d)%n",
                                isMaster ? "Master" : "Worker", idx, threadName, osTid);
                    })
            ).join(); // implicit barrier equivalent
        } finally {
            customPool.shutdown();
        }
        System.out.println("Parallel region closed. Execution returned to master thread.");
    }

    /** Task 1.2 - thread team instantiation/join cost vs. team size P. */
    static void sweep() {
        int[] sizes = {1, 2, 4, 8, 16, 32, 64};
        int trials = 10;
        // Warm-up so JIT cost does not pollute P=1
        for (int i = 0; i < 3; i++) forkJoinOnce(4);
        System.out.println("threads,avg_ms,min_ms");
        for (int p : sizes) {
            double totalMs = 0;
            double minMs = Double.MAX_VALUE;
            for (int t = 0; t < trials; t++) {
                long dt = forkJoinOnce(p);
                double ms = dt / 1e6;
                totalMs += ms;
                minMs = Math.min(minMs, ms);
            }
            System.out.printf("%d,%.3f,%.3f%n", p, totalMs / trials, minMs);
        }
    }

    /** Creates a pool of P threads, runs P no-op parallel tasks, joins. Returns ns. */
    static long forkJoinOnce(int p) {
        long t0 = System.nanoTime();
        ForkJoinPool pool = new ForkJoinPool(p);
        try {
            pool.submit(() -> IntStream.range(0, p).parallel().forEach(i -> {
            })).join();
        } finally {
            pool.shutdown();
        }
        return System.nanoTime() - t0;
    }

    /** Task 1.3 - CPU saturation: 10M sqrt per thread, sweep team size. */
    static void load() {
        int[] sizes = {1, 2, 4, 8, 16, 32, 64};
        // JIT warm-up
        for (int i = 0; i < 2; i++) cpuWorkload(4);
        System.out.println("threads,time_ms,throughput_Msqrt_per_s,checksum");
        for (int p : sizes) {
            long t0 = System.nanoTime();
            double checksum = cpuWorkload(p);
            double ms = (System.nanoTime() - t0) / 1e6;
            double totalOps = (double) p * WORK_PER_THREAD;
            double mops = totalOps / (ms * 1e6) * 1e3; // Msqrt/s
            System.out.printf("%d,%.1f,%.1f,%.2f%n", p, ms, mops, checksum);
        }
    }

    static double cpuWorkload(int p) {
        ForkJoinPool pool = new ForkJoinPool(p);
        try {
            return pool.submit(() ->
                    IntStream.range(0, p).parallel().mapToDouble(i -> {
                        double acc = 0.0;
                        for (int k = 0; k < WORK_PER_THREAD; k++) {
                            acc += Math.sqrt(k);
                        }
                        return acc;
                    }).sum()
            ).join();
        } finally {
            pool.shutdown();
        }
    }
}
