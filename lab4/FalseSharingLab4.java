/**
 * Lab 4: Memory Hierarchy, Cache Coherency, and False Sharing.
 *
 * Three variants of a counter benchmark, P in {1,2,4,8,16}:
 *   unpadded - adjacent volatile longs share a 64-byte cache line (false sharing).
 *   padded   - counters spaced by 8 longs = 64 bytes (cache line isolation).
 *   local    - thread-local register accumulator, single shared write at the end.
 *
 * 'volatile' forces a real memory read-modify-write on every iteration so the
 * JIT cannot collapse the loop into one addition.
 */
public class FalseSharingLab4 {

    static final long ITERATIONS = 100_000_000L;
    static final int STRIDE = 8; // 8 longs * 8 bytes = 64 bytes = 1 cache line

    static class UnpaddedCounters {
        volatile long[] values;
        UnpaddedCounters(int n) { values = new long[n]; }
    }

    static class PaddedCounters {
        volatile long[] values;
        PaddedCounters(int n) { values = new long[n * STRIDE]; }
    }

    interface Bench {
        void run(int tid) throws InterruptedException;
    }

    static long timeAll(int numThreads, Bench b) throws InterruptedException {
        Thread[] threads = new Thread[numThreads];
        long t0 = System.nanoTime();
        for (int t = 0; t < numThreads; t++) {
            final int tid = t;
            threads[t] = new Thread(() -> {
                try { b.run(tid); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            });
            threads[t].start();
        }
        for (Thread t : threads) t.join();
        return (System.nanoTime() - t0) / 1_000_000;
    }

    static long runUnpadded(int threads) throws InterruptedException {
        UnpaddedCounters c = new UnpaddedCounters(threads);
        return timeAll(threads, tid -> {
            for (long i = 0; i < ITERATIONS; i++) c.values[tid]++;
        });
    }

    static long runPadded(int threads) throws InterruptedException {
        PaddedCounters c = new PaddedCounters(threads);
        return timeAll(threads, tid -> {
            int idx = tid * STRIDE;
            for (long i = 0; i < ITERATIONS; i++) c.values[idx]++;
        });
    }

    static long runLocalAccumulator(int threads) throws InterruptedException {
        long[] results = new long[threads];
        return timeAll(threads, tid -> {
            long local = 0; // CPU register / L1-resident scalar
            for (long i = 0; i < ITERATIONS; i++) local++;
            results[tid] = local; // exactly one shared write per thread
        });
    }

    public static void main(String[] args) throws Exception {
        int[] threadCounts = {1, 2, 4, 8, 16};
        int trials = 2;

        // Warm-up JIT (results discarded)
        runUnpadded(2); runPadded(2); runLocalAccumulator(2);

        System.out.println("# False sharing benchmark: " + ITERATIONS + " volatile increments per thread");
        System.out.println("variant,threads,run,time_ms");
        for (int p : threadCounts) {
            for (int r = 0; r < trials; r++) {
                System.out.printf("unpadded,%d,%d,%d%n", p, r, runUnpadded(p));
                System.out.printf("padded,%d,%d,%d%n", p, r, runPadded(p));
                System.out.printf("local,%d,%d,%d%n", p, r, runLocalAccumulator(p));
            }
        }
    }
}
