import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

/** Task 7: low versus high arithmetic intensity on the same large float64 input. */
public class Task7 {
    static final int ELEMENTS = 30_000_000;
    static final int TRIALS = 7;

    enum Kernel { COPY, COMPUTE }

    static double expression(double value) {
        return Math.sin(value) * Math.cos(value) + Math.tan(value);
    }

    static void kernel(double[] input, double[] output, int from, int to, Kernel kind) {
        if (kind == Kernel.COPY) {
            for (int i = from; i < to; i++) output[i] = input[i];
        } else {
            for (int i = from; i < to; i++) output[i] = expression(input[i]);
        }
    }

    static long measure(double[] input, double[] output, int workers,
                        ThreadPoolExecutor pool, Kernel kind) throws Exception {
        if (workers < 1 || input.length != output.length || input == output || kind == null) {
            throw new IllegalArgumentException("Distinct same-size arrays, a kernel and positive workers required");
        }
        long start = System.nanoTime();
        List<Future<?>> futures = new ArrayList<>(workers);
        for (int worker = 0; worker < workers; worker++) {
            int from = BenchmarkSupport.boundary(worker, input.length, workers);
            int to = BenchmarkSupport.boundary(worker + 1, input.length, workers);
            futures.add(pool.submit(() -> kernel(input, output, from, to, kind)));
        }
        for (Future<?> future : futures) future.get();
        return System.nanoTime() - start;
    }

    static double speedup(long baselineNs, long elapsedNs) {
        return (double) baselineNs / elapsedNs;
    }

    static double efficiency(long baselineNs, long elapsedNs, int workers) {
        return speedup(baselineNs, elapsedNs) / workers;
    }

    /** Useful traffic only: one double read plus one double write, decimal GB/s. */
    static double usefulGBps(int elements, long elapsedNs) {
        return 2.0 * Double.BYTES * elements / elapsedNs;
    }

    static boolean scalingStalled(long twoWorkersNs, long maxWorkersNs, int maxWorkers) {
        // Explicit reporting heuristic, not a statistical or hardware-counter test.
        return maxWorkers > 2 && speedup(twoWorkersNs, maxWorkersNs) <= 1.20;
    }

    static void validate(double[] input, double[] output, Kernel kind, boolean full) {
        int step = full ? 1 : Math.max(1, input.length / 17);
        for (int i = 0; i < input.length; i += step) {
            if (kind == Kernel.COPY) {
                if (Double.doubleToRawLongBits(input[i]) != Double.doubleToRawLongBits(output[i])) {
                    throw new AssertionError("Copy mismatch at " + i);
                }
            } else {
                double expected = StrictMath.sin(input[i]) * StrictMath.cos(input[i]) + StrictMath.tan(input[i]);
                BenchmarkSupport.checkClose(output[i], expected);
            }
        }
        if (input.length > 0) BenchmarkSupport.blackhole = output[input.length - 1];
    }

    static void reportConclusion(int[] workers, long[][] medians) {
        if (workers.length < 3) {
            System.out.println("Scaling conclusion: fewer than three distinct worker counts; no 2-to-max saturation test.");
            return;
        }
        int last = workers.length - 1;
        boolean copyStall = scalingStalled(medians[0][1], medians[0][last], workers[last]);
        boolean computeStall = scalingStalled(medians[1][1], medians[1][last], workers[last]);
        double copyGain = speedup(medians[0][1], medians[0][last]);
        double computeGain = speedup(medians[1][1], medians[1][last]);
        System.out.printf("2 -> %d worker throughput gain: copy=%.4fx, compute=%.4fx; ideal=%.4fx%n",
                workers[last], copyGain, computeGain, workers[last] / 2.0);
        System.out.println("Stall heuristic: at most 20% throughput gain from 2 workers to max (gain <= 1.20x).");
        if (copyStall) {
            System.out.println("Kernel A (memory-bound copy) stalled in scaling: consistent with memory channel saturation.");
        } else {
            System.out.println("Kernel A (memory-bound copy) did NOT meet the scaling-stall threshold in this run.");
        }
        System.out.println(computeStall
                ? "Kernel B (trigonometric compute) also stalled; timings cannot attribute that stall to RAM alone."
                : "Kernel B (trigonometric compute) continued scaling beyond 2 workers; no comparable plateau observed.");
        System.out.println("Inference, not proof: no RAM-controller counters or independent bandwidth ceiling measured.");
        System.out.println("CPU frequency, SMT, scheduling, JVM code generation and memory traffic can all affect scaling.");
    }

    public static void main(String[] args) throws Exception {
        BenchmarkSupport.header("TASK 7 - Arithmetic intensity versus memory bandwidth");
        int maxWorkers = Runtime.getRuntime().availableProcessors();
        LinkedHashSet<Integer> distinct = new LinkedHashSet<>(List.of(1, 2, maxWorkers));
        int[] workers = distinct.stream().mapToInt(Integer::intValue).sorted().toArray();
        double[] input = BenchmarkSupport.data(ELEMENTS);
        double[] output = new double[ELEMENTS];
        Arrays.fill(output, Double.NaN);
        System.out.printf("Elements: %,d float64 per array; seed: %d; workers: %s%n",
                ELEMENTS, BenchmarkSupport.STUDENT_ID, Arrays.toString(workers));
        System.out.printf("Two-array payload: %,d bytes; useful read+write traffic per pass: %,d bytes%n",
                2L * Double.BYTES * ELEMENTS, 2L * Double.BYTES * ELEMENTS);
        System.out.println("Kernel A: B[i] = A[i] (explicit copy loop).");
        System.out.println("Kernel B: B[i] = sin(A[i])*cos(A[i]) + tan(A[i]); A is uniform in [0,1), in radians.");
        System.out.println("Same seeded input and output buffer for all runs; no fused/simplified trig expression.");
        System.out.println("Persistent prestarted pools; dispatch + kernel + future joins timed for EVERY worker count, including 1.");
        System.out.println("Generation, allocation, sentinel reset, warmup and validation excluded from timing.");
        System.out.println("Output reset to NaN before every measured pass; no cache flushing or affinity pinning.");
        System.out.println("Seven trials; rotate worker-count order and alternate kernel order to reduce order bias.");
        long[][][] samples = new long[2][workers.length][TRIALS];
        long[][] medians = new long[2][workers.length];
        List<ThreadPoolExecutor> pools = new ArrayList<>();
        try {
            for (int count : workers) {
                ThreadPoolExecutor pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(count);
                pools.add(pool);
                BenchmarkSupport.startPool(pool, count);
            }
            double[] warmInput = BenchmarkSupport.data(100_000);
            double[] warmOutput = new double[warmInput.length];
            for (int warmup = 0; warmup < 12; warmup++) {
                for (Kernel kind : Kernel.values()) {
                    measure(warmInput, warmOutput, workers[workers.length - 1], pools.get(pools.size() - 1), kind);
                }
            }
            for (int index = 0; index < workers.length; index++) {
                for (int warmup = 0; warmup < 2; warmup++) {
                    for (Kernel kind : Kernel.values()) measure(input, output, workers[index], pools.get(index), kind);
                }
            }
            for (int trial = 0; trial < TRIALS; trial++) {
                for (int offset = 0; offset < workers.length; offset++) {
                    int index = (trial + offset) % workers.length;
                    for (int order = 0; order < 2; order++) {
                        Kernel kind = Kernel.values()[(trial + order) % 2];
                        Arrays.fill(output, Double.NaN);
                        samples[kind.ordinal()][index][trial] = measure(input, output, workers[index], pools.get(index), kind);
                        // Every final configuration is checked fully against an independent scalar reference.
                        validate(input, output, kind, trial == TRIALS - 1);
                    }
                }
            }
        } finally {
            for (ThreadPoolExecutor pool : pools) pool.close();
        }
        System.out.printf("%-10s | %7s | %16s | %9s | %10s | %15s%n",
                "Kernel", "Cores", "Median Wall (ms)", "Speedup", "Efficiency", "Useful GB/s");
        for (Kernel kind : Kernel.values()) {
            int k = kind.ordinal();
            for (int index = 0; index < workers.length; index++) {
                medians[k][index] = BenchmarkSupport.median(samples[k][index]);
                System.out.printf("%-10s | %7d | %16.6f | %8.4fx | %9.4f%% | %15.4f%n",
                        kind, workers[index], BenchmarkSupport.milliseconds(medians[k][index]),
                        speedup(medians[k][0], medians[k][index]),
                        100.0 * efficiency(medians[k][0], medians[k][index], workers[index]),
                        usefulGBps(ELEMENTS, medians[k][index]));
            }
        }
        System.out.println("Speedup=T(1)/T(p); Efficiency=Speedup/p (displayed as percent). Cores means worker count.");
        System.out.println("Useful GB/s=(N*8 bytes read + N*8 bytes written)/time/1e9; not measured physical RAM traffic.");
        System.out.println("Write allocation, cache effects and other bus traffic are NOT included in useful byte counts.");
        reportConclusion(workers, medians);
        System.out.println("Raw wall times (ns):");
        for (Kernel kind : Kernel.values()) {
            for (int index = 0; index < workers.length; index++) {
                System.out.printf("kernel=%s workers=%d %s%n", kind, workers[index],
                        Arrays.toString(samples[kind.ordinal()][index]));
            }
        }
        System.out.println("Validation PASS: final pass of every kernel/worker configuration checked across all 30,000,000 elements.");
        System.out.println("Copy checked bit-for-bit; compute checked against StrictMath reference, relative tolerance 1e-11.");
    }
}