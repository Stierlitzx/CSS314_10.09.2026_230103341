import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.Callable;
import java.util.concurrent.ThreadPoolExecutor;

/** Fixed CPU-bound SHA-256 batch across platform-thread pool sizes. */
public class Task5 {
    static final int TASKS = 10_000;
    static final int HASHES = 500;
    static final int BYTES = 1_024;
    static final int TRIALS = 3;

    record Measurement(long ns, long checksum) { }

    static byte[][] inputs(int tasks) {
        byte[][] values = new byte[tasks][BYTES];
        SplittableRandom random = new SplittableRandom(PracticeSupport.STUDENT_ID);
        for (byte[] value : values) {
            for (int i = 0; i < value.length; i++) value[i] = (byte) random.nextInt(256);
        }
        return values;
    }

    static long hash(byte[] source, int repetitions) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] input = source.clone(); // Task-private evolving 1 KiB payload, same for every pool size.
        byte[] result = new byte[32];
        for (int i = 0; i < repetitions; i++) {
            digest.update(input);
            digest.digest(result, 0, result.length);
            // Depend on the preceding digest while STILL hashing a full 1 KiB each time.
            System.arraycopy(result, 0, input, 0, result.length);
        }
        long value = 0;
        for (int i = 0; i < Long.BYTES; i++) value = (value << 8) | (result[i] & 0xffL);
        return value;
    }

    static Measurement measure(byte[][] inputs, int repetitions, ThreadPoolExecutor pool) throws Exception {
        long begin = System.nanoTime();
        List<Callable<Long>> jobs = new ArrayList<>(inputs.length);
        for (byte[] input : inputs) jobs.add(() -> hash(input, repetitions));
        long checksum = 0;
        for (long value : PracticeSupport.invoke(pool, jobs)) checksum += value;
        return new Measurement(System.nanoTime() - begin, checksum);
    }

    static int[] sizes(int cores) {
        if (cores < 1) throw new IllegalArgumentException("Positive core count required");
        LinkedHashSet<Integer> sizes = new LinkedHashSet<>(List.of(1, Math.max(1, cores / 2), cores,
                Math.multiplyExact(cores, 2), Math.multiplyExact(cores, 4), Math.multiplyExact(cores, 16), 256));
        return sizes.stream().mapToInt(Integer::intValue).sorted().toArray();
    }

    static int firstWorsening(long[] times) {
        for (int i = 1; i < times.length; i++) if (times[i] > times[i - 1]) return i;
        return -1;
    }

    public static void main(String[] args) throws Exception {
        PracticeSupport.header("PRACTICE TASK 5 - Thread oversubscription threshold");
        int cores = Runtime.getRuntime().availableProcessors();
        int physical = PracticeSupport.physicalCores();
        int[] sizes = sizes(cores);
        System.out.printf("Cores for requested pool-size formula = logical cores (%d); physical cores=%d.%n", cores, physical);
        System.out.println("Pool sizes: " + Arrays.toString(sizes));
        System.out.printf("Batch: %,d independent tasks, each hashing %,d-byte payload %d times with SHA-256.%n", TASKS, BYTES, HASHES);
        System.out.println("Task-private digest and payload; digest feeds next 1 KiB hash; deterministic checksum checked for every batch.");
        System.out.println("Platform threads (NOT virtual); prestart/pool startup excluded; submit, hashing, allocations and joins included.");
        System.out.println("Three trials; rotate pool-size order; no affinity pinning or forced context switches.");
        byte[][] inputs = inputs(TASKS);
        long[][] raw = new long[sizes.length][TRIALS];
        long[] medians = new long[sizes.length];
        List<ThreadPoolExecutor> pools = new ArrayList<>();
        long expected;
        try {
            for (int size : sizes) pools.add(PracticeSupport.pool(size));
            // Warm code/provider and each pool; only one batch is active at any instant.
            byte[][] warm = inputs(200);
            for (ThreadPoolExecutor pool : pools) measure(warm, HASHES, pool);
            expected = measure(inputs, HASHES, pools.get(0)).checksum();
            for (int trial = 0; trial < TRIALS; trial++) {
                for (int offset = 0; offset < sizes.length; offset++) {
                    int index = (offset + trial * 2) % sizes.length;
                    Measurement result = measure(inputs, HASHES, pools.get(index));
                    if (result.checksum() != expected) throw new AssertionError("Batch checksum mismatch");
                    raw[index][trial] = result.ns();
                    System.out.printf("trial=%d pool=%d wall=%.6f ms (%d ns) checksum=%d%n",
                            trial + 1, sizes[index], PracticeSupport.ms(result.ns()), result.ns(), result.checksum());
                }
            }
        } finally {
            for (ThreadPoolExecutor pool : pools) pool.close();
        }
        for (int index = 0; index < sizes.length; index++) medians[index] = PracticeSupport.median(raw[index]);
        int first = firstWorsening(medians);
        int best = 0;
        for (int i = 1; i < medians.length; i++) if (medians[i] < medians[best]) best = i;
        System.out.println("Pool Size | Execution Time (ms) | Speedup relative to 1 thread | Status");
        for (int index = 0; index < sizes.length; index++) {
            String status = index == 0 ? "BASELINE" : medians[index] > medians[index - 1] ? "WORSENED" : "NOT SLOWER";
            if (index == first) status += " / FIRST OBSERVED WORSENING";
            if (index == best) status += " / FASTEST";
            if (sizes[index] > cores) status += " / OVERSUBSCRIBED";
            System.out.printf("%d | %.6f | %.4fx | %s%n", sizes[index], PracticeSupport.ms(medians[index]),
                    (double) medians[0] / medians[index], status);
        }
        if (first >= 0) {
            System.out.printf("First observed worsening at tested pool size %d: %.6f -> %.6f ms from pool %d.%n",
                    sizes[first], PracticeSupport.ms(medians[first - 1]), PracticeSupport.ms(medians[first]), sizes[first - 1]);
        } else {
            System.out.println("No worsening observed among the tested median pool sizes.");
        }
        System.out.printf("Fastest tested pool: %d; verified batch checksum: %d%n", sizes[best], expected);
        System.out.println("Interpretation: oversubscription can add context-switching/scheduling cost, but wall times alone do NOT prove its cause.");
        System.out.println("This is the exact first worsening among sampled sizes, NOT an exact universal OS context-switch threshold.");
        System.out.println("Curve need not be monotonic: JVM compilation, SHA intrinsics, frequency, thermal and system load can change results.");
        System.out.println("Idle prestarted pools coexist to rotate measurement order; their creation/memory cost is outside timing.");
        for (int i = 0; i < sizes.length; i++) System.out.printf("Raw pool=%d wall times (ns): %s%n", sizes[i], Arrays.toString(raw[i]));
    }
}