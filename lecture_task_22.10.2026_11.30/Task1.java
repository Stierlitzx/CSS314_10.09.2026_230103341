/** Task 1: hardware probe, seeded float64 data, strictly serial reduction. */
public class Task1 {
    public static void main(String[] args) {
        BenchmarkSupport.header("TASK 1 - Hardware probe and serial baseline");
        // Compile the same scalar loops before timing the full-size workload.
        double[] warmup = BenchmarkSupport.data(100_000);
        for (int i = 0; i < 10; i++) {
            BenchmarkSupport.blackhole = BenchmarkSupport.statistics(warmup).sum();
        }
        long totalStart = System.nanoTime();
        long generationStart = System.nanoTime();
        double[] values = BenchmarkSupport.data(BenchmarkSupport.ARRAY_SIZE);
        long generationNs = System.nanoTime() - generationStart;
        long computeStart = System.nanoTime();
        BenchmarkSupport.Statistics result = BenchmarkSupport.statistics(values);
        long computeNs = System.nanoTime() - computeStart;
        long totalNs = System.nanoTime() - totalStart;
        System.out.printf("Elements: %,d float64 (Java double), seed: %d%n",
                values.length, BenchmarkSupport.STUDENT_ID);
        System.out.println("Computation workers: 1 (explicit scalar loops, no parallel library)");
        System.out.printf("Sum: %.15f%n", result.sum());
        System.out.printf("Population standard deviation (ddof=0): %.15f%n", result.standardDeviation());
        System.out.printf("Array allocation + generation wall time: %.6f ms (%d ns)%n",
                BenchmarkSupport.milliseconds(generationNs), generationNs);
        System.out.printf("Serial sum + standard deviation wall time: %.6f ms (%d ns)%n",
                BenchmarkSupport.milliseconds(computeNs), computeNs);
        System.out.printf("Total generation + computation wall time: %.6f ms (%d ns)%n",
                BenchmarkSupport.milliseconds(totalNs), totalNs);
        System.out.println("Timing excludes JVM startup, hardware reporting, warmup and printing.");
    }
}