import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

/** Real OS-pipe serialization versus a reusable cross-process file mapping (CPU option). */
public class Task4 {
    private static final int MAGIC = 0x54344950;
    private static final int INIT = 1, SERIALIZED = 2, SHARED = 3, QUIT = 4;
    private static final int TRIALS = 5;
    private static final long TIMEOUT_SECONDS = 120;

    private record Reply(long sequence, double checksum, long childNs) { }
    private record Sample(long elapsedNs, long childNs, double checksum) { }

    private static double checksum(double[] values) {
        double sum = 0;
        for (double value : values) sum += value;
        return sum;
    }

    private static double checksum(DoubleBuffer values, int length) {
        double sum = 0;
        for (int i = 0; i < length; i++) sum += values.get(i);
        return sum;
    }

    private static void verify(double actual, double expected, String context) {
        try {
            PracticeSupport.closeEnough(actual, expected);
        } catch (AssertionError failure) {
            throw new AssertionError(context + ": checksum " + actual + ", expected " + expected, failure);
        }
    }

    /* The parent deliberately NEVER maps the file: no nonportable unmap/GC trick is needed.
       Copy the exact seeded parent array, not an independently regenerated approximation.
       For updates in selfTest, do not truncate a file that the child already has mapped. */
    private static void populate(Path file, double[] values) throws IOException {
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.WRITE)) {
            ByteBuffer chunk = ByteBuffer.allocate(64 * 1024).order(ByteOrder.BIG_ENDIAN);
            int index = 0;
            while (index < values.length) {
                chunk.clear();
                while (chunk.remaining() >= Double.BYTES && index < values.length) {
                    chunk.putDouble(values[index++]);
                }
                chunk.flip();
                while (chunk.hasRemaining()) channel.write(chunk);
            }
            channel.force(false);
        }
    }

    private static void deleteFile(Path file) throws IOException {
        try {
            Files.deleteIfExists(file);
        } catch (IOException failure) {
            // A virus scanner/other external opener can still temporarily lock a Windows file.
            file.toFile().deleteOnExit();
            System.err.println("Task4: immediate temp-file deletion failed; deleteOnExit registered for "
                    + file + ": " + failure);
            throw failure;
        }
    }

    @FunctionalInterface
    private interface IOAction<T> { T run() throws IOException; }

    /** Binary-only stdout. stderr goes directly to the parent's stderr, never into the protocol. */
    private static final class Child implements AutoCloseable {
        private final Process process;
        private final ScheduledExecutorService watchdog;
        private ObjectOutputStream commands;
        private DataInputStream replies;
        private volatile String timedOut;
        private long sequence;
        private boolean closed;
        final long startupNs;

        Child() throws IOException {
            long start = System.nanoTime();
            String executable = System.getProperty("os.name").startsWith("Windows") ? "java.exe" : "java";
            Path java = Path.of(System.getProperty("java.home"), "bin", executable);
            process = new ProcessBuilder(java.toString(), "-Xmx512m", "-cp",
                    System.getProperty("java.class.path"), Task4.class.getName(), "--child")
                    .redirectError(ProcessBuilder.Redirect.INHERIT).start();
            watchdog = Executors.newSingleThreadScheduledExecutor(r -> {
                Thread thread = new Thread(r, "Task4-child-watchdog");
                thread.setDaemon(true);
                return thread;
            });
            try {
                guarded("startup handshake", () -> {
                    commands = new ObjectOutputStream(new BufferedOutputStream(process.getOutputStream()));
                    // Flush the serialization header BEFORE the child constructs ObjectInputStream.
                    commands.flush();
                    replies = new DataInputStream(new BufferedInputStream(process.getInputStream()));
                    if (replies.readInt() != MAGIC) throw new IOException("Bad child handshake");
                    return null;
                });
                startupNs = System.nanoTime() - start;
            } catch (IOException | RuntimeException | Error failure) {
                abort();
                throw failure;
            }
        }

        private <T> T guarded(String operation, IOAction<T> action) throws IOException {
            ScheduledFuture<?> alarm = watchdog.schedule(() -> {
                timedOut = operation;
                process.destroyForcibly(); // Also releases a parent blocked on pipe read/write.
            }, TIMEOUT_SECONDS, TimeUnit.SECONDS);
            try {
                return action.run();
            } catch (IOException failure) {
                throw new IOException("Task4 child protocol failed during " + operation
                        + (timedOut == null ? "" : " (timeout: " + timedOut + ")")
                        + (process.isAlive() ? "" : "; exit=" + process.exitValue())
                        + "; see inherited stderr", failure);
            } finally {
                alarm.cancel(false);
            }
        }

        private Reply readReply(long expectedSequence) throws IOException {
            if (replies.readInt() != MAGIC) throw new IOException("Invalid reply marker");
            Reply reply = new Reply(replies.readLong(), replies.readDouble(), replies.readLong());
            if (reply.sequence() != expectedSequence) throw new IOException("Out-of-order child reply");
            return reply;
        }

        Sample initialize(Path file, int length) throws IOException {
            return guarded("map and first-touch", () -> {
                long id = ++sequence;
                long start = System.nanoTime();
                commands.writeInt(INIT);
                commands.writeLong(id);
                commands.writeUTF(file.toAbsolutePath().toString());
                commands.writeInt(length);
                commands.flush();
                Reply reply = readReply(id);
                return new Sample(System.nanoTime() - start, reply.childNs(), reply.checksum());
            });
        }

        Sample transfer(boolean serialized, double[] values) throws IOException {
            return guarded(serialized ? "serialized transfer" : "shared consumption", () -> {
                long id = ++sequence;
                long start = System.nanoTime();
                if (serialized) commands.reset(); // No stream handle/reference cache across trials.
                commands.writeInt(serialized ? SERIALIZED : SHARED);
                commands.writeLong(id);
                if (serialized) commands.writeUnshared(values); // Actual full double[] on OS pipe.
                commands.flush();
                Reply reply = readReply(id);
                return new Sample(System.nanoTime() - start, reply.childNs(), reply.checksum());
            });
        }

        private void abort() {
            process.destroyForcibly();
            // Do not delete the mapped file until the OS has actually reaped the child.
            boolean interrupted = false;
            while (true) {
                try {
                    process.waitFor();
                    break;
                } catch (InterruptedException failure) {
                    interrupted = true;
                }
            }
            watchdog.shutdownNow();
            if (interrupted) Thread.currentThread().interrupt();
        }

        @Override
        public void close() throws IOException {
            if (closed) return;
            closed = true;
            try {
                guarded("shutdown", () -> {
                    long id = ++sequence;
                    commands.writeInt(QUIT);
                    commands.writeLong(id);
                    commands.flush();
                    readReply(id);
                    return null;
                });
                if (!process.waitFor(10, TimeUnit.SECONDS)) throw new IOException("Child failed to exit");
                if (process.exitValue() != 0) throw new IOException("Child exit=" + process.exitValue());
            } catch (InterruptedException failure) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted waiting for child exit", failure);
            } finally {
                abort();
                if (commands != null) commands.close();
                if (replies != null) replies.close();
            }
        }
    }

    private static void reply(DataOutputStream out, long sequence, double sum, long childNs)
            throws IOException {
        out.writeInt(MAGIC);
        out.writeLong(sequence);
        out.writeDouble(sum);
        out.writeLong(childNs);
        out.flush();
    }

    private static void childMain() throws Exception {
        try (ObjectInputStream in = new ObjectInputStream(new BufferedInputStream(System.in));
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(System.out))) {
            out.writeInt(MAGIC);
            out.flush();
            MappedByteBuffer mapping = null;
            DoubleBuffer shared = null;
            int length = -1;
            while (true) {
                int command = in.readInt();
                long sequence = in.readLong();
                if (command == QUIT) {
                    reply(out, sequence, 0, 0);
                    return; // Process exit unmaps, including on Windows. No unsafe/internal APIs.
                }
                if (command == INIT) {
                    if (length >= 0) throw new IOException("Child mapping already initialized");
                    Path file = Path.of(in.readUTF());
                    length = in.readInt();
                    if (length < 0 || length > 10_000_000) throw new IOException("Invalid array length");
                    long bytes = (long) length * Double.BYTES;
                    long start = System.nanoTime();
                    try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
                        if (channel.size() != bytes) throw new IOException("Unexpected backing-file length");
                        if (length > 0) {
                            mapping = channel.map(FileChannel.MapMode.READ_ONLY, 0, bytes);
                            shared = mapping.order(ByteOrder.BIG_ENDIAN).asDoubleBuffer();
                        }
                    }
                    // Every double/page is consumed before any reused zero-payload-copy timing.
                    double sum = checksum(shared, length);
                    reply(out, sequence, sum, System.nanoTime() - start);
                } else if (command == SERIALIZED || command == SHARED) {
                    if (length < 0) throw new IOException("Mapping is not initialized");
                    double[] values = null;
                    if (command == SERIALIZED) {
                        Object object = in.readUnshared();
                        if (!(object instanceof double[] array) || array.length != length) {
                            throw new IOException("Expected float64 array of length " + length);
                        }
                        values = array;
                    }
                    long start = System.nanoTime();
                    double sum = command == SERIALIZED ? checksum(values) : checksum(shared, length);
                    reply(out, sequence, sum, System.nanoTime() - start);
                } else {
                    throw new IOException("Unknown command " + command);
                }
            }
        }
    }

    private static void benchmark(int length) throws Exception {
        double[] values = PracticeSupport.data(length);
        double expected = checksum(values);
        Path file = Files.createTempFile("task4-" + PracticeSupport.STUDENT_ID + "-", ".f64");
        try {
            long start = System.nanoTime();
            populate(file, values);
            long populationNs = System.nanoTime() - start;
            try (Child child = new Child()) {
                Sample initialization = child.initialize(file, length);
                verify(initialization.checksum(), expected, "initial mapped first-touch");
                // Warm both paths, including serialization metadata and the checksum loops.
                verify(child.transfer(true, values).checksum(), expected, "serialized warmup");
                verify(child.transfer(false, values).checksum(), expected, "mapped warmup");
                System.out.printf("%nN=%,d float64, payload=%,d bytes%n", length, (long) length * Double.BYTES);
                System.out.printf("Excluded setup: raw file population+force=%.3f ms; process+handshake=%.3f ms;%n"
                                + "  child mapping+full first-touch+ack=%.3f ms (child local=%.3f ms).%n",
                        PracticeSupport.ms(populationNs), PracticeSupport.ms(child.startupNs),
                        PracticeSupport.ms(initialization.elapsedNs()), PracticeSupport.ms(initialization.childNs()));
                long[] serialized = new long[TRIALS];
                long[] shared = new long[TRIALS];
                for (int trial = 0; trial < TRIALS; trial++) {
                    for (int order = 0; order < 2; order++) {
                        boolean useSerialization = ((trial + order) & 1) == 0;
                        Sample sample = child.transfer(useSerialization, values);
                        verify(sample.checksum(), expected, "trial " + (trial + 1));
                        (useSerialization ? serialized : shared)[trial] = sample.elapsedNs();
                        System.out.printf("trial=%d order=%d %s wall=%.3f ms; child checksum=%.3f ms; "
                                        + "sum=%.17g OK%n", trial + 1, order + 1,
                                useSerialization ? "A serialized pipe" : "B reused file map",
                                PracticeSupport.ms(sample.elapsedNs()), PracticeSupport.ms(sample.childNs()),
                                sample.checksum());
                    }
                }
                long medianA = PracticeSupport.median(serialized);
                long medianB = PracticeSupport.median(shared);
                System.out.printf("Median (%d alternating trials): A=%.3f ms; B=%.3f ms; A/B=%.3fx%n",
                        TRIALS, PracticeSupport.ms(medianA), PracticeSupport.ms(medianB),
                        (double) medianA / medianB);
                System.out.println("Raw A serialized-pipe wall times (ns): " + java.util.Arrays.toString(serialized));
                System.out.println("Raw B reused-file-map wall times (ns): " + java.util.Arrays.toString(shared));
            } // Child fully exited and its mapping released before the finally deletion.
        } finally {
            deleteFile(file);
        }
    }

    /** Fast protocol/cleanup hook: tiny sizes only; no benchmark-sized CPU workload. */
    public static void selfTest() throws Exception {
        for (int length : new int[] {0, 1, 101, 10_003}) {
            double[] values = PracticeSupport.data(length);
            Path file = Files.createTempFile("task4-test-", ".f64");
            try {
                populate(file, values);
                try (Child child = new Child()) {
                    double expected = checksum(values);
                    verify(child.initialize(file, length).checksum(), expected, "test initialization N=" + length);
                    // Reusing the SAME array identity must not suppress actual serialization.
                    for (int iteration = 0; iteration < 3; iteration++) {
                        if (length > 0) {
                            values[0] += 16.0 + iteration;
                            values[length - 1] -= 0.25;
                        }
                        expected = checksum(values);
                        // Existing child mapping must see actual changes written by the parent.
                        populate(file, values);
                        for (int order = 0; order < 2; order++) {
                            boolean serialized = ((iteration + order) & 1) == 0;
                            verify(child.transfer(serialized, values).checksum(), expected,
                                    "test " + (serialized ? "pipe" : "shared updates")
                                            + " N=" + length + " iteration=" + iteration);
                        }
                    }
                }
            } finally {
                deleteFile(file);
            }
            if (Files.exists(file)) throw new AssertionError("Temp file not deleted: " + file);
        }
    }

    public static void main(String[] args) throws Exception {
        // This branch MUST precede header/output: child stdout is a dedicated binary protocol.
        if (args.length == 1 && args[0].equals("--child")) {
            childMain();
            return;
        }
        if (args.length == 1 && args[0].equals("--self-test")) {
            selfTest();
            System.out.println("Task4 selfTest: PASS (0, 1, 101, 10003; reused arrays, shared updates, cleanup)");
            return;
        }
        if (args.length != 0) throw new IllegalArgumentException("Usage: Task4 [--self-test]");
        PracticeSupport.header("TASK 4 - Real interprocess float64 transfer (CPU option)");
        System.out.println("A: ObjectOutputStream reset + writeUnshared(double[]) over a parent/child OS pipe.");
        System.out.println("B: child maps the parent's raw file once using file-backed MappedByteBuffer (not allocateDirect).");
        System.out.println("Reused B has NO repeated shared-memory payload copy; it DOES read every double for a checksum,");
        System.out.println("and includes control IPC + acknowledgment; OS mapping/page first-use may still affect timings.");
        System.out.println("Both wall times include the full child checksum + acknowledgment; checksums validated every trial.");
        System.out.println("Process startup, file population, mapping/full first-touch and warmups excluded; setup reported separately.");
        System.out.println("Five trials alternate A/B then B/A; medians are end-to-end wall times, not pure bandwidth.");
        System.out.println("Parent never maps; child exit releases mappings before Windows deletion (fallback deleteOnExit is reported).");
        for (int length : new int[] {100_000, 1_000_000, 10_000_000}) benchmark(length);
    }
}