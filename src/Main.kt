import java.util.Random
import java.util.concurrent.atomic.AtomicLong
import kotlin.system.measureNanoTime

const val TOTAL_POINTS_PART1 = 50_000_000L
const val NUM_THREADS_PART1 = 4

const val TOTAL_POINTS_PART3 = 100_000_000L
val THREAD_COUNTS = intArrayOf(1, 2, 4, 8, 16, 32)

/**
 * Worker: counts how many random (x, y) points in [0,1) x [0,1) land inside the
 * unit quarter-circle. Each thread gets its own Random instance so RNG
 * contention never becomes a confounding factor in the benchmark.
 */
private fun countHits(iterations: Long): Long {
    val rnd = Random()
    var hits = 0L
    var i = 0L
    while (i < iterations) {
        val x = rnd.nextDouble()
        val y = rnd.nextDouble()
        if (x * x + y * y <= 1.0) hits++
        i++
    }
    return hits
}

/**
 * PART 1 — The Phantom Bug
 * A single shared, unsynchronized Long is incremented from 4 threads via totalHits++.
 * Because ++ is read-modify-write (three separate bytecode steps), concurrent threads
 * clobber each other's updates, and the final count under-reports the true hit count.
 */
private var sharedHitsUnsafe = 0L

fun runPart1(): Double {
    sharedHitsUnsafe = 0L
    val perThread = TOTAL_POINTS_PART1 / NUM_THREADS_PART1

    val threads = List(NUM_THREADS_PART1) {
        Thread {
            val rnd = Random()
            var i = 0L
            while (i < perThread) {
                val x = rnd.nextDouble()
                val y = rnd.nextDouble()
                if (x * x + y * y <= 1.0) {
                    sharedHitsUnsafe++ // NOT atomic — this is the bug
                }
                i++
            }
        }
    }

    threads.forEach { it.start() }
    threads.forEach { it.join() }

    return 4.0 * sharedHitsUnsafe / TOTAL_POINTS_PART1
}

/**
 * PART 2 — The Synchronization Trap
 * Same logic as Part 1, but the increment is made atomic. Two variants are provided:
 * one using AtomicLong.incrementAndGet(), one using a synchronized block. Both are
 * correct but slow, since every single increment now fights for a lock / CAS on a
 * cache line shared across cores.
 */
fun runPart2Atomic(): Pair<Double, Long> {
    val hits = AtomicLong(0)
    val perThread = TOTAL_POINTS_PART1 / NUM_THREADS_PART1

    val elapsedNanos = measureNanoTime {
        val threads = List(NUM_THREADS_PART1) {
            Thread {
                val rnd = Random()
                var i = 0L
                while (i < perThread) {
                    val x = rnd.nextDouble()
                    val y = rnd.nextDouble()
                    if (x * x + y * y <= 1.0) {
                        hits.incrementAndGet()
                    }
                    i++
                }
            }
        }
        threads.forEach { it.start() }
        threads.forEach { it.join() }
    }

    val pi = 4.0 * hits.get() / TOTAL_POINTS_PART1
    return Pair(pi, elapsedNanos / 1_000_000)
}

fun runSingleThreadedBaseline(totalPoints: Long): Pair<Double, Long> {
    var hits = 0L
    val elapsedNanos = measureNanoTime {
        hits = countHits(totalPoints)
    }
    val pi = 4.0 * hits / totalPoints
    return Pair(pi, elapsedNanos / 1_000_000)
}

/**
 * PART 3 — OpenMP-Style Reduction
 * Each thread accumulates into its own private local counter (its own slot in the
 * results array), with zero shared-state writes during the loop. Partial sums are
 * combined only once, after every thread has finished — equivalent to
 * #pragma omp parallel for reduction(+:totalHits).
 */
fun runPart3Reduction(numThreads: Int, totalPoints: Long): Pair<Double, Long> {
    val perThread = totalPoints / numThreads
    val results = LongArray(numThreads)

    val elapsedNanos = measureNanoTime {
        val threads = List(numThreads) { idx ->
            Thread {
                results[idx] = countHits(perThread)
            }
        }
        threads.forEach { it.start() }
        threads.forEach { it.join() }
    }

    val totalHits = results.sum()
    val pi = 4.0 * totalHits / totalPoints
    return Pair(pi, elapsedNanos / 1_000_000)
}

fun main() {
    println("=".repeat(70))
    println("PART 1 — The Phantom Bug (unsynchronized totalHits++, 4 threads)")
    println("=".repeat(70))
    repeat(5) { run ->
        val pi = runPart1()
        println("Run ${run + 1}: pi ≈ %.6f".format(pi))
    }

    println()
    println("=".repeat(70))
    println("PART 2 — The Synchronization Trap")
    println("=".repeat(70))

    val (piSingle, timeSingle) = runSingleThreadedBaseline(TOTAL_POINTS_PART1)
    println("Single-threaded baseline: pi ≈ %.6f, time = %d ms".format(piSingle, timeSingle))

    val (piAtomic, timeAtomic) = runPart2Atomic()
    println("AtomicLong (4 threads):   pi ≈ %.6f, time = %d ms".format(piAtomic, timeAtomic))
    println("Slowdown vs single thread: %.2fx".format(timeAtomic.toDouble() / timeSingle))

    println()git add .gitignore src/Main.kt
    println("=".repeat(70))
    println("PART 3 — OpenMP-Style Reduction (local counters, 100,000,000 points)")
    println("=".repeat(70))

    var baselineTime = 0L
    println("%-10s %-12s %-22s %-12s".format("Threads", "Runtime(ms)", "Speedup vs 1 Thread", "Efficiency"))
    println("-".repeat(60))

    for (t in THREAD_COUNTS) {
        val (pi, timeMs) = runPart3Reduction(t, TOTAL_POINTS_PART3)
        if (t == 1) baselineTime = timeMs

        val speedup = baselineTime.toDouble() / timeMs
        val efficiency = speedup / t * 100

        println(
            "%-10d %-12d %-22s %-12s   (pi ≈ %.6f)".format(
                t,
                timeMs,
                "%.2fx".format(speedup),
                "%.1f%%".format(efficiency),
                pi
            )
        )
    }
}