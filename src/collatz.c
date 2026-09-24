/*
 * CSS314 Laboratory: The Amdahl Reality Gap
 * Student ID workload: 230103341 -> N = 10,341,000
 *
 * Compile from the repository root:
 *   gcc -O2 -fopenmp -std=c11 -Wall -Wextra -Wpedantic src/collatz.c -o src/collatz.exe
 *
 * Run from src so output CSV files remain beside the source:
 *   cd src
 *   .\collatz.exe all
 *
 * The program writes raw per-run measurements to results.csv. Run 1 is a
 * warmup and must be discarded; Runs 2 and 3 are averaged for analysis.
 */
#include <inttypes.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STUDENT_ID "230103341"
#define WORKLOAD_N 10341000ULL
#define CHECKSUM_MOD 1000000007ULL
#define MAX_THREADS 64
#define PHYSICAL_THREADS 6
#define RUNS_PER_TEST 3

typedef struct {
    double seconds;
    uint32_t max_steps;
    uint64_t checksum;
    uint64_t hits_over_100;
} Result;

typedef struct {
    omp_sched_t kind;
    const char *name;
    int chunk;
} ScheduleTest;

static const int scale_threads[] = {1, 2, 4, 6, 8, 12};
static int naive_counts[MAX_THREADS];

typedef struct __attribute__((aligned(64))) {
    int count;
    char pad[60];
} PaddedCounter;

static PaddedCounter padded_counts[MAX_THREADS];
_Static_assert(sizeof(PaddedCounter) == 64, "PaddedCounter must be 64 bytes");

static inline uint32_t collatz_steps(uint64_t n)
{
    uint32_t steps = 0;
    while (n > 1) {
        if ((n & 1U) == 0U)
            n >>= 1;
        else
            n = 3U * n + 1U;
        steps++;
    }
    return steps;
}

static Result make_result(double start, uint32_t max_steps,
                          uint64_t total_steps, uint64_t hits)
{
    Result r;
    r.seconds = omp_get_wtime() - start;
    r.max_steps = max_steps;
    r.checksum = total_steps % CHECKSUM_MOD;
    r.hits_over_100 = hits;
    return r;
}

static Result run_sequential(void)
{
    const double start = omp_get_wtime();
    uint64_t total_steps = 0;
    uint64_t hits = 0;
    uint32_t max_steps = 0;

    for (uint64_t i = 1; i <= WORKLOAD_N; ++i) {
        const uint32_t steps = collatz_steps(i);
        total_steps += steps;
        if (steps > max_steps)
            max_steps = steps;
        if (steps > 100U)
            hits++;
    }

    return make_result(start, max_steps, total_steps, hits);
}

static Result run_parallel_reduction(int threads)
{
    uint64_t total_steps = 0;
    uint64_t hits = 0;
    uint32_t max_steps = 0;

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    const double start = omp_get_wtime();
#pragma omp parallel for schedule(runtime) \
    reduction(+ : total_steps, hits) reduction(max : max_steps)
    for (uint64_t i = 1; i <= WORKLOAD_N; ++i) {
        const uint32_t steps = collatz_steps(i);
        total_steps += steps;
        if (steps > max_steps)
            max_steps = steps;
        if (steps > 100U)
            hits++;
    }

    return make_result(start, max_steps, total_steps, hits);
}

static Result run_false_sharing(int use_padding, int threads)
{
    uint64_t total_steps = 0;
    uint32_t max_steps = 0;
    uint64_t hits = 0;

    memset(naive_counts, 0, sizeof(naive_counts));
    memset(padded_counts, 0, sizeof(padded_counts));
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    const double start = omp_get_wtime();
    if (!use_padding) {
#pragma omp parallel for schedule(static) \
        reduction(+ : total_steps) reduction(max : max_steps)
        for (uint64_t i = 1; i <= WORKLOAD_N; ++i) {
            const uint32_t steps = collatz_steps(i);
            total_steps += steps;
            if (steps > max_steps)
                max_steps = steps;
            if (steps > 100U)
                naive_counts[omp_get_thread_num()]++;
        }
        for (int t = 0; t < threads; ++t)
            hits += (uint64_t)naive_counts[t];
    } else {
#pragma omp parallel for schedule(static) \
        reduction(+ : total_steps) reduction(max : max_steps)
        for (uint64_t i = 1; i <= WORKLOAD_N; ++i) {
            const uint32_t steps = collatz_steps(i);
            total_steps += steps;
            if (steps > max_steps)
                max_steps = steps;
            if (steps > 100U)
                padded_counts[omp_get_thread_num()].count++;
        }
        for (int t = 0; t < threads; ++t)
            hits += (uint64_t)padded_counts[t].count;
    }

    return make_result(start, max_steps, total_steps, hits);
}

static void write_row(FILE *csv, const char *phase, const char *variant,
                      const char *schedule, int chunk, int threads,
                      int run, Result r)
{
    fprintf(csv, "%s,%s,%s,%d,%d,%d,%.9f,%" PRIu32 ",%" PRIu64 ",%" PRIu64 "\n",
            phase, variant, schedule, chunk, threads, run, r.seconds,
            r.max_steps, r.checksum, r.hits_over_100);
    printf("%-12s %-22s threads=%-2d run=%d time=%.6f s max=%" PRIu32
           " checksum=%" PRIu64 "\n",
           phase, variant, threads, run, r.seconds, r.max_steps, r.checksum);
}

static void execute_sequential(FILE *csv)
{
    for (int run = 1; run <= RUNS_PER_TEST; ++run)
        write_row(csv, "sequential", "sequential", "none", 0, 1, run,
                  run_sequential());
}

static void execute_scaling(FILE *csv)
{
    omp_set_schedule(omp_sched_static, 0);
    for (size_t i = 0; i < sizeof(scale_threads) / sizeof(scale_threads[0]); ++i) {
        const int threads = scale_threads[i];
        for (int run = 1; run <= RUNS_PER_TEST; ++run)
            write_row(csv, "scaling", "reduction", "static_default", 0,
                      threads, run, run_parallel_reduction(threads));
    }
}

static void execute_scale6(FILE *csv)
{
    omp_set_schedule(omp_sched_static, 0);
    for (int run = 1; run <= RUNS_PER_TEST; ++run)
        write_row(csv, "scaling", "reduction", "static_default", 0,
                  6, run, run_parallel_reduction(6));
}

static void execute_false_sharing(FILE *csv)
{
    for (int run = 1; run <= RUNS_PER_TEST; ++run)
        write_row(csv, "false_sharing", "naive_shared_line", "static_default",
                  0, PHYSICAL_THREADS, run,
                  run_false_sharing(0, PHYSICAL_THREADS));
    for (int run = 1; run <= RUNS_PER_TEST; ++run)
        write_row(csv, "false_sharing", "cache_padded", "static_default", 0,
                  PHYSICAL_THREADS, run,
                  run_false_sharing(1, PHYSICAL_THREADS));
}

static void execute_schedules(FILE *csv)
{
    static const ScheduleTest tests[] = {
        {omp_sched_static, "static", 0},
        {omp_sched_static, "static_1000", 1000},
        {omp_sched_dynamic, "dynamic_100", 100},
        {omp_sched_dynamic, "dynamic_10000", 10000},
        {omp_sched_guided, "guided", 1}
    };
    const int threads = omp_get_max_threads();

    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
        omp_set_schedule(tests[i].kind, tests[i].chunk);
        for (int run = 1; run <= RUNS_PER_TEST; ++run)
            write_row(csv, "scheduling", tests[i].name, tests[i].name,
                      tests[i].chunk, threads, run,
                      run_parallel_reduction(threads));
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s seq|scale|scale6|falsesharing|schedule|all\n", argv[0]);
        return EXIT_FAILURE;
    }

    FILE *csv = fopen("results.csv", "w");
    if (csv == NULL) {
        perror("results.csv");
        return EXIT_FAILURE;
    }

    fprintf(csv, "phase,variant,schedule,chunk,threads,run,seconds,max_steps,checksum,hits_over_100\n");
    printf("Student ID: %s\nWorkload N: %" PRIu64 "\n", STUDENT_ID, WORKLOAD_N);
    printf("omp_get_max_threads(): %d\n", omp_get_max_threads());

    if (strcmp(argv[1], "seq") == 0 || strcmp(argv[1], "all") == 0)
        execute_sequential(csv);
    if (strcmp(argv[1], "scale") == 0 || strcmp(argv[1], "all") == 0)
        execute_scaling(csv);
    if (strcmp(argv[1], "scale6") == 0)
        execute_scale6(csv);
    if (strcmp(argv[1], "falsesharing") == 0 || strcmp(argv[1], "all") == 0)
        execute_false_sharing(csv);
    if (strcmp(argv[1], "schedule") == 0 || strcmp(argv[1], "all") == 0)
        execute_schedules(csv);
    if (strcmp(argv[1], "seq") != 0 && strcmp(argv[1], "scale") != 0 &&
        strcmp(argv[1], "scale6") != 0 &&
        strcmp(argv[1], "falsesharing") != 0 && strcmp(argv[1], "schedule") != 0 &&
        strcmp(argv[1], "all") != 0) {
        fprintf(stderr, "Unknown mode: %s\n", argv[1]);
        fclose(csv);
        return EXIT_FAILURE;
    }

    fclose(csv);
    return EXIT_SUCCESS;
}
