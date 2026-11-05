# Task 11 — Gustafson versus Sun–Ni memory-bounded scaling

**Student ID:** 230103341

## Model and baseline

I compare Gustafson's fixed-time viewpoint with Sun–Ni's memory-bounded workload expansion. [G88, SN93] I normalize the one-core baseline runtime to 1, with serial work s and parallelizable work 1−s. I retain serial work and expand parallel work by G(P), assuming identical processors, balance, no overhead, G(1) = 1 and 0 ≤ s ≤ 1, excluding zero work.

```text
T₁(Wₚ) = s+(1−s)G(P)
Tₚ(Wₚ) = s+(1−s)G(P)/P
S_SN = [s+(1−s)G(P)]/[s+(1−s)G(P)/P]
```

Here s describes the original one-core workload; the numerator describes the **expanded** workload. If work scales as N^q, storage as N^r and usable memory as P, filling memory gives Nₚ/N₁ = P^(1/r) and G(P) = P^(q/r). Replication, temporary storage and buffers can invalidate this mapping.

## Relationship to the familiar laws

I recover Amdahl with G = 1 and Gustafson with G = P:

```text
G = 1: S = 1/[s+(1−s)/P]
G = P: Tₚ = 1; S = s+P(1−s)
αₚ = s/[s+(1−s)G/P]
S = αₚ+P(1−αₚ)
```

Only G = P makes the baseline fraction s equal to the parallel-observed fraction αₚ along this construction. For general growth, αₚ varies; equal numerical symbols do not make the workload references interchangeable.

For P > 1 and 0 < s < 1, I derive:

```text
dS_SN/dG = s(1−s)(1−1/P)/[s+(1−s)G/P]² > 0
```

Holding baseline s fixed, the G = P prediction overestimates speedup when feasible G < P and underestimates it when G > P. This compares workload families, not faster execution of one fixed job.

## Calculated example

At s = 0.1 and P = 16, I obtain:

| G | Parallel time | Scaled speedup |
|---:|---:|---:|
| 1 | 0.15625 | 6.4000 |
| 4 | 0.32500 | 11.3846 |
| 16 | 1.00000 | 14.5000 |
| 64 | 3.70000 | 15.5946 |

The last row buys higher speedup with longer runtime, not the same time budget. All positive-work cases satisfy S ≤ P. At s = 0, S = P; at s = 1, S = 1.

I use Sun–Ni when memory sets feasible resolution and Gustafson when time sets useful work. Neither removes communication or guarantees invariant setup. A practical model needs storage, useful work, serial work and communication as separate input-size functions.

## References

- **[SN93]** X.-H. Sun and L. M. Ni, “Scalable Problems and Memory-Bounded Speedup,” *JPDC* 19(1), 27–37 (1993): [DOI](https://doi.org/10.1006/jpdc.1993.1087); [Crossref](https://api.crossref.org/works/10.1006/jpdc.1993.1087). Existing preparation verified metadata, not inaccessible publisher full text.
- **[G88]** J. L. Gustafson, “Reevaluating Amdahl's Law,” *CACM* 31(5), 532–533 (1988): [DOI](https://doi.org/10.1145/42411.42415).