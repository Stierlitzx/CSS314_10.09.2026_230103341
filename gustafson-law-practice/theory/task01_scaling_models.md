# Task 01 — Fixed-work and scaled-work performance models

**Student ID:** 230103341

## Definitions

I use P ≥ 1 identical processors and workload W, including input, resolution, algorithm and stopping criterion. Let a(W) be serial time and b(W) the one-core time of parallelizable work. I assume positive runtime, unchanged processor speed, perfect balance, no extra operations and no communication or synchronization overhead.

```text
T₁(W) = a+b                     Tₚ(W) = a+b/P
S(P,W) = T₁(W)/Tₚ(W)            E(P,W) = S/P
s_A = a/(a+b)                   αₚ = a/(a+b/P)
```

I distinguish the one-core serial fraction s_A from the parallel-observed fraction αₚ. Their denominators differ, matching Gustafson's distinction between one-processor and parallel-system quantities. [G88]

## Fixed work versus scaled work

For **strong scaling**, I hold W, a, b and s_A fixed and derive Amdahl's expression:

```text
S_A = 1/[s_A+(1−s_A)/P] = P/[1+(P−1)s_A]
αₚ = P s_A/[1+(P−1)s_A]
```

This asks how much sooner the same problem finishes. For s_A > 0, speedup approaches 1/s_A, while αₚ increases toward 1. I follow [G88]'s attribution to Amdahl's 1967 paper, “Validity of the single-processor approach to achieving large scale computing capabilities.”

For **scaled work**, I choose Wₚ with constant a and b(Wₚ) = P b₀:

```text
T₁(Wₚ) = a+P b₀                 Tₚ(Wₚ) = a+b₀
α = a/(a+b₀)
S_G = α+P(1−α) = P−(P−1)α        E_G = 1−α+α/P
```

I now enlarge computation within a fixed time budget. This resembles weak scaling, but fixed data per processor need not mean fixed computation: iterations and global operations may grow. The weak-scaling time ratio T₁(W₁)/Tₚ(Wₚ) equals 1 here, unlike E_G.

## Conversion and numerical check

For the same workload and processor count, I obtain inverse mappings:

```text
αₚ = P s_A/[1+(P−1)s_A]
s_A = αₚ/[P−(P−1)αₚ]
```

At P = 16, s_A = 0.05 gives S_A = 64/7 ≈ 9.142857 and α₁₆ ≈ 0.457143. Substitution into Gustafson's formula reproduces 9.142857. A different workload with α₁₆ = 0.05 gives S_G = 15.25 and s_A ≈ 0.003278689. Thus the laws agree after normalization; identical numerical fractions are not interchangeable.

For real runs I extend runtime to a+b/P+h(P,W), where h includes additional communication, synchronization, imbalance and operations without double-counting. A measured serial fraction alone then cannot determine speedup. [GMB88]

## References

- **[G88]** J. L. Gustafson, “Reevaluating Amdahl's Law,” *CACM* 31(5), 532–533 (May 1988): [article](http://www.johngustafson.net/pubs/pub13/amdahl.htm); [DOI](https://doi.org/10.1145/42411.42415).
- **[GMB88]** J. L. Gustafson, G. R. Montry and R. E. Benner, “Development of Parallel Methods for a 1024-Processor Hypercube,” *SIAM J. Sci. Stat. Comput.* 9(4), 609–638 (July 1988): [full text, §§2–4](http://www.johngustafson.net/pubs/pub16/1024.html), with minor 1995 web revisions.