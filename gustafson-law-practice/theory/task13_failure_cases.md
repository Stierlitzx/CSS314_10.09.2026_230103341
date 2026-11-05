# Task 13 — Two limits of Gustafson-style reasoning

**Student ID:** 230103341

## Conditional scaling

For useful serial time A and balanced parallel time B, I obtain the overhead-free identity:

```text
α = A/(A+B)
S = (A+PB)/(A+B) = α+P(1−α)
```

Failure means an application cannot realize the workload family, constant fraction or negligible overhead—not that this identity is false. [G88]

## Single-system Thomas algorithm

I consider the standard Thomas tridiagonal solver with nonzero elimination pivots. Forward elimination depends on the previous row; back substitution depends on the next unknown. Both work W and dependency depth D are Θ(N), giving available parallelism W/D = Θ(1). [T]

Dividing rows among P workers does not create independent recurrences. Enlarging one system to PN grows work and critical path together, so runtime cannot remain constant. With implementation-dependent positive constants c,k, an illustrative bound is:

```text
Tₚ(N) ≥ cN; T₁(N) ≈ kN; S(P,N) ≤ k/c
```

I do not treat k/c as a measured numerical limit. Assuming a small fixed α would hide the dominant dependency. A batch of independent systems supplies outer parallelism; cyclic reduction or parallel prefix changes the algorithm. Neither proves that the original single-system Thomas execution scales. The limitation concerns this algorithm, not every solver for the mathematical problem.

## Distributed three-dimensional FFT

Slab/pencil FFT decompositions require global redistributions between directional transforms. FFTW documents distributed transposes explicitly. [F] I use the illustrative model:

```text
Tₚ(N) ≈ k N log(N)/P + τ q(P) + V(P,N)/B_eff(P) + I(P,N)
```

Here N counts grid values, k scales arithmetic time, τ is exchange latency, q counts critical-path exchanges, V is per-rank transferred volume, B_eff is effective bandwidth and I is imbalance. Overlap and collective algorithms change these terms.

At memory-weak scaling N = nP, arithmetic per rank is approximately kn log(nP), not constant. Even a time-scaled family keeping N log(N)/P constant retains transpose costs. With constant useful A+B and illustrative exposed C(P) = c log P, c > 0:

```text
E(P) = (B+A/P)/(A+B+c log P) → 0
```

Coordination is not necessarily useful serial work and must not enter the reconstructed useful-work numerator. Buffers and network bisection bandwidth may also constrain scaling. Better decomposition, placement, collectives and overlap can help; more work per rank changes the size/time family.

I would measure critical-path time for Thomas, and compute/transpose/imbalance/total time for FFT. Comparisons must match system and solver, or grid, precision and transform convention. Larger inputs must remain scientifically useful; initialization and output may grow too. Comparing a small serial input directly with a large parallel input measures time/throughput, not matched-work speedup.

## References

- **[G88]** Gustafson (1988): [DOI](https://doi.org/10.1145/42411.42415).
- **[T]** [LAPACK tridiagonal reference implementation](https://www.netlib.org/lapack/double/dgtsv.f). Its pivoting details differ from standard Thomas, but retain sequential sweeps.
- **[F]** [FFTW MPI transpose documentation](https://www.fftw.org/fftw3_doc/FFTW-MPI-Transposes.html).