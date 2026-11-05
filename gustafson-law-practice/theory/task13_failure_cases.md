# Task 13 — Two concrete limits of Gustafson-style reasoning

**Student ID:** 230103341

## 1. A conditional identity is not a scalability guarantee

For useful serial work \(A\) and balanced useful parallel time \(B\), an overhead-free run gives

\[
S=\frac{A+PB}{A+B}=\alpha+P(1-\alpha),
\qquad \alpha=\frac{A}{A+B}.
\]

Gustafson's original argument changes the problem size rather than demanding a large speedup for an unchanged small input. [G88] Failure means the application cannot realize the assumed workload family, constant serial fraction, or negligible overhead—not that the algebraic identity is false. Two algorithms expose different limitations.

## 2. Case A: Thomas algorithm for a tridiagonal system

The standard Thomas algorithm solves a tridiagonal linear system using forward elimination followed by back substitution, provided its elimination pivots are nonzero. In its usual formulation, a modified coefficient at row \(i\) depends on row \(i-1\); back substitution depends on the next unknown. Both passes have \(\Theta(N)\) work and \(\Theta(N)\) dependency depth. [T]

For **one system using this algorithm**, merely dividing the rows among \(P\) workers does not produce \(P\) independent forward recurrences. With unit-cost row operations, the work/depth bound gives available parallelism \(W/D=\Theta(1)\). Growing the system to \(PN\) grows both work and the critical path by the same factor. It cannot keep runtime constant while turning the new rows into independent useful worker time.

An illustrative execution model is

\[
T_P(N)\geq cN,\qquad T_1(N)\approx kN,
\qquad S(P,N)\leq k/c,
\]

where constants depend on the implementation. This is a dependency argument, not a measured speedup bound with a known numerical constant. Substituting a small fixed \(\alpha\) into Gustafson's law would assume away the dominant recurrence.

**What can change?** A batch of independent systems provides outer parallelism and can be an excellent scaled workload. Cyclic reduction or parallel prefix formulations change the algorithm and its work/depth trade-off. Those are legitimate remedies, but they do not demonstrate that the original single-system Thomas execution scales. Distinguish “this algorithm has a serial dependency” from “this mathematical problem admits no parallel algorithm.”

## 3. Case B: Distributed three-dimensional FFT

A distributed 3-D FFT has substantial local computation, but common slab/pencil decompositions require global redistributions between transforms along different dimensions. FFTW's MPI documentation explicitly describes distributed transposes and their communication role. [F] Increasing global data so each rank holds the same number of grid values does not ensure constant time: transform lengths, partner counts, contention and exposed communication can all grow.

For \(N\) grid values, a simple analytical model is

\[
T_P(N)\approx k\frac{N\log N}{P}
+\tau q(P)+\frac{V(P,N)}{\mathcal B(P)}+I(P,N),
\]

where \(q\) is the number of latency-bearing exchanges on the critical path, \(V\) is per-rank transferred data, \(\mathcal B\) is effective bandwidth, and \(I\) is imbalance. This is not a universal FFT performance formula. Overlap and collective algorithms change its terms.

At memory-weak scaling \(N=nP\), local useful arithmetic is approximately \(kn\log(nP)\), not constant. Even if a time-scaled family instead keeps \(N\log N/P\) constant, the transpose cost remains. For an illustrative exposed cost \(C(P)=c\log P\) and constant useful elapsed work \(A+B\),

\[
E(P)=\frac{B+A/P}{A+B+c\log P}\longrightarrow0.
\]

The global coordination is not necessarily useful serial work. Folding its elapsed time into a reconstructed useful-work numerator would report fictitious computation. At larger scale, memory buffers and network bisection bandwidth may constrain the workload before arithmetic capacity does.

**What can change?** Better decomposition, topology-aware placement, collective tuning and communication overlap can move the practical limit. Increasing work per rank may amortize latency, but changes the time/size family. A larger FFT can still provide valuable higher-resolution output even when its efficiency decreases.

## 4. Diagnostic checklist

| Question | Thomas single-system example | Distributed FFT example |
|---|---|---|
| Main obstruction | Dependency depth | Redistribution and changing local cost |
| Does larger input guarantee more parallelism? | No | Some, but not communication-free |
| Most revealing measurement | Critical-path time / available tasks | Compute, transpose, imbalance, total time |
| Honest comparison | Same system and solver | Same grid, precision and transform convention |

Also test whether the enlarged input is scientifically useful and whether serial initialization/output grows with it. Scaled speedup is a statement about matched useful work on one versus many processors; comparing the small one-processor input directly to the large parallel input is only a time/throughput comparison.

## Sources

- [G88] Gustafson (1988): <https://doi.org/10.1145/42411.42415>.
- [T] LAPACK's reference tridiagonal solver implementation, showing the recurrences: <https://www.netlib.org/lapack/double/dgtsv.f>. Pivoting variants differ in detail but retain the sequential sweep structure discussed here.
- [F] FFTW authors' MPI transpose documentation: <https://www.fftw.org/fftw3_doc/FFTW-MPI-Transposes.html>.
