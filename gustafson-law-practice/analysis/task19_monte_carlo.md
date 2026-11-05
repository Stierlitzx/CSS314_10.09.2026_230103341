# Task 19 — Monte Carlo: nearly independent work, nonzero coordination

**Student ID:** 230103341

## 1. A favorable scaled workload

Consider independent Monte Carlo samples \(X_i\) with mean \(\mu\) and finite variance \(\sigma^2\). The estimator is

\[
\widehat\mu_M=\frac1M\sum_{i=1}^M X_i,
\qquad \operatorname{Var}(\widehat\mu_M)=\frac{\sigma^2}{M}.
\]

Suppose each worker generates and evaluates \(m\) samples at average cost \(k\) seconds per sample. The scaled job has \(M=Pm\), useful serial setup \(A\), and useful worker compute time \(B=km\). Neglecting coordination,

\[
T_P=A+km,\qquad
T_1(M)=A+kPm,\qquad
S=\frac{A+kPm}{A+km}.
\]

This is a strong setting for Gustafson's interpretation: extra processors can produce more samples in nearly the same time. Standard error falls by \(P^{-1/2}\) relative to \(m\) independent samples; it does not fall by \(P^{-1}\). Halving standard error requires four times as many samples under these assumptions. Correlated Markov-chain samples require an effective sample-size analysis instead.

## 2. Keep the tree reduction in the denominator

Each worker can retain a sample count, sum and second-moment statistic, rather than send all samples. Assume a fixed-size aggregate of \(r\) bytes, a balanced binary reduction, and \(P\) a power of two. A simple unoverlapped model is

\[
C_{\rm red}(P)=\left(\tau+\frac{r}{\mathcal B}\right)\log_2P
=\ell\log_2P,
\]

with per-stage latency \(\tau\), effective bandwidth \(\mathcal B\), and aggregate stage cost \(\ell\). For other counts use a suitable tree depth, approximately \(\lceil\log_2P\rceil\). An all-reduce may require another dissemination phase; do not silently equate it to root-only reduction.

Including exposed stream setup \(C_{\rm rng}\) and output \(C_{\rm out}\),

\[
\boxed{S_{\rm useful}(P)=
\frac{A+kPm}{A+km+\ell\log_2P+C_{\rm rng}(P)+C_{\rm out}(P)}}.
\]

Reduction communication is not useful sample work and does not belong in the numerator. With fixed \(m\), negligible additional terms, and positive \(\ell\), speedup is asymptotically \(\Theta(P/\log P)\) and efficiency tends to zero. “Embarrassingly parallel” describes the sample loop, not an assertion of zero total coordination.

As an illustrative calculation, let \(A=0.01\) s, \(km=1\) s, \(P=1024\), and \(\ell=10\) microseconds. Tree depth is 10 and reduction cost is 0.0001 s. Then

\[
S=\frac{1024.01}{1.0101}\approx1013.771,
\qquad E\approx0.99001.
\]

These assumed costs show why finite-scale efficiency can be excellent even though its infinite-scale limit is zero. A centralized sequential collection of \(P\) partials can instead expose \(\Theta(P)\) work at the root. Writing every sample also changes output volume from \(O(1)\) aggregate statistics to \(\Theta(Pm)\) values.

## 3. Reproducible randomness is part of correctness

Assigning a different integer seed to each rank is **not a general guarantee** of independent or nonoverlapping pseudorandom streams. Seeds can initialize nearby or correlated states; varying \(P\) can also change the result solely through stream assignment. Suitable designs include:

1. **Block splitting/skip-ahead:** reserve disjoint, bounded-length subsequences of a generator whose jump operation and period are documented. Account for variable numbers of random draws per sample and prevent block overlap.
2. **Counter-based generation:** derive random blocks from a key and a counter identifying the experiment, global sample index, draw index and block/lane. Use an injective mapping within the supported counter range and avoid reusing the same key/counter/lane for different requested draws.

Salmon, Moraes, Dror and Shaw's Random123 work introduces counter-based generators for parallel computation; their authors' page explains that the \(N\)th random output can be obtained by stateless mixing rather than sequential state iteration. [R123] Disjoint counter assignments prevent accidental reuse of the same generator input; they do not make a deterministic generator mathematically independent in the literal probabilistic sense. Use a tested generator and its documented stream construction, and validate the statistical application.

Mapping random inputs to **global sample IDs**, not worker ranks, lets the same fixed collection of samples be reconstructed for different \(P\). Floating-point sums may still differ with reduction order. Use a documented deterministic tree or reproducible accumulation if bitwise equality matters; otherwise validate within a justified numerical tolerance. For integer hit counts, such as Monte Carlo estimation of \(\pi\), check accumulator overflow.

## 4. What to measure

Report samples/s, total samples, standard error/confidence interval, PRNG algorithm and stream mapping, setup, evaluation, reduction, output and total time. For fixed-work speedup, run the same \(M\) samples on one and many workers. For scaled-work speedup, use the serial time of the enlarged \(Pm\)-sample job or label a clearly modeled extrapolation. Do not compare the small serial job to the large parallel job and call their runtime ratio scaled speedup. Variable-cost samples may also need scheduling to avoid a long-tail straggler.

## Primary source

- [R123] J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, “Parallel Random Numbers: As Easy as 1, 2, 3,” SC11 (2011): <https://doi.org/10.1145/2063384.2063405>; [authors' Random123 explanation and paper links](https://www.thesalmons.org/john/random123/). Timing and variance calculations above are analytical, not measured Random123 performance.
