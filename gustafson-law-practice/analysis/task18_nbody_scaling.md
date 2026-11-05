# Task 18 — N-body scaling: direct summation and Barnes–Hut

**Student ID:** 230103341

## 1. Constant work per worker is not constant particles per worker

Let \(N\) be the number of particles and \(P\) the number of identical workers. Compare **one iteration/time step**, at a specified accuracy and particle distribution. Direct all-pairs force evaluation has \(\Theta(N^2)\) useful arithmetic. Barnes and Hut's original hierarchical method uses a recursively subdivided spatial tree and describes \(O(N\log N)\) force calculation; the tree is rebuilt each time step. [BH86]

For a transparent analytical comparison, initially neglect setup, communication and imbalance. Write useful parallel time as \(F(N)/P\), and choose \(N_0>1\) at \(P=1\). A fixed compute-time budget requires

\[
F(N_P)=P F(N_0).
\]

This is time-scaled growth of **work**, not necessarily memory-weak scaling. Constants, force tolerances and input distributions must be comparable within each algorithm. A Barnes–Hut approximation is not the same numerical calculation as direct summation unless an accuracy criterion establishes an acceptable comparison.

## 2. Direct all-pairs force evaluation

Using \(F(N)=kN^2\),

\[
kN_P^2=P kN_0^2,\qquad
\boxed{N_P=N_0\sqrt P}.
\]

Thus 64 workers permit eight times as many particles at the same ideal compute time, not 64 times as many. With \(N_P=PN_0\), compute time instead grows by a factor \(P\): doubling each particle's possible partners makes particle-count weak scaling inappropriate for quadratic work. For exact pair counts \(N(N-1)/2\), the corresponding relation is \(N_P(N_P-1)=PN_0(N_0-1)\); the square-root expression is its large-\(N\) approximation.

Storage need not be quadratic. Coordinates, velocities and forces can be stored in \(O(N)\) memory without retaining every pair interaction. An implementation that materializes a full pair matrix has a different memory constraint. Distributed direct summation may require exchanging positions or partial forces; the ideal arithmetic derivation does not pay those costs.

## 3. Barnes–Hut: invert \(N\log N\), not just \(N\)

Assume an applicable \(F(N)=kN\ln N\) model, fixed opening/accuracy settings and comparable spatial statistics. Let \(K=N_0\ln N_0\). Then

\[
\boxed{N_P\ln N_P=KP}.
\]

To solve, put \(x=\ln N_P\). Since \(xe^x=KP\), the principal Lambert function \(\operatorname W_0\), defined by \(\operatorname W_0(z)e^{\operatorname W_0(z)}=z\), gives

\[
\boxed{N_P=e^{\operatorname W_0(KP)}
=\frac{KP}{\operatorname W_0(KP)}}.
\]

For fixed \(N_0\) and \(P\to\infty\), \(\operatorname W_0(KP)\sim\ln(KP)\), so

\[
N_P\sim\frac{KP}{\ln(KP)}=\Theta(P/\ln P).
\]

The asymptotic form is not an exact finite-size substitution. For example, choose \(N_0=1024\) and \(P=64\). Direct scaling gives \(N_P=8192\). Barnes–Hut's implicit equation has a continuous solution \(N_P\approx42613.86\), giving \(N_P\ln N_P/(N_0\ln N_0)=64\). This example illustrates each algorithm's own growth factor, not equal absolute runtimes between them. The naive value \(PN_0=65536\) would increase the modeled Barnes–Hut compute time by **1.6**, since \(\ln65536/\ln1024=1.6\), rather than preserving it.

## 4. The serial ratio cannot be inferred from force complexity alone

Let serial setup/update time be \(A(N)\). Then a more complete model is

\[
T_P(N)=A(N)+F(N)/P+C(P,N)+I(P,N),
\qquad \alpha_P=\frac{A(N_P)}{T_P(N_P)}.
\]

Knowing \(F(N)\) does not determine \(A(N)\), so it does not determine a serial fraction. If \(A\) is constant and overhead negligible, the ideal work-growth paths above preserve \(\alpha\). If, instead, **as an illustration**, \(A(N)=aN\) is serial and \(F(N_P)/P=B\) is kept constant, then:

- direct: \(A(N_P)=aN_0\sqrt P\);
- Barnes–Hut: \(A(N_P)=aKP/\operatorname W_0(KP)=\Theta(P/\ln P)\).

In either case the total time is no longer constant, and \(\alpha_P=A/(A+B)\to1\). This does not prove that real implementations have serial linear setup. Parallel tree construction, distributed initialization and output policy can change it.

For genuinely fixed **total** time \(T_*\), solve

\[
A(N_P)+F(N_P)/P+C(P,N_P)+I(P,N_P)=T_*.
\]

With serial \(A=aN\) and nonnegative other terms, \(N_P\leq T_*/a\); unbounded input growth is impossible. The square-root and Lambert results are therefore ideal compute-budget results or total-time results conditional on bounded/parallelized non-force phases.

## 5. Validation plan and source

Measure force computation, tree construction, communication, integration and output separately; hold force tolerance and time-step policy fixed. Check imbalance under clustered particle distributions. Confirm errors against a direct reference on manageable sizes. Scaling a particle count does not automatically hold physical simulation time or error fixed, so report those quantities too. No empirical performance is claimed here.

- [BH86] J. Barnes and P. Hut, “A hierarchical O(N log N) force-calculation algorithm,” Nature 324, 446–449 (1986), original publication and accessible abstract: <https://www.nature.com/articles/324446a0>, DOI <https://doi.org/10.1038/324446a0>.
