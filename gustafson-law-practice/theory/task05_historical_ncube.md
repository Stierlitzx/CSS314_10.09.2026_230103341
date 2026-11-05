# Task 05 — The 1988 Sandia experiments and the NCUBE/ten

**Student ID:** 230103341

## 1. Primary-source scope

This report distinguishes two original 1988 publications:

1. **John L. Gustafson, “Reevaluating Amdahl's Law,”** *Communications of the ACM* 31(5), 532–533, May 1988. This two-page article presents the scaled-speedup argument and gives three application results. The publisher record verifies the publication date and pagination. Its article text was consulted on the author's site. [G88]
2. **John L. Gustafson, Gary R. Montry and Robert E. Benner, “Development of Parallel Methods for a 1024-Processor Hypercube,”** *SIAM Journal on Scientific and Statistical Computing* 9(4), 609–638, July 1988. This detailed experimental paper supplies the hardware identification, methods, application results, efficiency accounting and Table 10 summary. The author's complete HTML presentation was consulted; it explicitly notes minor revisions for its 1995 web presentation. [GMB88]

The short article's reference [2] is the related Sandia report **SAND 88-0317**, “Development and analysis of scientific application programs on a 1024-processor hypercube,” by Benner, Gustafson and Montry. That report title should not be substituted for the SIAM article's title. It is identified here from the short article's bibliography, not represented as independently read. [G88]

The suggested Sandia URL, `https://www.sandia.gov/app/uploads/sites/143/2024/02/Gustafson1988.pdf`, returned HTTP 404 when checked. No historical claim below depends on that unavailable file. The consulted, accessible primary texts are linked in the bibliography.

## 2. Verified machine name: NCUBE/ten, not an unverified “nCUBE10”

Section 3 of the detailed paper is entitled “The NCUBE parallel computer” and explicitly begins:

> “The NCUBE/ten is well suited for parallel speedup research.”

Thus **NCUBE/ten** is the model designation directly supported by the consulted primary text. If a worksheet calls the machine “nCUBE10,” the precise sourced identification for the experiments is **the 1024-processor NCUBE/ten hypercube at Sandia National Laboratories**. The worksheet spelling should not be treated as a separately verified product name. [GMB88, §3]

### Architecture relevant to the measurements

The paper documents the following properties. [GMB88, §§3–3.1]

| Property | Primary-source description |
|---|---|
| Execution organization | A 1024-processor MIMD ensemble |
| Memory organization | Distributed memory; data are exchanged by explicit communication |
| Per-node memory | 512 KBytes, with error correction |
| Node processor | Proprietary custom VLSI processor with integral 32-bit and 64-bit IEEE floating-point arithmetic |
| Communication hardware | Eleven bidirectional direct-memory-access communication channels on the processor chip |

A 1024-node hypercube has dimension \(d=10\), because \(2^{10}=1024\). In the abstract graph, nodes can be labeled by 10-bit strings; an edge changes one bit, giving ten cube neighbors and maximum graph distance ten. This is a mathematical consequence of the stated topology, not an independent manufacturer naming claim. In particular, ten hypercube neighbors must not be confused with the eleven on-chip DMA channels reported in §3.1.

The paper's wave example needs 340 MBytes for its large problem's data structures alone, far beyond a single node's 512 KBytes. This memory fact is central to interpreting the large-problem serial reference. [GMB88, §2.2.2]

## 3. Results in “Reevaluating Amdahl's Law”

The short article reports these speedup factors on a 1024-processor hypercube. Its descriptions are preserved below rather than silently replacing them with the detailed paper's terminology. [G88]

| Application in the short article | Numerical method | Reported speedup |
|---|---|---:|
| Beam stress analysis | Conjugate gradients | 1021 |
| Baffled surface wave simulation | Explicit finite differences | 1020 |
| Unstable fluid flow | Flux-corrected transport | 1016 |

In the same introductory discussion, the article gives a serial-percentage range of **0.4–0.8 percent** for the three applications. Its subsequent derivation explicitly distinguishes parallel-system times \(s',p'\) from the initial one-core times. In the notation used throughout Tasks 01–04, the relevant scaled-run fraction is \(\alpha_P\), not the fixed-work one-core fraction \(s_A\).

**Reporting precision matters:** the short article gives a range, not a table pairing an exact measured \(\alpha_P\) with every listed speedup. Therefore one should report the range as stated, and not assert three exact application-specific measured fractions from it. Nor should one force the rounded speedups and approximate percentage range to satisfy the ideal equation exactly.

For illustration only, if an idealized 1024-processor run has \(\alpha_{1024}=0.004\), then

\[
S_G=1024-1023(0.004)=1019.908.
\]

That calculation demonstrates near-linear scaled speedup for a small parallel-observed fraction. It is not an exact reproduction of the beam result 1021 or a claim about an application-specific timing measurement.

## 4. Detailed results in “Development of Parallel Methods…”

The detailed paper's **Table 10**, “Performance summary for applications studied,” reports both fixed-sized and scaled-sized speedups. At \(P=1024\), the summary is: [GMB88, §8, Table 10]

| Application | Scaled speedup | Fixed-sized speedup | Scaled MFLOPS | Fixed-sized MFLOPS |
|---|---:|---:|---:|---:|
| Baffled surface wave simulation using finite differences | 1020 | 637 | 113 | 71 |
| Unstable fluid flow using flux-corrected transport | 1009 | 519 | 69 | 36 |
| Beam strain analysis using conjugate gradients | 1019 | 502 | 132 | 67 |

The abstract consequently summarizes fixed-sized speedups of **502–637** and scaled speedups of **1009–1020**. These are the detailed paper's results; the preceding section separately records the short article's figures. They should not be combined into one purported table or used to imply a contradiction. The distinction needed for this report is the source, workload/reference protocol, and accounting convention attached to each number. The consulted texts do not justify assigning a particular cause to every numerical difference between the publications.

### Reported serial fractions in the detailed paper

The application discussions explicitly give the following approximate serial fractions using the paper's \(s'\) parallel-system convention. [GMB88, §§5.5, 6.5, 7.5]

| Application | Source's reported parallel fraction \(s'\), denoted \(\alpha_{1024}\) here | Percentage | Source's corresponding approximate one-core fraction \(s\), denoted \(s_A\) here |
|---|---:|---:|---:|
| Wave mechanics | 0.0034 | 0.34% | 0.000003 |
| Fluid dynamics | 0.013 | 1.3% | 0.00001 |
| Beam-strain analysis | 0.005 | 0.5% | 0.000005 |

The summary in §8 states a parallel-system fraction range of approximately **0.003–0.013**, and an equivalent one-core fraction range of approximately **0.000003–0.00001** for the scaled problems. These are dimensionless fractions, not percentages; multiplying by 100 converts them to percentages. The short article's 0.4–0.8% statement and this detailed range must remain attributed to their respective sources.

These \(s'\) values are the paper's reported experimental timing/overhead-accounting quantities. They are not all independently timed, purely algorithmic single-thread sections. The study includes communication, startup, imbalance and operation-efficiency effects in explaining its performance. In particular:

- Wave mechanics: the paper describes scaled efficiency as approximately 99.66%, reports \(s'=0.0034\), and apportions losses among loading, the serial-to-parallel program transition and data-dependent arithmetic imbalance. [§5.5]
- Fluid dynamics: it reports operation efficiency of 99.5%, approximately 1.0% further loss from communication, loading and startup, and \(s'=0.013\). [§6.5]
- Beam strain: it reports operation efficiency of 99.6%, approximately 0.1% further loss from communication, loading and startup, and \(s'=0.005\). [§7.5]

The cited figures are reported at the precision of the original discussions. They should not be reverse-engineered into supposedly exact measurements of a unique \(t_s\).

## 5. Measured timings versus reconstructed one-core reference

For the fixed-sized experiments, a problem small enough to fit a single node was executed across processor counts. The detailed paper explains why it chose production-relevant problem parameters yielding long one-core runtimes rather than benchmarks dominated by startup. [GMB88, §§2.3.1–2.3.2]

For the largest scaled problems, the one-core reference could not always be executed on one physical node because its memory was insufficient. Section 2.2 describes a **hypothetical processor with direct access to the ensemble's memory**, supported by measured execution rates, timing and operation counts. Section 2.2.2 illustrates the calculation using a 6144-by-6144 wave grid. Its 10,000-second total and 40-second overhead are an **illustrative calculation**, not a table of exact measured runtimes for all applications.

The distinction is important: the scaled results are experimentally supported performance estimates under an explicit serial-reference model, not claims that every enlarged problem was directly timed from start to finish on an actual 512-KByte node. Sections 5.5–7.5 explicitly describe extrapolated one-core runtimes, including approximately five months for the large wave problem, 3.6 years for the fluid problem, and twenty years for the beam problem.

### Why operation efficiency is needed

Let \(C_1(W)\) be the operation count for the best serial algorithm and \(C_P(W)\) the total count performed by its parallel implementation. The detailed paper defines the operation-efficiency concept

\[
\eta_{\mathrm{op}}=\frac{C_1(W)}{C_P(W)}\leq1.
\]

Keeping every processor busy does not imply that every operation contributes to the minimal serial work. Redundant local calculations may avoid costly communication but reduce operation efficiency. [GMB88, §2.2.1]

In a simplified accounting model with one-core operation time \(\tau\), per-node computational time \(C_P\tau/P\), and nonoverlapped overhead \(h\),

\[
S=\frac{C_1\tau}{C_P\tau/P+h}
=P\eta_{\mathrm{op}}\frac{C_P\tau/P}{C_P\tau/P+h}.
\]

Useful speedup depends on both operation efficiency and processor-time utilization. This explains why the measured accounting is richer than a single ideal serial fraction.

## 6. Inferred ideal fractions are not measured fractions

If one **chooses** to represent a reported scaled speedup \(S\) by the ideal Gustafson equation, its effective fraction is

\[
\alpha_{\mathrm{eff}}=\frac{1024-S}{1023}.
\]

Applied to the detailed paper's rounded Table 10 values:

| Scaled result | Inferred \(\alpha_{\mathrm{eff}}\) | Inferred percentage |
|---|---:|---:|
| Wave, \(S=1020\) | \(4/1023\approx0.00391007\) | 0.391007% |
| Fluid, \(S=1009\) | \(15/1023\approx0.01466276\) | 1.466276% |
| Beam, \(S=1019\) | \(5/1023\approx0.00488759\) | 0.488759% |

These are **derived effective values**, not the reported \(s'\) measurements in §4. They absorb the total loss into a two-component ideal model and inherit rounding in \(S\). A richer operation-count and timing analysis need not identify its reported overhead components with that single fitted parameter. Reporting both columns with their definitions avoids falsely equating unlike quantities.

For any internally consistent ideal decomposition, the one-core and parallel-observed fractions obey

\[
\boxed{s_A=\frac{\alpha_P}{P-(P-1)\alpha_P}},\qquad
\boxed{\alpha_P=\frac{P s_A}{1+(P-1)s_A}}.
\]

For example, a hypothetical exact \(\alpha_{1024}=0.004\) implies

\[
s_A=\frac{0.004}{1019.908}\approx3.92192\times10^{-6},
\]

or approximately \(0.000392192\%\), not \(0.4\%\). Inserting 0.004 as \(s_A\) into an Amdahl calculation would instead describe a different fixed-work reference.

## 7. Historical and mathematical conclusion

The verified machine is the **1024-processor NCUBE/ten**, and the detailed study demonstrates both substantial fixed-work speedups and near-linear scaled speedups on practical numerical applications. Its results are not an algebraic refutation of Amdahl's law. A fixed workload with fixed positive \(s_A\) has a finite speedup ceiling; an enlarged workload can have a decreasing one-core fraction while its parallel-observed fraction stays small.

The historical lesson is methodological: specify the workload, timing denominator, serial-reference construction and efficiency accounting before comparing speedups. Preserve each primary source's figures and precision, and distinguish the source's reported fractions from fractions inferred by fitting an ideal equation.

## Bibliography — consulted primary sources and access notes

- **[G88]** John L. Gustafson, “Reevaluating Amdahl's Law,” *Communications of the ACM* **31**(5), 532–533 (May 1988). [Consulted author-hosted article text](http://www.johngustafson.net/pubs/pub13/amdahl.htm); [consulted ACM publisher record, DOI 10.1145/42411.42415](https://doi.org/10.1145/42411.42415). The publisher record is used for the publication month and pagination; the article text supplies the argument and application figures.
- **[GMB88]** John L. Gustafson, Gary R. Montry and Robert E. Benner, “Development of Parallel Methods for a 1024-Processor Hypercube,” *SIAM Journal on Scientific and Statistical Computing* **9**(4), 609–638 (July 1988), DOI **10.1137/0909041**. [Consulted author-hosted full text](http://www.johngustafson.net/pubs/pub16/1024.html); [consulted author publication landing page](http://www.johngustafson.net/pubs/pub16/pub16.htm). Relevant locations: §§2.1–2.3 (model and measurement), §§3–3.1 (hardware), §§5.5–7.5 (application accounting), §8 and Table 10 (summary). The HTML explicitly notes minor web-presentation revisions in 1995. The [publisher page](https://epubs.siam.org/doi/10.1137/0909041) returned HTTP 403, so it is not represented as a consulted full text. [Consulted publisher-deposited Crossref metadata](https://api.crossref.org/works?query.title=Development%20of%20Parallel%20Methods%20for%20a%201024-Processor%20Hypercube&rows=1) verifies the bibliographic details and deposited abstract; historical numerical claims above rely on the authors' full text, not a secondary account.
