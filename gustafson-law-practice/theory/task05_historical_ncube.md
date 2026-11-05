# Task 05 — The 1988 Sandia experiments and NCUBE/ten

**Student ID:** 230103341

## Machine and sources

I distinguish Gustafson's short article [G88] from the detailed Sandia study [GMB88]. The latter identifies **NCUBE/ten**, not independently verified “nCUBE10”: 1024 MIMD processors, distributed memory, 512 KBytes/node with error correction, custom VLSI processors with 32/64-bit IEEE floating-point arithmetic, and eleven bidirectional DMA channels. Since 2¹⁰ = 1024, the hypercube has ten neighbors/node and diameter ten—not eleven. [GMB88, §3]

## Reported findings

I retain each publication's figures separately; neither source establishes every cause of their differences.

| [G88] application | Method | Speedup |
|---|---|---:|
| Beam stress | Conjugate gradients | 1021 |
| Baffled surface wave | Explicit finite differences | 1020 |
| Unstable fluid flow | Flux-corrected transport | 1016 |

[G88] reports 0.4–0.8% serial time, not exact application-specific fractions. Its scaled derivation uses parallel-system quantities s′,p′.

| [GMB88], Table 10 | Scaled speedup | Fixed speedup | Scaled MFLOPS | Fixed MFLOPS |
|---|---:|---:|---:|---:|
| Wave, finite differences | 1020 | 637 | 113 | 71 |
| Fluid, flux-corrected transport | 1009 | 519 | 69 | 36 |
| Beam strain, conjugate gradients | 1019 | 502 | 132 | 67 |

Thus its ranges are 1009–1020 scaled and 502–637 fixed.

| Reported quantity, §§5.5–7.5 | Wave | Fluid | Beam |
|---|---:|---:|---:|
| Parallel-system α = s′ | 0.0034 | 0.013 | 0.005 |
| Percentage | 0.34% | 1.3% | 0.5% |
| Approximate one-core s_A | 0.000003 | 0.00001 | 0.000005 |

Section 8 summarizes α ≈ 0.003–0.013 and s_A ≈ 0.000003–0.00001, as fractions. Wave efficiency is approximately 99.66%, with loading, transition and arithmetic-imbalance losses. Fluid operation efficiency is 99.5%, plus approximately 1.0% communication/loading/startup loss; beam operation efficiency is 99.6%, plus approximately 0.1%. These are rounded accounting quantities, not uniquely measured serial-code times.

## Reference construction and interpretation

Fixed problems fit one node and used production-relevant, long runtimes. Large scaled references instead use measured rates/counts and a hypothetical processor accessing ensemble memory: wave data require 340 MBytes versus 512 KBytes/node. The 6144×6144 grid's 10,000-second total and 40-second overhead illustrate calculation, not universal measured timings. Extrapolated one-core times are approximately five months (wave), 3.6 years (fluid), and twenty years (beam).

I define C₁ as best-serial operation count, Cₚ as total parallel count, τ as operation time and h as exposed overhead:

```text
η_op = C₁/Cₚ ≤ 1
S = C₁τ/(Cₚτ/P+h) = P η_op × (Cₚτ/P)/(Cₚτ/P+h)
```

Redundant operations can keep processors busy without useful speedup. Fitting the ideal law gives **effective**, not reported, fractions:

```text
α_eff = (1024−S)/1023
Wave: 4/1023 ≈ 0.00391007 = 0.391007%
Fluid: 15/1023 ≈ 0.01466276 = 1.466276%
Beam: 5/1023 ≈ 0.00488759 = 0.488759%
s_A = αₚ/[P−(P−1)αₚ]; αₚ = P s_A/[1+(P−1)s_A]
```

Hypothetical α₁₀₂₄ = 0.004 gives S = 1019.908 and s_A ≈ 3.92192×10⁻⁶ = 0.000392192%, not 0.4%. I conclude that scaled workloads complement, rather than refute, Amdahl's fixed-work limit.

## References and access notes

- **[G88]** J. L. Gustafson, “Reevaluating Amdahl's Law,” *CACM* 31(5), 532–533 (May 1988): [text](http://www.johngustafson.net/pubs/pub13/amdahl.htm), [publisher](https://doi.org/10.1145/42411.42415). Its reference [2], Benner/Gustafson/Montry's SAND 88-0317, “Development and analysis of scientific application programs on a 1024-processor hypercube,” was not independently consulted.
- **[GMB88]** Gustafson, Montry and Benner, “Development of Parallel Methods for a 1024-Processor Hypercube,” *SIAM J. Sci. Stat. Comput.* 9(4), 609–638 (July 1988): [text](http://www.johngustafson.net/pubs/pub16/1024.html), [landing page](http://www.johngustafson.net/pubs/pub16/pub16.htm), [Crossref metadata](https://api.crossref.org/works?query.title=Development%20of%20Parallel%20Methods%20for%20a%201024-Processor%20Hypercube&rows=1). HTML notes minor 1995 revisions. Existing access notes record [publisher](https://epubs.siam.org/doi/10.1137/0909041) HTTP 403 and [suggested Sandia PDF](https://www.sandia.gov/app/uploads/sites/143/2024/02/Gustafson1988.pdf) HTTP 404; neither supplies the numerical claims.