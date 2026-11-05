# Task 02 — Derivation of Gustafson's law and parameter conversion

**Student ID:** 230103341

## Derivation

I fix P ≥ 1 and workload Wₚ. Let tₛ be serial elapsed time, tₚ balanced parallel-phase elapsed time, and Tₚ = tₛ+tₚ > 0. I assume identical processor rates, full participation during the parallel phase, no imbalance, extra operations or overhead. The same workload therefore takes T₁ = tₛ+P tₚ on one core, not the time for a smaller input. [G88]

Defining αₚ = tₛ/Tₚ, I derive:

```text
1−αₚ = tₚ/Tₚ
S_G = T₁/Tₚ = (tₛ+P tₚ)/(tₛ+tₚ)
    = αₚ+P(1−αₚ) = P−(P−1)αₚ
E_G = S_G/P = 1−αₚ+αₚ/P
P−S_G = (P−1)αₚ
```

Setting Tₚ = 1 only normalizes time units. A straight speedup line with slope 1−α additionally requires αₚ = α across workloads.

During serial execution, P−1 workers are idle. I obtain unused capacity (P−1)tₛ and loss 1−E_G = (P−1)tₛ/(P Tₚ) = (1−1/P)αₚ. Busy processors doing redundant work need not provide high useful-work efficiency. [GMB88]

## Parameter conversion and equivalence

The one-core serial fraction is s_A = tₛ/(tₛ+P tₚ). Substitution gives:

```text
s_A = αₚ/[P−(P−1)αₚ]
s_A[P−(P−1)αₚ] = αₚ
P s_A = αₚ[1+(P−1)s_A]
αₚ = P s_A/[1+(P−1)s_A]

S_A = P/[1+(P−1)s_A]
    = P/[1+(P−1)αₚ/[P−(P−1)αₚ]]
    = P−(P−1)αₚ = S_G
```

Denominators remain positive for P ≥ 1 and fractions in [0,1]. These inverse transformations describe the same workload, not two fractions held constant simultaneously across scaling paths.

## Example and checks

For eight processors taking 10 s, with 1 s serial and 9 s parallel, I obtain:

```text
α₈ = 0.1          T₁ = 1+8×9 = 73 s
S_G = 7.3         E_G = 0.9125
s_A = 1/73 ≈ 0.01369863
α₈ = (8/73)/(1+7/73) = 8/80 = 0.1
S_A = 1/[1/73+(72/73)/8] = 7.3
```

At P = 1, S = E = 1 and s_A = α₁. At αₚ = 0, S = P and E = 1; at αₚ = 1, S = 1 and E = 1/P. Thus 1 ≤ S ≤ P. For fixed P, dS/dαₚ = −(P−1): one percentage point costs 0.01(P−1) speedup. Superlinear measurements require revisiting cache effects, rates, algorithms or work counts.

## References

- **[G88]** J. L. Gustafson, “Reevaluating Amdahl's Law,” *CACM* 31(5), 532–533 (1988): [article](http://www.johngustafson.net/pubs/pub13/amdahl.htm); [DOI](https://doi.org/10.1145/42411.42415).
- **[GMB88]** J. L. Gustafson, G. R. Montry and R. E. Benner, “Development of Parallel Methods for a 1024-Processor Hypercube,” *SIAM J. Sci. Stat. Comput.* 9(4), 609–638 (1988): [§§2.1–2.2.2](http://www.johngustafson.net/pubs/pub16/1024.html), minor 1995 web revisions.