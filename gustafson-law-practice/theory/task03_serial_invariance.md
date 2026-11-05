# Task 03 — What remains invariant in the serial part?

**Student ID:** 230103341

## Three different quantities

I distinguish absolute serial time a(Wₚ), one-core serial fraction s_A and parallel-observed fraction αₚ. With b(Wₚ) denoting one-core parallelizable work, I assume identical processors, balanced execution and no additional overhead:

```text
Tₚ = a+b/P
s_A = a/(a+b)                 αₚ = a/(a+b/P)
```

Gustafson treats unchanged loading, startup, bottleneck and I/O costs as a **first approximation**, not a universal property of growing inputs. [G88]

## Two scaling paths

For fixed work, a, b and s_A stay constant. I derive:

```text
αₚ = P s_A/[1+(P−1)s_A]
dαₚ/dP = s_A(1−s_A)/[1+(P−1)s_A]²
```

Thus αₚ strictly increases for 0 < s_A < 1 and tends to 1 for s_A > 0. The serial section has not grown; the parallel section has shrunk. At s_A = 0 or 1, αₚ stays 0 or 1 respectively.

For scaled work, I hold a fixed and set b(Wₚ) = P b₀, with a+b₀ > 0:

```text
Tₚ = a+b₀                    α = a/(a+b₀)
s_A(Wₚ) = a/(a+P b₀) = α/[P−(P−1)α]
ds_A/dP = −a b₀/(a+P b₀)²
s_A(Wₚ) ~ α/[(1−α)P] → 0     when 0 < α < 1
```

I therefore obtain constant serial seconds and α, but a decreasing one-core fraction.

More generally, for aₚ > 0, constant α in (0,1) is equivalent to bₚ/(P aₚ) = (1−α)/α. This fixes a ratio, not runtime: doubling aₚ and bₚ/P leaves α unchanged while doubling Tₚ. Fixed runtime additionally needs fixed aₚ. The endpoints α = 0 and 1 mean no serial or no parallel phase.

## Numerical comparison

I start with a = 2 s and b₀ = 18 s, giving baseline s_A = 0.1.

| Family | P | One-core b, s | Tₚ, s | s_A | αₚ | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Fixed b = 18 | 1 | 18 | 20 | 0.1 | 0.1 | 1 |
| Fixed | 4 | 18 | 6.5 | 0.1 | 0.307692 | 3.076923 |
| Fixed | 16 | 18 | 3.125 | 0.1 | 0.64 | 6.4 |
| Scaled b = 18P | 1 | 18 | 20 | 0.1 | 0.1 | 1 |
| Scaled | 4 | 72 | 20 | 0.027027 | 0.1 | 3.7 |
| Scaled | 16 | 288 | 20 | 0.006897 | 0.1 | 14.5 |

If instead aₚ = a₀P^γ and bₚ = P b₀, then Tₚ = a₀P^γ+b₀ and αₚ = a₀P^γ/Tₚ. For positive a₀,b₀, γ = 0 recovers invariance; γ > 0 gives growing runtime and αₚ → 1; γ < 0 represents reduced serial work. Growing output, preprocessing, control, communication and redundant operations can invalidate the ideal family. I would record workload, total/serial/communication/idle time and serial-reference construction separately. [GMB88]

## References

- **[G88]** Gustafson, “Reevaluating Amdahl's Law,” *CACM* 31(5), 532–533 (1988): [article](http://www.johngustafson.net/pubs/pub13/amdahl.htm); [publisher](https://doi.org/10.1145/42411.42415).
- **[GMB88]** Gustafson, Montry and Benner, “Development of Parallel Methods for a 1024-Processor Hypercube,” *SIAM J. Sci. Stat. Comput.* 9(4), 609–638 (1988): [§§2.1–2.2.2, §8](http://www.johngustafson.net/pubs/pub16/1024.html), minor 1995 web revisions.