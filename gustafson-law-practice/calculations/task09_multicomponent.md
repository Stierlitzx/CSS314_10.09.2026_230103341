# Task 09 — I/O, broadcast, synchronization, and computation

**Student ID:** 230103341

## My timing model

I normalize parallel elapsed time to one and use the assignment's **non-overlapping critical-path fractions**:

- Phase A: combined disk I/O and broadcast, a = 0.02; no internal split is supplied.
- Phase B: boundary synchronization, y = 0.06.
- Phase C: useful parallel flux computation, c = 0.92.

I count each phase once, not overlapping timers or CPU-time sums across ranks. For this calculation only, I treat A and B as non-scaling, assume balanced computation and equal-speed processors, and retain these fractions at both requested processor counts P.

```text
a + y + c = 1
α = a + y = 0.08 = 2/25; c = 1 − α = 23/25
T_ref = a + y + Pc
S_G = T_ref/1 = 0.08 + 0.92P = (2 + 23P)/25
E_G = S_G/P = 0.92 + 0.08/P
```

Here α is a fraction of **parallel runtime**, S_G is model speedup, and E_G is efficiency. I/O, broadcast, and synchronization are not intrinsically serial: concurrency, waiting, overlap, and rank count can change their costs. If some disappear on one processor, T_ref is only a serial-equivalent model, not an actual serial timing prediction.

## My numerical results

| P | Non-scaling contribution | Scalable contribution 0.92P | Exact S_G | S_G decimal | Efficiency (%) |
|---:|---:|---:|---|---:|---:|
| 512 | 0.08 | 471.04 | 11778/25 | 471.12 | 92.015625 |
| 2048 | 0.08 | 1884.16 | 47106/25 | 1884.24 | 92.00390625 |

```text
P = 512:  E_G = 5889/6400 = 0.92015625
P = 2048: E_G = 23553/25600 = 0.9200390625
Ideal-speedup losses:
512 − 471.12 = 40.88 = 0.08 × 511
2048 − 1884.24 = 163.76 = 0.08 × 2047
```

I obtain efficiency approaching 92% and no finite speedup ceiling under fixed α. The matched serial-reference fraction f differs from 8%:

```text
f = α/S_G
f_512 = 1/5889; f_2048 = 1/23553
S_A(512, 1/5889) = 471.12
S_A(2048, 1/23553) = 1884.24
```

I checked these values with exact fractions. For general problem dimension N, useful work W(N), and per-processor rate r, my non-overlapping model would instead be:

```text
T_P(N) = T_IO(P,N) + T_bcast(P,N) + T_sync(P,N) + W(N)/(Pr)
α(P,N) = (T_IO + T_bcast + T_sync)/T_P(N)
```

I would need consistent end-to-end timing and a critical-path model for overlap before extrapolating the assumed 8%.