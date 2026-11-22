# Results on other hardware

The main timings come from one desktop-sized computer. This page reports the same savings workload
on rented cloud machines, and a test of mixed precision on the GPU. The
[glossary](../README.md#glossary) explains the terms.

The workload is the one in the main timings: 10,000 lifelib policies under 1,000 scenarios, which
is ten million rows of up to 1,141 months each. lifelib's own version takes about 2,794 seconds for
this workload on the desktop computer. Each figure below is the best of three runs on processor
cores, or of five runs on a GPU, after a warm-up. Times leave out input preparation, compilation,
and copying to the GPU. Each machine ran once, so treat small differences as noise.

## What an insurer could expect

The table scales the measured rates for the savings model to three sizes of job. The rates are 279
seconds per million rows for lifelib's version on the desktop computer, 0.125 seconds on 64 AWS
cores, and 0.006 seconds on an H100 GPU in double precision.

| Job | Rows | lifelib's version, one desktop | Compiled, 64 rented cores | Cost of the cores | H100 GPU, 16 digits |
|---|---|---|---|---|---|
| 100,000 policies under 1,000 scenarios | 100 million | 7.8 hours | 12.5 seconds | about 1 US cent | 0.6 seconds |
| One million policies under 1,000 scenarios | 1 billion | 3.2 days | 2.1 minutes | about 11 US cents | 6 seconds |
| 10,000 policies, 1,000 scenarios inside each of 1,000 scenarios | 10 billion | 32 days | 21 minutes | about 1.14 US dollars | 1 minute |

An overnight stochastic run becomes seconds or minutes on rented processor cores, with results
identical to the original on the tested portfolios, and with no GPU to find.

Read the table with these limits in mind.

- We measured the 100-million-row job on 64 cores directly. The other figures scale in proportion
  from measured rates, and assume the job splits into batches that fit in memory.
- The figures are projection times. Preparing inputs and writing results are extra. Input
  preparation is not yet optimised: it took 1.6 seconds for ten million rows on the desktop
  computer, which is longer than the projection takes on 64 cores.
- The GPU figures leave out copying data to the GPU. We did not measure the cost of a whole job on
  a GPU.
- Production models have more products and features than these examples. Expect different rates
  and the same pattern.

## Processor cores

The fast model with early exit, in double precision (about 16 digits).

| Machine | Cores used | Seconds for ten million rows |
|---|---|---|
| Desktop computer (NVIDIA DGX Spark, Arm) | 20 | 3.12 |
| AWS c7a.16xlarge (AMD EPYC 9R14) | 8 | 7.04 |
| AWS c7a.16xlarge | 16 | 3.58 |
| AWS c7a.16xlarge | 32 | 1.89 |
| AWS c7a.16xlarge | 64 | 1.18 |
| AWS c7a.48xlarge (AMD EPYC 9R14) | 192 | 0.61 |
| Modal, 64 cores requested | 80 logical | 1.11 |

A workload ten times as large, 100 million rows, took 12.5 seconds on the 64 cores: ten times as
long, so the time grows in step with the number of rows.

Doubling the cores roughly halves the time up to 32 cores. From 32 to 64 cores the time falls by a
further 37%. The 192-core figure is less than a second, so start-up costs inside the run limit it;
a larger workload would use those cores better.

At the on-demand list price of 3.28 US dollars an hour for the 64-core machine, a billion rows of
this model cost about 11 US cents to project. The price was the AWS list price for us-east-1 on 22
November 2026.

## Vectorised code on a GPU

A common way to speed up a vectorised Python model is to run the same code on a GPU. CuPy is a
library that mirrors NumPy on a GPU, so whole-array code moves across with almost no change.
`benchmarks/vectorised_gpu.py` runs the savings model in this style: one month for all ten million
rows, then the next month. The table compares it with the compiled fast model.

| Route | Hardware | Seconds for ten million rows | Speed against lifelib's version |
|---|---|---|---|
| lifelib's vectorised model | Desktop processor | 2,794 | 1 |
| The same arithmetic as whole arrays in NumPy | Desktop processor | 597 | 4.7 |
| Whole arrays on a GPU with CuPy | NVIDIA GB10 | 176 | 16 |
| Whole arrays on a GPU with CuPy | NVIDIA H100 | 12.5 | 224 |
| Compiled fast model | 20-core desktop, no GPU | 3.1 | 896 |
| Compiled fast model | 64 rented cores, no GPU | 1.2 | 2,370 |
| Compiled fast model | NVIDIA GB10 | 1.5 | 1,840 |
| Compiled fast model | NVIDIA H100 | 0.06 | 46,000 |

All rows use double precision. The NumPy figure scales up a run of 200,000 rows. The whole-array
runs cover every month, as lifelib's model does, and the compiled runs on processor cores use early
exit.

The whole-array results equal the compiled results exactly in double precision, because both
perform the same operations in the same order.

Vectorised code on the fastest GPU we tried took about ten times as long as the compiled model on
64 ordinary processor cores, and about 200 times as long as the compiled model on that GPU. Each
whole-array operation reads and writes all ten million values, so the speed of memory limits it on
any hardware. The compiled model takes one policy through all its months and keeps that policy's
few values inside the processor.

In single precision, the whole-array runs took 110 seconds on the GB10 and 9.9 seconds on the H100,
and 1.48% of rows came out undefined. Whole-array code keeps calculating every policy to the last
month. In single precision, the account values of finished policies grow until they overflow. The
compiled model stops each policy when nothing is left in force, so it avoids this.

This test covers one way to run vectorised code on a GPU. A platform that compiles whole formulas
for the GPU could do better than CuPy. We did not test any commercial platform.

## GPUs

Seconds for ten million rows.

| GPU | 16 digits | Seven digits | Mixed precision |
|---|---|---|---|
| NVIDIA GB10 (the desktop computer) | 1.52 | 0.127 | 0.469 |
| NVIDIA L4 (Modal) | 1.49 | 0.179 | 0.476 |
| NVIDIA A10 (Modal) | 1.45 | 0.148 | 0.447 |
| NVIDIA H100 (Modal) | 0.060 | 0.048 | 0.053 |

The H100 is a GPU built for data centres. It runs 16-digit arithmetic about 25 times as fast as the
other three GPUs, and only about 25% slower than seven-digit arithmetic. On a GPU of this kind there
is no reason to reduce precision. On the other three GPUs, 16 digits is 8 to 12 times slower than
seven digits, and no faster than 64 processor cores.

The H100 run waited about 15 minutes for a free GPU on one attempt. The processor instances started
at once.

## Accuracy on the GPU

Differences from the processor result in double precision, on the same ten million rows. The size
measure for a row is its largest present value.

| GPU arithmetic | Portfolio totals | Worst row | Typical row |
|---|---|---|---|
| 16 digits | 3 parts in 10^16 | 6 parts in 10^15 of the row's largest value | 6 parts in 10^16 |
| Seven digits | 0.16 parts per million | 6.7 parts per million | 0.64 parts per million |
| Mixed precision | 0.13 parts per million | 2.8 parts per million | 0.44 parts per million |

These are measured differences on one workload, not guaranteed bounds. The table shows the three
Modal GPUs, which gave the same differences as each other. The desktop GPU's differences were
within 3% of them.

A present value close to zero can differ by a large percentage while the difference stays small
against the size of the policy. For example, net cashflow is the difference between large amounts.
In seven digits, the worst net present value differs from the processor result by more than its own
size, although the difference is under seven parts per million of the row's largest value. Set
error limits for such outputs in money terms, or against a size measure, not as a percentage of
the output.

## Mixed precision

Mixed precision does each month's arithmetic in seven digits and keeps the running totals in 16
digits. It is `ftype="mixed"` in `exactuary/backends/cuda.py`.

It gave a small gain in accuracy: the worst row improved from 6.7 to 2.8 parts per million. Most of
the seven-digit error arises inside each month, in the account value, and the running totals add
little to it. On the three smaller GPUs, mixed precision took about three times as long as seven
digits. It is not worth using in this form. Carrying the account value in 16 digits might help
more, and we did not test it.

## Reproducing these results

```bash
python benchmarks/cloud.py --no-gpu --threads 8,16,32      # any machine with lifelib installed
modal run benchmarks/cloud_modal.py --gpu H100             # needs a Modal account
```

`benchmarks/results/cloud.jsonl` holds the results. The AWS runs used the same script on
Ubuntu 24.04 with the package versions in `benchmarks/cloud_modal.py`.
