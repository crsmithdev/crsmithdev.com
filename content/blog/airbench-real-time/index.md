+++
title = "Timing a CIFAR-10 speedrun end to end"
date = 2026-09-21T21:00:00-07:00
type = "post"
+++

[airbench](https://github.com/KellerJordan/cifar10-airbench) trains a small convnet to 94% test accuracy on CIFAR-10 in about 2.6 seconds on an A100. Here I reproduce it, collect timing data from a *real-time attack* (RTA) run, which speedrunners use for the wall clock from start to finish, to see how the infrastructure performs, and look at which tricks in airbench matter the most.

I haven't published anything for a bit, so this is a straightforward warm-up for future posts on more complex and interesting topics.

The code and raw results are in [speedrun-rta](https://github.com/crsmithdev/speedrun-rta).

## Timing


I'm using airbench unchanged and there's no new method or record here; the reproduction and ablations repeat published results. I'm using them to check that the record holds and to put each trick's share of the accuracy on one scale. I also measured real time and cost: I timed every phase from container start to the first result on nine GPU types, with and without a compile cache, priced each, and noted where things broke.

## Reproduction

I ran `airbench94_muon.py` unchanged on torch 2.11, with 200 seeds per GPU. The H100 and A100 runs were on [Modal](https://modal.com); the RTX 5070 is local.

| GPU | Mean accuracy | Std (points) | Train time (s) |
| - | - | - | - |
| H100 | 93.99% | 0.12 | 1.34 |
| A100 40GB | 94.01% | 0.12 | 2.65 |
| RTX 5070 (local) | 94.00% | 0.13 | 6.91 |

The A100 comes in 2.3% above the published 2.59 s, which was measured on torch 2.4.1.

## Ablations

Next I turned off one trick at a time, leaving the rest in place, and ran 50 seeds per setting on the 5070. The standard error of each difference is about 0.025 points.

{{< svg "ablations.svg" "Accuracy lost when each trick is turned off, in points. RTX 5070, 50 seeds per setting; test-time augmentation from 200 seeds." >}}

Two tricks carry most of the weight: test-time augmentation and the patch-whitening init are each worth about 0.8 points. The rest are worth 0.04 to 0.16 points each. I didn't retune anything after turning a trick off, so each loss is an upper bound.

That caveat matters most for the optimizer. Swapping Muon for normalized SGD at Muon's learning rate gets 80.8%, but that learning rate was tuned for Muon, so the number mostly reflects the tuning. A fairer comparison is airbench's earlier record, which runs tuned SGD for 9.9 epochs against Muon's 8. Both reach 94%, and on the 5070 the Muon version trains 13% faster (6.91 s against 7.91 s).

## Overhead

To measure real time, each call starts a fresh Modal container and records the wall-clock time of three phases:

- **Setup:** from container start to a working CUDA context.
- **Compile:** the warmup run that compiles the model with `torch.compile(mode="max-autotune")`, including Inductor's autotuning and CUDA graph capture.
- **Train:** the first real training run to 94%.

Each GPU type got three calls with an empty compile cache (*cold*) and three with the Inductor and Triton caches left by an earlier call (*warm*).

{{< svg "clock.svg" "Seconds from container start to the first 94% result, the median of 3 calls per row." >}}

From cold, the compile takes 77 to 214 s and training takes 1 to 12 s, so the time the speedrun reports is only 1% to 5% of the time to a first result. A warm cache cuts the compile by 3.4 to 4.6 times.

Modal bills from container start, so the same phases set the cost. At Modal's GPU prices on the day of the runs:

| GPU | First result, cold | First result, warm | $ first result, warm | $ 100 runs, warm |
| - | - | - | - | - |
| L4 | 230 s | 78 s | 0.017 | 0.26 |
| L40S | 170 s | 46 s | 0.025 | 0.20 |
| RTX PRO 6000 | 112 s | 36 s | 0.031 | 0.19 |
| H100 | 132 s | 38 s | 0.042 | 0.21 |
| H200 | 130 s | 34 s | 0.043 | 0.22 |
| B200 | 79 s | 32 s | 0.055 | 0.24 |

For a single result the cheaper GPUs win, because the compile dominates and doesn't get much faster on a faster GPU. Over 100 runs in one container the costs come out close on every GPU: a faster GPU costs more per second and needs fewer seconds, by about the same ratio.

## Problems

I encountered a few unexpected issues while doing this:

- torchvision downloads CIFAR-10 from the University of Toronto. One download took 46 minutes, and another timed out the Modal call after an hour, so the runner now ships prepared tensors with the code.
- For a while the compile cache was slower than no cache at all. The Inductor and Triton cache for this model is 5,165 files (170 MB), and reading it file by file from a Modal Volume took about 0.39 s per file, or 33 minutes. Stored as a single tar file, it unpacks in 1 to 5 s.
- Modal didn't always give me the GPU I asked for. Sometimes it was a different variant of the requested type: an A100 80GB for an A100 40GB request, SXM4 or PCIe cards for A100 80GB, and an A10G for an A10. The compile cache is specific to the device, so this showed up as all-or-nothing cache behavior. Each warm call hit all 10 FX graphs or missed all 10, and every miss was on a different variant from the one most calls of that type got. A cache for these types should be keyed by device name.
- The T4 doesn't work at all. The unchanged script runs out of its 16 GB during the compile warmup, with 12.5 GB sitting in CUDA graph memory pools.
- Some calls waited 17 to 90 s between the request and the container start. The cause is unclear, but this isn't included in the figures here and wasn't billed.

## Limits

- Each real-time cell is the median of only three calls.
- The prices are from a single day.
- The containers used Modal's default CPU and memory, which affects the compile time.

## Afterwards

I've liked working with Modal and expect to use it in future projects and posts, so I pulled out all the Modal-related code into a small library, [modal-lib](https://github.com/crsmithdev/modal-lib). It runs a training script or a server on a Modal GPU from a recipe, and keeps each run's log, phase times, GPU use and cost in a folder on a Modal Volume.
