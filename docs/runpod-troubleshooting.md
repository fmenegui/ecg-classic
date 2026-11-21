# RunPod troubleshooting: every problem we hit, and what each one cost

How to recognise each failure, confirm it, fix it, and the safeguard that now prevents it. Costs
are estimates from run durations × the hourly price printed in the log (Nov 16-20, 2026). About
28 of the ~53 USD spent in those five days went to runs that produced nothing. The goal of this
page is that none of these repeats.

## Quick triage

| Symptom in the run log | Most likely cause | Section |
|---|---|---|
| `RunPod balance X USD is below ...` | balance too low (guard refused the pod, nothing spent) | [1](#1-balance-at-zero) |
| Several runs fail at the same moment after hours | balance reached 0, RunPod stopped every pod | [1](#1-balance-at-zero) |
| `[waiting: 404 HTTPError]` / `520` / `502` for many minutes, then `the pod never answered` | pod not reachable through the proxy (data center or volume outage, image still pulling) | [2](#2-pods-that-never-answer-through-the-proxy) |
| `could not send <file>` after an hour | same as above | [2](#2-pods-that-never-answer-through-the-proxy) |
| Log stops, run continues to the 6 h limit, artifact empty | follower hung, or GitHub killed the job at 355 min | [3](#3-the-6-hour-github-limit), [4](#4-the-follower-hung) |
| `the job failed on the pod (exit code N)` | the job itself crashed; read the pod log | [5](#5-failed-jobs-reported-as-success) |
| `exit code 124` | the job hit its own time limit (`max_hours`) | [10](#10-evaluations-killed-by-their-time-limit) |
| `exit code 137` or `Killed` | out of memory on the pod | [6](#6-out-of-memory-on-small-hosts) |
| `nogpu`, CUDA errors, clone fails | broken host; replaced automatically | [7](#7-broken-hosts) |
| `allows 64 concurrent vCPUs` | account vCPU limit; the pod waits in line | [8](#8-the-64-vcpu-account-limit) |
| `There are no longer any instances available` | no machine of that kind free in that data center | [9](#9-no-machine-available) |
| `no signed URL: HTTP 404` on a pod, 302 on GitHub | pod image; use the PyTorch image | [11](#11-the-kaggle-archive-answers-404-on-a-pod) |
| `TOO MANY REQUESTS` (Zenodo or Kaggle) | rate limit; back-off and retries handle it | [12](#12-rate-limits-zenodo-kaggle) |
| CPU load far below the vCPU count | the work is not parallel (downloads, batch waiting for its slowest item) | [13](#13-paying-for-idle-cpus) |
| A pod listing or environment in a log | **security incident**: delete the run, rotate if needed | [14](#14-secrets-in-logs) |

---

## 1. Balance at zero

- **What happened.** On Nov 18-19 two GPU trainings stopped after 3-4 hours. RunPod stops every
  pod of the account when the balance reaches 0. On Nov 19 at 17:25 a new run would have pushed the
  balance under zero while three pods were running.
- **Cost.** About 6-7 USD of training thrown away (runs 44420706651 and 44427796105, 3090 at
  0.5 USD/h, 6 h each, no usable result).
- **Prevention, now in the code:**
  - `runpod_ctl.py` refuses a GPU pod unless the balance covers `max_hours` × 0.5 USD/h.
  - `runpod_v2.py` refuses a CPU pod unless the balance covers its hours × price + 2 USD.
  - Checkpoints are copied every 30 min, so a stop costs at most 30 min of training.
- **Practice.** Before launching a batch, add up the worst case of every running and planned pod.
  The guards look at one pod at a time, so they cannot see the others running.

## 2. Pods that never answer through the proxy

- **What happened.** From Nov 19 ~15:20 UTC no pod in EUR-IS-1 with the network volume mounted
  answered through `https://<pod>-8000.proxy.runpod.net`; it gave 404, 502 or 520 for hours.
  - The row-network training's log stopped at 15:19. It trained 5 h on the volume, and its
    checkpoints are still stuck there.
  - Two 32-vCPU evaluation pods (Nov 19, 23:10) and a pod-job (Nov 20, 16:47) ran an hour each
    without answering.
  - Dewarp v2 could not receive its init checkpoint for an hour (Nov 19, 23:10).
  - Three volume-fetch pods never answered.
  - A dc-probe on Nov 20 showed that pods in EUR-IS-1 *without* the volume did answer, so the
    mounted volume was the trigger.
- **Cost.** About 5-6 USD.
- **How to confirm.**
  - Download the run's `podid-*` artifact and `curl` the proxy (`hybrid/gpu/podlog.sh RUN_ID`).
  - A 404 or 520 for more than ~25 min after start means the pod will not come.
  - Run `dc-probe.yml` for the data center, with and without the volume.
- **Prevention, now in the code:**
  - The follower gives up on a pod that never answered within `BOOT_MIN` = 35 min, instead of
    waiting up to 5 h 40.
  - The volume moved to `ecgdig-data2`, created in EU-RO-1 first.
  - `use_volume` (training) is off by default.
  - The pod id is uploaded as an artifact right after the start, so a running pod can be inspected.
- **Misdiagnosis to avoid.**
  - We first blamed the 8 GB CUDA image ("pulling for an hour") and moved CPU pods to
    `python:3.11-bookworm`. That broke Kaggle access (section 11).
  - We blamed 98 MB uploads ("HTTP 502 on large bodies") and added 16 MB chunks. The chunks are
    harmless and stay, but the cause was the outage.
  - Test one variable at a time with a cheap probe pod before changing the infrastructure.

## 3. The 6-hour GitHub limit

- **What happened.** GitHub ends a job at 360 min; our jobs set 355. A training of 280 min plus a
  slow setup (realstrips and bank built on the pod) ran into it. Run 44546313205 was cancelled at
  5 h 55: the cancelled job uploaded nothing, and its checkpoints stayed on the volume.
- **Cost.** About 1.7 USD and 5 h of training.
- **Prevention, now in the code.** The workflows record the job start (`JOB_T0`). The follower stops
  at start + 340 min, copies the checkpoints once more, uploads them and fails the run.
- **Practice.**
  - `--minutes` ≤ 285 for a 5.6-5.8 h run.
  - Anything long that is not training (strips, banks) is built beforehand on the volume
    (`pod-job.yml`) and reused through `--cache`.

## 4. The follower hung

- **What happened.** A response trickling through the proxy held `urllib`'s read for 5 h. Its
  timeout applies to each socket operation, not to the whole transfer. The run saw neither the log
  nor the checkpoints of a pod that kept training.
- **Prevention, now in the code.**
  - Every transfer has a hard total limit (4 × its timeout).
  - A log that has not grown for 25 min (`STALL_MIN`) means a dead job, because every pod logs a
    heartbeat every 2 min.

## 5. Failed jobs reported as success

- **What happened.** The page v3 training crashed in its first minutes (Nov 19, run 44558780904).
  The run was green and started a useless benchmark.
- **Prevention, now in the code.** The follower reads the job's `exit code N` from the pod log and
  fails the run when it is not 0, so no benchmark starts.

## 6. Out of memory on small hosts

- **What happened.** The row network died twice with exit 137 while building real strips in
  parallel on 8-vCPU hosts with little RAM.
- **Prevention, now in the code.**
  - GPU pods ask for at least 46 GB RAM (`MIN_RAM_GB`).
  - Fewer strip workers on the training pod.
  - Strips are now built once on a CPU pod (`pod-job.yml`), not during the training.

## 7. Broken hosts

- **What happened.** Some hosts had broken DNS (the clone failed), a broken CUDA, or no access to
  GitHub.
- **Prevention, now in the code.**
  - The clone is retried.
  - A pod whose CUDA check or clone fails writes `nogpu` and is replaced by another pod, preferring
    secure cloud.
  - The replacement can be dearer: a secure-cloud 3090 costs 0.5 USD/h against 0.22 on community
    cloud.

## 8. The 64-vCPU account limit

- **What happened.** The account may run 64 vCPU of CPU pods at once. Five development benchmarks
  at 32 vCPU each were started together. Three failed after 30 min with
  `allows 64 concurrent vCPUs`.
- **Prevention, now in the code.** On that answer the pod waits in line up to `QUEUE_MIN` = 150 min.
- **Practice.** Plan at most two 32-vCPU pods at a time, or use 16 vCPU each.
- **Second case (Nov 20).** `env-check.yml` was started while two PM dev evaluations held 2 x 32
  vCPU. Its 2-vCPU test pod waited in line for 60 min and the job was cancelled without a verdict.
  Now env-check waits only 5 min (`QUEUE_MIN=5`) and reports `FAIL ... 64 vCPUs are in use`.
  Run env-check when no 32-vCPU pod is running (GPU pods do not count toward the limit).

## 9. No machine available

- **What happened.** `There are no longer any instances available` for some CPU flavours
  (cpu3g, cpu5g) in some data centers. Cheap GPUs are sometimes all taken.
- **Prevention, now in the code.** Other flavours and other data centers are tried. GPU starts
  retry every 3 min for 45 min.
- **Practice.** Do not widen the price cap to get a card faster. The waiting is cheaper than
  0.5-0.9 USD/h cards.

## 10. Evaluations killed by their time limit

- **What happened.** Development benchmarks with the page network or the dewarp network read each
  page twice (select mode). Four runs hit `max_hours` = 1.5-2 and ended with `exit code 124` and no
  results.
- **Cost.** About 7 USD (runs 44653812007, 44739837330, 44754015663, 44773619533).
- **Prevention, now in the code:**
  - `experiment.yml` has `pm_hours` (2.5 by default; 3 for page or dewarp).
  - The evaluator keeps every core busy (section 13).
- **Practice.** Estimate the time from a previous run of the same configuration before choosing
  `max_hours`.
- **Fifth case (Nov 20, 44787684688, rowheavy3e + page v3, ~3.4 USD).** Even at 3 h the PM dev
  part did not finish (dewarp v2 on the same images took 1.7 h; the page network on CPU is
  slower), and the run again kept nothing: the evaluator printed nothing until the end, so the
  log could not tell slow from stuck.
- **Prevention, now in the code (Nov 20):**
  - `eval_pm.py` prints `eval progress: k of N images, ETA` to the log once a minute.
  - It stops at `ECGDIG_DEADLINE`, which `podeval.sh` sets 15 min before the pod's limit (60% of
    it when a Kaggle part follows), and writes the results of the images read so far, with
    `PARTIAL: n of N ECGs` and `not_read` in results.json. Coverage must then be reported.
  - The per-image cache stays on the volume under `/workspace/runs/<run>-0/pm/cache`.

## 11. The Kaggle archive answers 404 on a pod

- **What happened.** After CPU pods moved to `python:3.11-bookworm`, the Kaggle archive answered 404
  to every pod in every data center (dc-probe 44772888742). It answered 302 to pods on the RunPod
  PyTorch image (dc-probe 44774777374) and to GitHub runners. Development benchmarks lost their
  Kaggle part, and the PM part ran alone.
- **Prevention, now in the code.** Every pod uses the PyTorch image (`CPU_IMAGE` default). The log
  shows the token kind and length and each HTTP answer, never the token.
- **Practice.** Change the pod image only with a dc-probe comparing both images.

## 12. Rate limits (Zenodo, Kaggle)

- **What happened.**
  - Zenodo answers 429 to PM-ECG-ID downloads after a burst; the first full download took 74 min.
  - Kaggle answers 429 when 20 runners ask for the archive at once.
- **Prevention, now in the code.**
  - Back-off with jitter, reopening the archive, and shards started with a stagger.
  - The data is kept on the volume, so it is downloaded once.

## 13. Paying for idle CPUs

- **What happened.**
  - **Prefill on 32 vCPU:** a 32-vCPU pod spent 1.5 h downloading at 1.13 USD/h. Downloads do not
    use CPUs. Cost about 1.7 USD.
  - **Prefill on 4 vCPU:** the same prefill on 4 vCPU read the 977 Kaggle records through one
    stream and had not finished after 4.5 h (0.66 USD).
  - **Batch evaluator:** it waited for the slowest image of each batch of 128, holding a 32-vCPU pod
    at a load of 3-5 for an hour.
  - **Real-strips builder:** it read pages through one stream, about 26 pages a minute, while 30
    labelling processes waited.
- **Prevention, now in the code.**
  - Prefills get 4 vCPU.
  - Downloads use parallel range readers: `kaggle_zip.py --threads`, `realstrips_build.py --readers`.
  - The evaluator keeps n images in flight with rolling executors.
  - Every pod logs its CPU load every 2 min, so idle pods show up in `podlog.sh`.
- **Practice.** Look at the first heartbeat lines of every new kind of job. A load far below the
  vCPU count means money is being wasted.

## 14. Secrets in logs

- **What happened.** On Nov 19 a read-only probe printed the pod listing, which includes each pod's
  environment and so the Kaggle token, to a public workflow log. The run was deleted at once and the
  probe changed. The owner chose to keep the same token.
- **Prevention, now in the code.**
  - No tool prints a pod listing or a pod's environment.
  - `runpod_v2.py stop` looks pods up by name in memory and prints nothing about them.
  - Upload tokens are masked (`::add-mask::`).
- **If it happens again:** delete the run (`gh run delete`), rotate the leaked secret, update the
  GitHub secret.

## 15. Wrong card for the job

- **What happened.** The first trainings (Nov 16-17) ran on RTX 4090 (0.89 USD/h) and once on an
  L40S (1.09). The training data comes from a CPU generator, so the 4090 waited for data 23-37% of
  the time. Several were cancelled.
- **Cost.** About 5 USD.
- **Prevention, now in the code.** Only cheap cards are on the list (A5000, 3090, A4500, 4000 Ada,
  A4000, L4, ...) under a 0.5 USD/h cap. A dearer card runs only when named.
- **Practice.** What matters is cost per training step, not per hour. Profile with
  `hybrid/gpu/profile_gen.py` (about 0.1 USD) before choosing.

## 16. Rendering, not the GPU, was the bottleneck

- **What happened.** On an RTX 4000 Ada, rendering took 80-90% of each step of the page and dewarp
  networks.
- **Fix.** `--reuse N` draws each rendered page N times with new crops and degradations. Page v3 ran
  38,190 steps in 290 min against 18,593 for v2, and dewarp v2 ran 75,827 against 12,041, at the
  same price.
- **A bug to remember.** The first `--reuse` crashed on mixed flat and photo pages, whose metadata
  differ (Nov 19, run 44558780904). Smoke-test a new data path for 10 min before a 5 h run.

## 17. Code that only works where it was written

- **What happened.** The dewarp network's `flatten` used torch. The evaluation hosts (CPU pods,
  GitHub runners) have no torch, so every dewarp reading failed silently with
  `ModuleNotFoundError` and the run reported the classical reading. Two benchmarks (about 1.2 USD)
  measured nothing, and the result "dewarp: 0.00 dB" was wrong.
- **Fix.** `flatten` now uses numpy and OpenCV only (same output, 7× faster).
- **Practice.** A new component that changes nothing at all in a benchmark is suspect. Check the
  per-image fields (`dewarp`, `select`) for errors before concluding.

## 18. Smaller ones

| Problem | Fix |
|---|---|
| Container disk 40 GB refused on 2 vCPU | disk = min(40, 10 × vCPU) |
| A cancelled download left a truncated PNG | atomic writes through per-writer temp files; truncated PNGs fetched again |
| Pods sharing the volume wrote the same file | per-writer temp names |
| Artifact downloads failed (HTTP 503) when 20 shards fetched the same model | one job fetches the models; the shards read them from the Actions cache |
| `eval_pm` stopped at the first crashed image (`BrokenProcessPool` on submit) | a new executor is started |
| Run annotations, meant to show the pod id, appear only when the job ends | pod id uploaded as an artifact right after the start (`podlog.sh`) |
| The 32-vCPU real-strips job ran on the broken EUR-IS-1 volume for an hour | cancelled; the volume moved; `BOOT_MIN` |

## 19. Scoring crashed after a good run (Nov 20, 44801832861, ~0.7 USD)

- **What happened.** Open-ECG-Digitizer read all 540 Kaggle dev images in 2.5 h, then the scoring
  started `nproc` = 128 processes (the host's cores, not the pod's share). OpenBLAS could not create
  its threads and the pool died after 45 images. The script printed `done` and exit code 0.
- **Prevention, now in the code.** The top-solution runners use at most 16 processes with one BLAS
  thread each (`TS_JOBS` to change it), and fail the job when fewer images are scored than read
  (`Kaggle scoring coverage: n of m`).
- **Practice.** On GPU pods never size a pool with `nproc`; read the coverage line before trusting
  a number.

## 20. Free GitHub runners killed by memory (Nov 20-21)

- **What happened.** `pmcardio.yml` shards with the page network died with `The runner has received
  a shutdown signal` at 14-15 of their 20 images, again and again (shards 7, 17, 0; whole-database
  shards 12 of 42). The progress line showed memory at 15.3-15.5 of 16 GB with 3-4 phone photos
  (16 Mpx) in flight, ECG 47 among them each time.
- **Cost.** Nothing in money (free runners), but runs lost 5-40% of their coverage.
- **Prevention, now in the code.** The progress line prints memory and the images in flight;
  `pmcardio.yml` runs 2 workers and caps each at 7 GB (`ECGDIG_MEM_LIMIT_MB`), so an image that
  needs more fails alone with MemoryError.

## Rules that save money

1. **Look at every new kind of job in its first 10 minutes** (`podlog.sh`). Check that it started,
   that the load matches the vCPUs, that the GPU is busy and that the progress rate is plausible.
2. **Smoke-test before a long run:** 10-20 minutes, same job, same inputs.
3. **Cancel what does not answer:** no proxy answer in 25-35 min means it will not come.
4. **Never pay for waiting:** downloads on small pods or free runners; evaluations on free runners
   unless time matters; one stream per 30 workers is a bug.
5. **Check the balance against everything that will run, not one pod.**
6. **One variable at a time when diagnosing**, with a probe that costs cents.
7. **A result of exactly zero change is a bug until proven otherwise.**
