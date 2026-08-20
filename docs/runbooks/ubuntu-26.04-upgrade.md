# Ubuntu 24.04 → 26.04.1 upgrade — pre-stage and validation plan

Written 2026-08-20 against the live box. Measurements are facts as of that date;
re-measure anything that looks stale. Scope is the whole workstation, not just
this repository — the Python and GPU axes cut across every project on it.

**Motivation note:** the original driver was a slurm dependency; that issue is
cleared, so this is update hygiene. Nothing here is time-critical except the
pre-stage items, which are cheap and remove most of the risk.

---

## Verified facts (checked 2026-08-20, not assumed)

| Question | Answer | Source |
|---|---|---|
| 26.04 codename | **Resolute Raccoon**, released 2026-04-23 | Canonical release notes |
| 26.04 default Python | **3.14.x** | Ubuntu package index |
| Is `python3.12` available on 26.04? | **No — the package does not exist in the release at all** | Ubuntu package index |
| mergerfs on 26.04 | **Yes — 2.40.2-5build1, universe.** Same upstream version as the installed `2.40.2~ubuntu-noble` | `packages.ubuntu.com/resolute/mergerfs` |
| mergerfs upstream .deb builds | Upstream (2.42.0) now publishes **no** per-Ubuntu assets — the archive is the supply route | GitHub releases API |
| `nvidia-driver-595-open` on 26.04 | **Yes — `595.84-0ubuntu0.26.04.1`**, identical driver version, rebuilt | `packages.ubuntu.com/resolute/…` |
| torch 2.11.0 on Python 3.14 | **Yes — `cp314` wheels published on PyPI** | `uv.lock` wheel inventory |
| Rest of the locked tree on 3.14 | **Zero blockers.** 22 packages ship `cp314`; 79 are pure-python; the 3 that look like `cp310`-only (`protobuf`, `safetensors`, `torchao`) are `abi3` stable-ABI and run on 3.14 | `uv.lock` wheel inventory |

**Consequences.** The two unknowns that could have vetoed the timing — mergerfs
and the NVIDIA driver — are both resolved in favour of proceeding. The driver is
the *same version* rebuilt, so there is no downgrade risk to the Blackwell/nvfp4
path. mergerfs comes from the archive rather than upstream after the upgrade,
at the same version.

**A related runbook already exists and should be read alongside this one:**
`local_agents/docs/runbooks/os-upgrade-resolute-preflight.md` (2026-08-20)
covers the AppArmor 4.x → 5.0.0~beta1 bump, the `bwrap` profile collision, and
`bubblewrap` 0.9.0 → 0.11.1. Those are that project's exposure, not this one's,
but they affect the same upgrade on the same box.

---

## Part A — Pre-stage before the upgrade

### A0. Per-repo Python decision — pin 3.12 or move to 3.14

Because 26.04 ships **no** `python3.12` package, "pin 3.12" can only mean a
uv-managed standalone interpreter. Both options therefore start from the same
action (A1); they differ only in which version uv installs.

The measured dependency evidence says **this repo's own tree is already
3.14-clean** — zero blockers across 101 wheel-bearing packages. So our pins are
not the constraint here. The constraint is **ComfyUI**: this repository's node
pack has to run inside whatever interpreter ComfyUI's venv uses, alongside other
custom node packs whose 3.14 readiness is not ours to determine.

That makes the split boundary and the Python boundary the same line:

| Repo | Target | Reason |
|---|---|---|
| `Eric_Qwen_Edit_Experiments` (node pack, post-ADR-045) | **Pin 3.12** | Bound to ComfyUI's ecosystem, not to our dependency choices |
| `comfyless_diffusion` (new) | **Move to 3.14** | Only our code; tree is verified clean; take the pain on a known-good OS |

The argument for moving the comfyless side now is that the wheel and dependency
fallout gets found on a machine whose toolchain still works, rather than on one
whose venv is already dead. The measured answer is that there is likely to be no
fallout at all — which is itself worth proving before the OS changes underneath
it.

Note ADR-045 has not been executed yet, so today both live in one repo. If the
upgrade lands first, pin 3.12 here and revisit when the split happens.

### A1. Decouple every venv from system Python  ← highest value

**Measured:** 10 venvs on this box, every one built on `/usr/bin` system
Python 3.12.3. uv is managing no interpreters (`~/.local/share/uv/python/` is
absent). When 26.04 replaces `/usr/bin/python3.12`, all ten break at once.

    code/Eric_Qwen_Edit_Experiments/.venv      3.12.3
    code/local_agents/.venv                    3.12.3
    code/scheduler/.venv                       3.12.3
    code/eric-hunyuan/.venv                    3.12.3
    code/eric-lora-convert/.venv               3.12.3
    code/OneTrainer/venv
    venvs/OneTrainer/venv
    ai-stack-data/comfyui/run/venv
    ai-stack-data/comfy1/run/venv
    ai-stack-data/comfy-dev/run/venv

**Fix, runnable today:** have uv own the interpreter, so the distro's Python
becomes irrelevant to every project.

    uv python install 3.12      # or 3.14 — see A0 for the per-repo call
    # then per uv-managed repo:
    rm -rf .venv && uv sync

Validate immediately after — `./.venv/bin/python3 -c "import torch"` and the
full `just tests` battery. Doing this now converts the largest upgrade risk into
a normal working day, and the rebuilt venv is exercised for a week before the
upgrade rather than discovered broken after it.

The three ComfyUI venvs are not uv-managed and need their own decision: rebuild
against a uv-managed 3.12, or accept rebuilding them post-upgrade. They are
already four months stale (see ADR-045), so "rebuild after" is defensible.

### A2. Resolve the package holds

**Measured:** `libmunge2`, `munge`, `slurm-wlm` are held. Holds complicate a
release upgrade and these predate the decision that cleared the slurm
requirement. Decide deliberately: release the holds, or confirm they are still
wanted and expect the upgrade to argue with them.

    apt-mark showhold
    sudo apt-mark unhold libmunge2 munge slurm-wlm   # if they are stale

### A3. Third-party apt sources

`do-release-upgrade` disables these and leaves `.distUpgrade` / `.save`
artifacts — the previous upgrade's artifacts are still on disk, so the pattern
is confirmed for this box. Inventory of live third-party sources and what each
needs:

| Source | Current | Action |
|---|---|---|
| **mergerfs** | `2.40.2~ubuntu-noble` (upstream build) | **Resolved:** archive ships `2.40.2-5build1` in resolute universe. Upstream no longer publishes per-Ubuntu debs, so the archive becomes the source. Not blocking. |
| docker | `noble` | Rewrite to `resolute` |
| nvidia-container-toolkit | distribution-independent | Verify it still resolves |
| CUDA repo | `cuda-ubuntu2404-...list.disabled` | Already disabled; becomes `ubuntu2604` if re-enabled |
| nvidia-driver | `595-open 595.84-0ubuntu0.24.04.1` | **Resolved:** resolute has `595.84-0ubuntu0.26.04.1`. Same driver, no downgrade. Still remove the stale `590-open` first. |
| nodesource | `node_22.x nodistro` | Codename-independent; note mise also pins node 22.22.2 — decide which wins |
| github-cli, chrome, signal, bruno, protonvpn, qbittorrent PPA, unit193 PPA | noble / stable | Re-enable post-upgrade; none are load-bearing |
| ookla speedtest-cli | **`jammy`** | Already two releases stale; drop or fix |

### A4. Pin container images by digest

**Measured running containers:** `llm-gateway` (nginx), `vllm-router`
(lmcache/lmstack-router), `vllm-minimax-m2.7-nvfp4` (vllm/vllm-openai),
`openwebui` (open-webui:**latest**), aider-benchmark:v0.86.2.

Several run on floating tags, which violates §11 and means a post-upgrade repull
can silently change the image. Capture digests now (`docker image inspect
--format '{{index .RepoDigests 0}}'`) and pin them, so container behaviour is a
constant across the upgrade rather than a second variable.

### A5. Capture golden state

Record the working configuration so post-upgrade comparison is mechanical rather
than remembered:

    mkdir -p ~/upgrade-baseline && cd ~/upgrade-baseline
    lsb_release -a                        > os.txt
    uname -r                             >> os.txt
    nvidia-smi                            > gpu.txt
    dpkg -l | grep -E "nvidia|cuda|mergerfs|docker" > packages.txt
    apt-mark showhold                     > holds.txt
    cp /etc/fstab                           fstab.bak
    cp -r /etc/apt/sources.list.d           apt-sources.bak
    systemctl --user list-unit-files      > user-units.txt
    docker ps -a --format '{{.Names}}\t{{.Image}}' > containers.txt
    mount | grep -E "mergerfs|hf-cache"   > mounts.txt

Plus, from this repo: `./.venv/bin/python3 -c "import torch;print(torch.__version__,
torch.version.cuda, torch.backends.cudnn.version())"` — currently
`2.11.0+cu130 / 13.0 / 91900`.

### A6. Establish a green baseline

Run `just tests` (25 suites), `just typecheck`, `just secrets`, `just sast`,
`just deps-cve`, and `just policy-test` **before** the upgrade and record the
results. A failure after the upgrade is only diagnostic if you know it passed
before. Commit any dirty work first — a broken box mid-recovery is the worst
time to reconstruct uncommitted changes.

---

## Part B — Validate after the upgrade

Ordered by consequence. Stop and fix before moving down.

### B1. Filesystem — highest risk of a silent problem

`/home/gawkahn/projects` is a mergerfs union of three NVMes, with
`/mnt/nvme-8tb/hf` bind-mounted to `.../ai-base/models/hf-cache` and ordered by
`x-systemd.requires-mounts-for=/home/gawkahn/projects`. Everything lives here.

    mount | grep -E "mergerfs|hf-cache"
    ls /home/gawkahn/projects/ai-lab/ai-base/models/hf-cache

**Do not use the old double-`flock` recipe.** It was corrected 2026-08-20: two
sequential `flock -c true` invocations each release the lock on exit, so both
succeed on *any* healthy filesystem — it reports "broken" for ext4 and tmpfs
too, and it tests `flock(2)` rather than the POSIX `fcntl` byte-range locks the
constraint is actually about. Use the two-process `probe_locks()` function in
`local_agents/docs/runbooks/os-upgrade-resolute-preflight.md` §8.2, which
exercises both mechanisms under real contention.

Baseline to compare against: on this host **both mechanisms currently work
correctly on the mergerfs path**, identically to ext4 and tmpfs. So the
post-upgrade check is for a *regression* from working, not confirmation of a
suspected break. mergerfs moves from the upstream `~ubuntu-noble` build to the
archive's `2.40.2-5build1` — same version, different packaging — which is
exactly the kind of change worth re-probing.

### B2. GPU stack

Two RTX PRO 6000 Blackwell, driver 595.84, CUDA 13.2. **Note both
`nvidia-driver-590-open` and `nvidia-driver-595-open` are installed** — resolve
that before or during the upgrade rather than letting apt choose.

    nvidia-smi                                   # both GPUs, driver >= 595
    ./.venv/bin/python3 -c "import torch;print(torch.cuda.device_count(), torch.version.cuda)"

Blackwell-specific: nvfp4 is gated on Blackwell support, and the Krea-2 identity
work carries a cuDNN pin. Re-run `test_quant.py` and `test_krea2_identity.py`
specifically, then one live generation at 2144² (the documented cuDNN check).

### B3. Python and the test battery

    ./.venv/bin/python3 --version
    just tests          # 25 suites, expect 0 failures
    just typecheck      # per-root baselines: comfyless=13 nodes=520 pipelines=454

### B4. Toolchain and quality gates

mise pins `just 1.56.0`, `gitleaks 8.30.0`, `osv-scanner 2.4.0`, `node 22.22.2`,
`pyright 1.1.411` — all distro-independent, so these should survive. Verify
rather than assume:

    mise trust ./mise.toml && mise install
    just policy-test && just secrets && just sast

### B5. Services

`Linger=yes` is set, so user units survive logout. `comfyless@.service` and
`scheduler-audit.service` are user units.

    systemctl --user status comfyless@cuda:0
    loginctl show-user gawkahn | grep -i linger

Note `comfyless@.service` currently hardcodes `PYTHONPATH=<repo root>` and
`ExecStart=<repo>/.venv/bin/python3 -m comfyless.generate` — both break if the
venv is rebuilt at a different path. ADR-045 replaces this with a console
script; if that lands first, this unit changes anyway.

### B6. Containers and the gateway

    docker ps
    docker run --rm --gpus all nvidia/cuda:13.0-base nvidia-smi     # runtime intact
    curl -o /dev/null -w '%{http_code}' http://localhost:8100/v1/models   # expect 401
    ss -ltn | grep -E '8100|8101'

The gateway (nginx :8100 → vllm_router :8101) is containerized and insulated
from the host Python change, but the nvidia container runtime is not.

### B7. End-to-end smoke

One real generation through the comfyless CLI, and one through the daemon, and
one enhancer call through the gateway. That exercises the whole stack —
filesystem, GPU, Python, service, network — in a way no unit test does.

---

## Sequencing

1. **Now:** A0 (per-repo Python call) + A1 (venv decoupling) + A2 (holds) +
   A6 (green baseline).
2. **Before the upgrade:** A3 (rewrite apt sources to `resolute`), A4, A5.
   Nothing here can veto the timing any more — mergerfs and the driver both
   verified available.
3. **Upgrade.**
4. **After:** B1 → B7 in order.

The single most valuable item is A1. It is the difference between "the OS
upgrade broke ten projects" and "the OS upgrade changed a Python I no longer
depend on."
