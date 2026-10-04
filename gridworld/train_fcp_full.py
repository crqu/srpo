"""Run the full FCP pipeline: self-play partner pool, then FCP best-response."""
import os
import sys
import time
import multiprocessing as mp

from coop_ppo import train_self_play, train_fcp

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
TOTAL_STEPS = 2_000_000


def _stamp(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_self_play():
    for seed in SEEDS:
        out = f'sp_seed{seed}.pt2'
        if os.path.exists(out):
            _stamp(f"SP seed={seed}: {out} exists, skipping")
            continue
        _stamp(f"SP seed={seed}: training -> {out}")
        train_self_play(seed=seed, total_steps=TOTAL_STEPS)


def run_fcp():
    pool_paths = [f'sp_seed{s}.pt2' for s in SEEDS]
    missing = [p for p in pool_paths if not os.path.exists(p)]
    if missing:
        _stamp(f"ERROR: missing SP checkpoints: {missing}")
        sys.exit(1)

    for seed in SEEDS:
        out = f'fcp_seed{seed}.pt2'
        if os.path.exists(out):
            _stamp(f"FCP seed={seed}: {out} exists, skipping")
            continue
        _stamp(f"FCP seed={seed}: training -> {out}")
        train_fcp(pool_paths, seed=seed, total_steps=TOTAL_STEPS)


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    t0 = time.time()
    _stamp(f"Starting full FCP pipeline | seeds={SEEDS} | steps={TOTAL_STEPS}")

    _stamp("Stage 1: self-play partner pool")
    run_self_play()

    _stamp("Stage 2: FCP best-response")
    run_fcp()

    _stamp(f"Done in {(time.time() - t0) / 3600:.2f} h")
