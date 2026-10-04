us import torch
import numpy as np
import copy
import multiprocessing as mp
import matplotlib.pyplot as plt
from coop_ppo import (
    Agent,
    train_independent_ppo,
    train_adversarial_marl,
    train_pure_adversarial,
    train_self_play,
    train_fcp,
    compute_crossplay_matrix,
    upload_agent,
    upload_rob_agent,
    obs_for_seat,
)


def load_van_pair(seed):
    p1, p2 = Agent(8, 5), Agent(8, 5)
    d = torch.load(f'van_ppo{seed}.pt2', weights_only=True)
    p1.load_state_dict(d['model_state_dict1'])
    p2.load_state_dict(d['model_state_dict2'])
    return p1, p2, d['rewards']


def load_risk_pair(seed):
    p1, p2 = Agent(8, 5), Agent(8, 5)
    d = torch.load(f'risk_ppo{seed}.pt2', weights_only=True)
    p1.load_state_dict(d['model_state_dict1'])
    p2.load_state_dict(d['model_state_dict2'])
    return p1, p2, d['rewards']


def load_pure_adv_pair(seed):
    p1, p2 = Agent(8, 5), Agent(8, 5)
    d = torch.load(f'pure_adv{seed}.pt2', weights_only=True)
    p1.load_state_dict(d['model_state_dict1'])
    p2.load_state_dict(d['model_state_dict2'])
    return p1, p2, d['rewards']


def load_fcp(seed):
    a = Agent(8, 5)
    d = torch.load(f'fcp_seed{seed}.pt2', weights_only=True)
    a.load_state_dict(d['state_dict'])
    return a, d['rewards']


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    TRAIN_VAN  = 0
    TRAIN_RISK = 0
    TRAIN_PURE_ADV = 0
    TRAIN_SP   = 0
    TRAIN_FCP  = 0
    BENCHMARK  = 1

    SEEDS = [1, 2, 3, 4, 5]
    TOTAL_STEPS = 2_000_000
    tau = 10
    risk = 1.0 / tau

    if TRAIN_VAN:
        for seed in SEEDS:
            train_independent_ppo(steps=TOTAL_STEPS, seed=seed)

    if TRAIN_RISK:
        for seed in SEEDS:
            train_adversarial_marl(risk=risk, total_steps=TOTAL_STEPS, seed=seed)

    if TRAIN_PURE_ADV:
        for seed in SEEDS:
            train_pure_adversarial(total_steps=TOTAL_STEPS, seed=seed)

    if TRAIN_SP:
        for seed in SEEDS:
            train_self_play(seed=seed, total_steps=TOTAL_STEPS)

    if TRAIN_FCP:
        pool_paths = [f'sp_seed{s}.pt2' for s in SEEDS]
        for seed in SEEDS:
            train_fcp(pool_paths, seed=seed, total_steps=TOTAL_STEPS)

    if BENCHMARK:
        # --- Reward curves ---
        van = [np.array(load_van_pair(s)[2]) for s in SEEDS]
        risk_r = [np.array(load_risk_pair(s)[2]) for s in SEEDS]
        pure_r = [np.array(load_pure_adv_pair(s)[2]) for s in SEEDS]
        fcp_r = [np.array(load_fcp(s)[1]) for s in SEEDS]

        def plot_curve(arrs, color, label):
            arr = np.array(arrs)
            mean = arr.mean(axis=0)[:, -1]
            std = arr.std(axis=0)[:, -1]
            plt.plot(mean, color=color, label=label)
            plt.fill_between(range(len(mean)), mean - std, mean + std, fc=color, alpha=0.1)

        plt.figure()
        plot_curve(van, 'salmon', 'Vanilla PPO')
        plot_curve(risk_r, 'cornflowerblue', 'Risk-Averse PPO')
        plot_curve(pure_r, 'orange', 'Pure Adversarial')
        plot_curve(fcp_r, 'mediumseagreen', 'FCP')
        plt.legend(); plt.xlabel('training iterations'); plt.ylabel('reward')
        plt.savefig('reward_curves.png', dpi=120)
        print("Saved reward_curves.png")

        # --- Cross-play matrices ---
        van_pairs  = [load_van_pair(s)  for s in SEEDS]
        risk_pairs = [load_risk_pair(s) for s in SEEDS]
        pure_pairs = [load_pure_adv_pair(s) for s in SEEDS]
        fcps       = [load_fcp(s)[0]    for s in SEEDS]

        van_seat0  = [p[0] for p in van_pairs]
        van_seat1  = [p[1] for p in van_pairs]
        risk_seat0 = [p[0] for p in risk_pairs]
        risk_seat1 = [p[1] for p in risk_pairs]
        pure_seat0 = [p[0] for p in pure_pairs]
        pure_seat1 = [p[1] for p in pure_pairs]

        print("Cross-play: Vanilla")
        C_van  = compute_crossplay_matrix(van_seat0,  van_seat1,  num_episodes=20, swap_obs_seat1=False)
        print("Cross-play: Risk-Averse")
        C_risk = compute_crossplay_matrix(risk_seat0, risk_seat1, num_episodes=20, swap_obs_seat1=False)
        print("Cross-play: Pure Adversarial")
        C_pure = compute_crossplay_matrix(pure_seat0, pure_seat1, num_episodes=20, swap_obs_seat1=False)
        print("Cross-play: FCP")
        C_fcp  = compute_crossplay_matrix(fcps,       fcps,       num_episodes=20, swap_obs_seat1=True)

        # --- FCP heatmap ---
        plt.figure(figsize=(5, 4))
        im = plt.imshow(C_fcp, cmap='viridis')
        plt.colorbar(im, label='shared return')
        for i in range(C_fcp.shape[0]):
            for j in range(C_fcp.shape[1]):
                plt.text(j, i, f"{C_fcp[i,j]:.1f}", ha='center', va='center',
                         color='white' if C_fcp[i,j] < C_fcp.mean() else 'black', fontsize=8)
        plt.xticks(range(len(SEEDS)), SEEDS); plt.yticks(range(len(SEEDS)), SEEDS)
        plt.xlabel('FCP seed (seat 1)'); plt.ylabel('FCP seed (seat 0)')
        plt.title('FCP cross-play (shared return)')
        plt.tight_layout()
        plt.savefig('fcp_crossplay.png', dpi=120)
        print("Saved fcp_crossplay.png")

        # --- Headline bar chart: mean off-diagonal ---
        def off_diag(C):
            mask = ~np.eye(C.shape[0], dtype=bool)
            return C[mask]

        bars = [
            ('Vanilla PPO',        off_diag(C_van),  'salmon'),
            ('Risk-Averse PPO',    off_diag(C_risk), 'cornflowerblue'),
            ('Pure Adversarial',   off_diag(C_pure), 'orange'),
            ('FCP',                off_diag(C_fcp),  'mediumseagreen'),
        ]
        plt.figure(figsize=(5, 4))
        means = [b[1].mean() for b in bars]
        stds  = [b[1].std()  for b in bars]
        plt.bar([b[0] for b in bars], means, yerr=stds, color=[b[2] for b in bars], capsize=6)
        plt.ylabel('Mean off-diagonal cross-play return')
        plt.tight_layout()
        plt.savefig('crossplay_comparison.png', dpi=120)
        print("Saved crossplay_comparison.png")

        plt.show()
