import warnings
# Silence "noisy" library warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pettingzoo")
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import os
import signal
import psutil
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import multiprocessing as mp
import time
from collections import deque
import traceback
import gc
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FixedFormatter, MultipleLocator, FixedLocator
import imageio
import matplotlib.animation as animation

# Import PettingZoo MPE
from pettingzoo.mpe import simple_tag_v3

# --- CLEANUP ZOMBIES ---
def kill_zombies():
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'ppid', 'name']):
        try:
            if proc.info['ppid'] == current_pid and proc.info['pid'] != current_pid:
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
kill_zombies()

# ==========================================
# 1) MPE Wrapper
# ==========================================
class MPEWrapper:
    def __init__(self, env_name="simple_tag"):
        self.env = simple_tag_v3.parallel_env(
            num_good=1, 
            num_adversaries=2, 
            num_obstacles=2, 
            max_cycles=100, 
            continuous_actions=False
        )
        self.env.reset()
        self.agents = self.env.agents
        self.num_players = len(self.agents)
        
        self.max_obs_dim = 0
        self.max_action_dim = 0
        for agent in self.agents:
            obs_shape = self.env.observation_space(agent).shape[0]
            act_shape = self.env.action_space(agent).n
            if obs_shape > self.max_obs_dim: self.max_obs_dim = obs_shape
            if act_shape > self.max_action_dim: self.max_action_dim = act_shape
        self.active_mask_buf = np.ones(self.num_players, dtype=np.float32)

    def get_input_dim(self): return int(self.max_obs_dim + self.max_action_dim)

    def _pad_obs(self, obs_dict):
        obs_stack = []
        for agent in self.agents:
            raw = obs_dict[agent]
            if len(raw) < self.max_obs_dim:
                pad = np.zeros(self.max_obs_dim - len(raw), dtype=np.float32)
                raw = np.concatenate([raw, pad])
            mask = np.zeros(self.max_action_dim, dtype=np.float32)
            mask[:self.env.action_space(agent).n] = 1.0
            obs_stack.append(np.concatenate([raw, mask]).astype(np.float32))
        return np.stack(obs_stack)

    def reset(self):
        o, _ = self.env.reset()
        return self._pad_obs(o), self.active_mask_buf.copy()

    def step(self, actions):
        act_d = {a: int(actions[i]) for i, a in enumerate(self.agents)}
        no, rew, term, trunc, _ = self.env.step(act_d)
        done = any(term.values()) or any(trunc.values())
        r = np.array([rew[a] for a in self.agents], dtype=np.float32) * 0.1
        return self._pad_obs(no), r, done, self.active_mask_buf.copy(), {}

    # --- ADD THIS METHOD ---
    def close(self):
        self.env.close()
# ==========================================
# 2) Parallel Env
# ==========================================
def vec_worker(remote, parent_remote, env_fn_wrapper, num_envs):
    os.environ["OMP_NUM_THREADS"] = "1"
    parent_remote.close()
    try:
        envs = [env_fn_wrapper.x() for _ in range(num_envs)]
        while True:
            try:
                cmd, data = remote.recv()
            except EOFError:
                break  # Main process closed the pipe, exit gracefully

            if cmd == "step":
                results = [env.step(d) for env, d in zip(envs, data)]
                obs, rew, done, active, _ = zip(*results)
                obs, active = list(obs), list(active)
                for i, d in enumerate(done):
                    if d: obs[i], active[i] = envs[i].reset()
                remote.send((np.stack(obs), np.stack(rew), np.stack(done), np.stack(active)))
            elif cmd == "reset":
                results = [env.reset() for env in envs]
                obs, active = zip(*results)
                remote.send((np.stack(obs), np.stack(active)))
            elif cmd == "close":
                remote.close()
                break
    except Exception as e:
        print(f"VecWorker Error: {e}")
        traceback.print_exc()
        raise e
    finally:
        # Ensure environments are closed on exit
        for env in envs:
            env.close()

class CloudpickleWrapper:
    def __init__(self, x): self.x = x
    def __getstate__(self): import cloudpickle; return cloudpickle.dumps(self.x)
    def __setstate__(self, ob): import pickle; self.x = pickle.loads(ob)

GLOBAL_ENV_NAME = "simple_tag"
def make_env(): return MPEWrapper(GLOBAL_ENV_NAME)

class ParallelEnv:
    def __init__(self, total_envs, num_workers=2, env_name="simple_tag"):
        global GLOBAL_ENV_NAME; GLOBAL_ENV_NAME = env_name
        self.num_workers = num_workers
        self.envs_per_worker = total_envs // num_workers
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_workers)])
        self.ps = [mp.Process(target=vec_worker, args=(wr, r, CloudpickleWrapper(make_env), self.envs_per_worker))
                   for wr, r in zip(self.work_remotes, self.remotes)]
        for p in self.ps: p.daemon = True; p.start()
        for remote in self.work_remotes: remote.close()

    def step(self, actions):
        chunks = np.array_split(actions, self.num_workers, axis=0)
        for remote, chunk in zip(self.remotes, chunks): remote.send(("step", chunk))
        results = [remote.recv() for remote in self.remotes]
        return [np.concatenate(x) for x in zip(*results)]

    def reset(self):
        for remote in self.remotes: remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        return [np.concatenate(x) for x in zip(*results)]

    def close(self):
        for remote in self.remotes: remote.send(("close", None))
        for p in self.ps: p.join(timeout=1.0)

# ==========================================
# 3) Agent & Optimizer
# ==========================================
class MPEAgent(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = input_dim - action_dim
        self.net = nn.Sequential(nn.Linear(self.obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, 0.01)

    def forward_heads(self, feat, mask):
        logits = self.actor(feat)
        logits = logits.masked_fill(mask < 0.5, -1e9)
        return logits, self.critic(feat).squeeze(-1)

    def get_action(self, x, hidden=None):
        obs, mask = x[:, :self.obs_dim], x[:, self.obs_dim:]
        feat = self.net(obs)
        logits, val = self.forward_heads(feat, mask)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze(-1)
        lp = torch.gather(torch.log(probs + 1e-8), -1, action.unsqueeze(-1)).squeeze(-1)
        return action, lp, val, None

@torch.jit.script
def compute_gae(rew, val, next_val, done, gamma: float, lam: float):
    adv = torch.zeros_like(rew)
    last_gae = torch.zeros_like(next_val)
    for t in range(rew.size(0)-1, -1, -1):
        non_term = 1.0 - done[t]
        nv = next_val if t == rew.size(0)-1 else val[t+1]
        delta = rew[t] + gamma * nv * non_term - val[t]
        last_gae = delta + gamma * lam * non_term * last_gae
        adv[t] = last_gae
    return adv, adv + val

def ppo_update(agent, opt, obs, act, lp, adv, ret, mask, ref_probs=None, risk=0.0, k_epochs=4, clip=0.2, ent_coef=0.01):
    full_dim = obs.shape[-1]
    b_obs = obs.reshape(-1, full_dim)
    b_act, b_lp, b_adv, b_ret = act.reshape(-1), lp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
    b_mask = mask.reshape(-1)
    b_ref = ref_probs.reshape(-1, ref_probs.shape[-1]) if ref_probs is not None else None

    for _ in range(k_epochs):
        feat = agent.net(b_obs[:, :agent.obs_dim])
        obs_mask = b_obs[:, agent.obs_dim:].reshape(-1, agent.action_dim)
        logits, vals = agent.forward_heads(feat, obs_mask)
        
        probs = F.softmax(logits, dim=-1)
        curr_lp = torch.log(torch.gather(probs, -1, b_act.unsqueeze(-1)) + 1e-8).squeeze()
        ratio = torch.exp(curr_lp - b_lp)
        
        surr1 = ratio * b_adv
        surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * b_adv
        pol_loss = -torch.min(surr1, surr2) * b_mask
        
        kl_loss = 0.0
        if b_ref is not None:
            kl_div = (b_ref * (torch.log(b_ref + 1e-8) - F.log_softmax(logits, dim=-1))).sum(-1)
            kl_loss = risk * kl_div * b_mask

        kl_term = kl_loss.mean() if isinstance(kl_loss, torch.Tensor) else kl_loss
        loss = pol_loss.mean() + 0.5 * ((vals - b_ret)**2).mean() - ent_coef * (-(probs*torch.log(probs+1e-8)).sum(-1)).mean() + kl_term
        
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), 10.0); opt.step()

# ==========================================
# 4) Standard PPO (3 Independent Agents) - 1 Runner
# ==========================================
def train_PPO_MPE(env_name, agents=None, steps=2_000_000, risk_factor=0.0, rewards=None, entropy_coef=0.01, lr=6e-4):
    print(f"=== Training {env_name} [Standard Independent] ===")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    num_envs, num_steps = 64, 128
    gamma, gae_lambda = 0.99, 0.95
    
    env = ParallelEnv(num_envs, 2, env_name)
    eval_env = ParallelEnv(num_envs, 2, env_name)
    dummy = make_env(); input_dim = dummy.get_input_dim(); max_act = dummy.max_action_dim; del dummy

    if agents is None:
        c1 = MPEAgent(input_dim, max_act).to(device)
        c2 = MPEAgent(input_dim, max_act).to(device)
        runner = MPEAgent(input_dim, max_act).to(device)
    else:
        c1, c2, runner = agents[0].to(device), agents[1].to(device), agents[2].to(device)

    opt_c1 = optim.Adam(c1.parameters(), lr=lr)
    opt_c2 = optim.Adam(c2.parameters(), lr=lr)
    #opt_r = optim.Adam(runner.parameters(), lr=lr/2)


    t_obs = torch.zeros((num_steps, num_envs, 3, input_dim), device=device)
    t_act = torch.zeros((num_steps, num_envs, 3), dtype=torch.long, device=device)
    t_lp = torch.zeros((num_steps, num_envs, 3), device=device)
    t_rew = torch.zeros((num_steps, num_envs, 3), device=device)
    t_val = torch.zeros((num_steps, num_envs, 3), device=device)
    t_done = torch.zeros((num_steps, num_envs), device=device)
    t_active = torch.zeros((num_steps, num_envs, 3), device=device)

    start_time = time.time()
    try:
        for update in range(1, steps // (num_envs * num_steps) + 1):

            obs, active = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
            active = torch.tensor(active, dtype=torch.float32, device=device)

            for t in range(num_steps):
                with torch.no_grad():
                    a1, lp1, v1, _ = c1.get_action(obs[:, 0])
                    a2, lp2, v2, _ = c2.get_action(obs[:, 1])
                    ar, lpr, vr, _ = runner.get_action(obs[:, 2])
                
                act_np = np.stack([a1.cpu().numpy(), a2.cpu().numpy(), ar.cpu().numpy()], axis=1)
                no, rew, d, n_act = env.step(act_np)
                
                t_obs[t] = obs; t_active[t] = active
                t_act[t,:,0]=a1; t_act[t,:,1]=a2; t_act[t,:,2]=ar
                t_lp[t,:,0]=lp1; t_lp[t,:,1]=lp2; t_lp[t,:,2]=lpr
                t_val[t,:,0]=v1; t_val[t,:,1]=v2; t_val[t,:,2]=vr
                t_rew[t] = torch.tensor(rew, device=device); t_done[t] = torch.tensor(d, device=device)
                
                obs = torch.tensor(no, dtype=torch.float32, device=device)
                active = torch.tensor(n_act, dtype=torch.float32, device=device)

            def do_update(agent, opt, p_idx):
                with torch.no_grad():
                    _, _, nv, _ = agent.get_action(obs[:, p_idx])
                    adv, ret = compute_gae(t_rew[:,:,p_idx], t_val[:,:,p_idx], nv, t_done, gamma, gae_lambda)
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ppo_update(agent, opt, t_obs[:,:,p_idx], t_act[:,:,p_idx], t_lp[:,:,p_idx], 
                           adv, ret, t_active[:,:,p_idx], None, 0.0, 4, 0.2, entropy_coef)

            do_update(c1, opt_c1, 0)
            do_update(c2, opt_c2, 1)
            #do_update(runner, opt_r, 2)

            if update % 5 == 0:
                score = evaluate_team(eval_env, c1, c2, runner, 100, device)
                print(f"Upd {update} | True Team Score: {score:.2f} | Time: {time.time()-start_time:.0f}s")
                rewards.append(score)
    finally: env.close()
    return [c1, c2, runner], rewards

# ==========================================
# 5) Train Adversarial (Split Data, 2 Independent Runners)
# ==========================================
def train_adversarial(env_name, agents=None, steps=2_000_000, risk_factor=0.1, lr=6e-4, rewards=[], entropy_coef=0.05):
    print(f"=== Training Adversarial KL-MPE (Split Runners, Risk={risk_factor}) ===")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    num_envs, num_steps = 64, 128
    half = num_envs // 2
    gamma, gae_lambda = 0.99, 0.95
    
    env = ParallelEnv(num_envs, 2, env_name)
    eval_env = ParallelEnv(num_envs, 2, env_name)
    dummy = make_env(); input_dim = dummy.get_input_dim(); max_act = dummy.max_action_dim; del dummy

    if agents is None:
        c1 = MPEAgent(input_dim, max_act).to(device)
        c2 = MPEAgent(input_dim, max_act).to(device)
        r1 = MPEAgent(input_dim, max_act).to(device)
    else:
        c1, c2, r1 = agents[0].to(device), agents[1].to(device), agents[2].to(device)
    
    # Second Runner (Clone R1 initially)
    #r2 = MPEAgent(input_dim, max_act).to(device)
    #r2.load_state_dict(r1.state_dict())
    
    adv1 = MPEAgent(input_dim, max_act).to(device); adv1.load_state_dict(c1.state_dict())
    adv2 = MPEAgent(input_dim, max_act).to(device); adv2.load_state_dict(c2.state_dict())

    opt_c1 = optim.Adam(c1.parameters(), lr=lr)
    opt_c2 = optim.Adam(c2.parameters(), lr=lr)
    #opt_r1 = optim.Adam(r1.parameters(), lr=lr/2)
    #opt_r2 = optim.Adam(r2.parameters(), lr=lr/2)
    opt_a1 = optim.Adam(adv1.parameters(), lr=lr)
    opt_a2 = optim.Adam(adv2.parameters(), lr=lr)


    t_obs = torch.zeros((num_steps, num_envs, 3, input_dim), device=device)
    t_act = torch.zeros((num_steps, num_envs, 3), dtype=torch.long, device=device)
    t_lp = torch.zeros((num_steps, num_envs, 3), device=device)
    t_rew = torch.zeros((num_steps, num_envs, 3), device=device)
    t_val = torch.zeros((num_steps, num_envs, 3), device=device)
    t_done = torch.zeros((num_steps, num_envs), device=device)
    t_active = torch.zeros((num_steps, num_envs, 3), device=device)

    start_time = time.time()
    try:
        for update in range(1, steps // (num_envs * num_steps) + 1):
            obs, active = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
            active = torch.tensor(active, dtype=torch.float32, device=device)
            for t in range(num_steps):
                with torch.no_grad():
                    # Batch A (0:32): C1 (True) vs Adv2 (Adv) vs R1
                    c1_a, lp1_a, v1_a, _ = c1.get_action(obs[:half, 0])
                    c2_a, lp2_a, v2_a, _ = adv2.get_action(obs[:half, 1])
                    rn_a, lpr_a, vr_a, _ = r1.get_action(obs[:half, 2])
                    
                    # Batch B (32:64): Adv1 (Adv) vs C2 (True) vs R2
                    c1_b, lp1_b, v1_b, _ = adv1.get_action(obs[half:, 0])
                    c2_b, lp2_b, v2_b, _ = c2.get_action(obs[half:, 1])
                    rn_b, lpr_b, vr_b, _ = r1.get_action(obs[half:, 2])
                
                act1 = torch.cat([c1_a, c1_b])
                act2 = torch.cat([c2_a, c2_b])
                act3 = torch.cat([rn_a, rn_b])
                act_np = np.stack([act1.cpu().numpy(), act2.cpu().numpy(), act3.cpu().numpy()], axis=1)
                
                no, rew, d, n_act = env.step(act_np)
                
                t_obs[t] = obs; t_active[t] = active
                t_act[t,:,0]=act1; t_act[t,:,1]=act2; t_act[t,:,2]=act3
                
                t_lp[t,:half,0]=lp1_a; t_lp[t,half:,0]=lp1_b
                t_lp[t,:half,1]=lp2_a; t_lp[t,half:,1]=lp2_b
                t_lp[t,:half,2]=lpr_a; t_lp[t,half:,2]=lpr_b
                
                t_val[t,:half,0]=v1_a; t_val[t,half:,0]=v1_b
                t_val[t,:half,1]=v2_a; t_val[t,half:,1]=v2_b
                t_val[t,:half,2]=vr_a; t_val[t,half:,2]=vr_b
                
                t_rew[t] = torch.tensor(rew, device=device); t_done[t] = torch.tensor(d, device=device)
                obs = torch.tensor(no, dtype=torch.float32, device=device)
                active = torch.tensor(n_act, dtype=torch.float32, device=device)

            def do_update(agent, opt, p_idx, subset_slice, ref=None, inv=False, epochs=4):
                with torch.no_grad():
                    _, _, nv, _ = agent.get_action(obs[subset_slice, p_idx])
                    r_use = -t_rew[:,subset_slice,p_idx] if inv else t_rew[:,subset_slice,p_idx]
                    adv, ret = compute_gae(r_use, t_val[:,subset_slice,p_idx], nv, t_done[:,subset_slice], gamma, gae_lambda)
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    
                    ref_p = None
                    if ref:
                        sub_obs = t_obs[:,subset_slice,p_idx].reshape(-1, input_dim)
                        mask = sub_obs[:, ref.obs_dim:].reshape(-1, ref.action_dim)
                        l, _ = ref.forward_heads(ref.net(sub_obs[:, :ref.obs_dim]), mask)
                        ref_p = F.softmax(l, -1).view(num_steps, -1, ref.action_dim)
                
                ppo_update(agent, opt, t_obs[:,subset_slice,p_idx], t_act[:,subset_slice,p_idx], t_lp[:,subset_slice,p_idx], 
                           adv, ret, t_active[:,subset_slice,p_idx], ref_p, risk, epochs, 0.2, entropy_coef)

            # C1, C2, Adv Updates
            do_update(c1, opt_c1, 0, slice(0, half))
            do_update(c2, opt_c2, 1, slice(half, num_envs))
            do_update(adv2, opt_a2, 1, slice(0, half), ref=c2, inv=True)
            do_update(adv1, opt_a1, 0, slice(half, num_envs), ref=c1, inv=True)
            
            # Runners
            #do_update(r1, opt_r1, 2, slice(0, half))
            #do_update(r2, opt_r2, 2, slice(half, num_envs))

            if update % 5 == 0:
                score = evaluate_team(eval_env, c1, c2, r1, 100, device)
                print(f"Upd {update} | True Team Score: {score:.2f} | Time: {time.time()-start_time:.0f}s")
                rewards.append(score)
    finally: env.close()
    return [c1, c2, r1, adv1, adv2], rewards

def evaluate_team(env, c1, c2, r, max_steps=100, dev='cpu'):
    obs_list = env.reset()
    obs = torch.tensor(obs_list[0], dtype=torch.float32, device=dev)
    total_rewards = np.zeros(obs.shape[0], dtype=np.float32)
    for _ in range(max_steps):
        with torch.no_grad():
            a1, _, _, _ = c1.get_action(obs[:, 0])
            a2, _, _, _ = c2.get_action(obs[:, 1])
            a3, _, _, _ = r.get_action(obs[:, 2])
        actions_np = np.stack([a1.cpu().numpy(), a2.cpu().numpy(), a3.cpu().numpy()], axis=1)
        results = env.step(actions_np)
        next_obs, rew, done, _ = results
        total_rewards += rew[:, 0]
        obs = torch.tensor(next_obs, dtype=torch.float32, device=dev)
    return total_rewards.mean()

def visualize_policy(env_name, agents, filename=None):
    print(f"--- Visualizing {env_name}... ---")
    from pettingzoo.mpe import simple_tag_v3
    env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=2, num_obstacles=2, render_mode="rgb_array", max_cycles=100, continuous_actions=False)
    
    obs_dict, _ = env.reset()
    frames = []
    
    max_obs = 0; max_act = 0
    for agent in env.agents:
        max_obs = max(max_obs, env.observation_space(agent).shape[0])
        max_act = max(max_act, env.action_space(agent).n)
        
    def pad(raw):
        if len(raw) < max_obs:
            p = np.zeros(max_obs - len(raw), dtype=np.float32)
            ret = np.concatenate([raw, p])
        else: ret = raw
        mask = np.zeros(max_act, dtype=np.float32); mask[:5] = 1.0
        return np.concatenate([ret, mask]).astype(np.float32)

    c1, c2, r = agents[0], agents[1], agents[2]
    device = next(c1.parameters()).device
    
    for _ in range(100):
        frames.append(env.render())
        o0 = torch.tensor(pad(obs_dict['adversary_0']), device=device).unsqueeze(0)
        o1 = torch.tensor(pad(obs_dict['adversary_1']), device=device).unsqueeze(0)
        o2 = torch.tensor(pad(obs_dict['agent_0']), device=device).unsqueeze(0)
        
        with torch.no_grad():
            a0,_,_,_ = c1.get_action(o0)
            a1,_,_,_ = c2.get_action(o1)
            a2,_,_,_ = r.get_action(o2)
            
        actions = {'adversary_0': a0.item(), 'adversary_1': a1.item(), 'agent_0': a2.item()}
        obs_dict, _, terms, truncs, _ = env.step(actions)
        if any(terms.values()) or any(truncs.values()): break
    
    env.close()
    
    # --- MODIFIED BLOCK START ---
    if filename is not None:
        fig, ax = plt.subplots()
        plt.axis('off')
        # Create a list of [Artist] objects for each frame
        ims = [[ax.imshow(frame, animated=True)] for frame in frames]
        # Create the animation
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        # Save the animation (requires 'pillow' for GIFs, which is standard)
        ani.save(filename, writer='pillow', fps=20)
        print(f"Saved video to {filename}")
        plt.close(fig)
    else:
        # Original interactive playback
        fig, ax = plt.subplots()
        plt.axis('off')
        img = ax.imshow(frames[0])
        for frame in frames:
            img.set_data(frame)
            plt.pause(0.05)
            plt.draw()
        plt.close(fig)

def upload_agent(chaser1,chaser2, runner, filename):
    saved = torch.load(filename, weights_only=False)
    chaser1.load_state_dict(saved.get('chaser1'))
    chaser2.load_state_dict(saved.get('chaser2'))
    runner.load_state_dict(saved.get('runner'))
    rewards=saved.get('rewards')
    return [chaser1,chaser2, runner, rewards]


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError: pass


    num=30

    ENV_NAME = "simple_tag"
    risk = 0.1
    train = 0
    benchmark=1
    entropy=0.01
    lr=1e-3
    
    ENV = "simple_tag"
    dev = torch.device("cpu")
    dum = make_env(); d_in = dum.get_input_dim(); d_act = dum.max_action_dim; del dum
    
    device = torch.device("cpu")
    
    if train:


        c1 = MPEAgent(d_in, d_act).to(dev)
        c2 = MPEAgent(d_in, d_act).to(dev)
        r = MPEAgent(d_in, d_act).to(dev)

        upload_agent(c1,  c2, r, "./training_agents/ippo_tag0001_3.pt2") 
        c1 = MPEAgent(d_in, d_act).to(dev)
        c2 = MPEAgent(d_in, d_act).to(dev)
        
        agents = [c1,c2,r]
        all_rewards = []

        agents, rews = train_PPO_MPE(ENV_NAME, agents=agents, steps=30_000_000, rewards=all_rewards,entropy_coef=entropy, lr=lr)
        torch.save({'type': 'Vanilla', 'chaser1': agents[0].state_dict(), 'chaser2': agents[1].state_dict(),'runner': agents[2].state_dict(), 'rewards': rews,'entropy':entropy}, 'vrun'+str(num)+'.pt2')

        c1r = MPEAgent(d_in, d_act).to(dev)
        c2r = MPEAgent(d_in, d_act).to(dev)
        rr = MPEAgent(d_in, d_act).to(dev)
        upload_agent(c1r, c2r, rr, "./training_agents/ippo_tag0001_3.pt2") 
        c1r = MPEAgent(d_in, d_act).to(dev)
        c2r = MPEAgent(d_in, d_act).to(dev)
        agents2 = [c1r,c2r,rr]
        all_rewards2 = []

        agents2, rews = train_adversarial(ENV_NAME, agents=agents2, steps=30_000_000, risk_factor=risk, rewards=all_rewards2,entropy_coef=entropy, lr=lr)
        torch.save({'type': 'Robust', 'chaser1': agents2[0].state_dict(), 'chaser2': agents2[1].state_dict(), 'runner': agents2[2].state_dict(),'adv1': agents2[3].state_dict(), 'adv2': agents2[4].state_dict(),'rewards': rews,'risk':risk,'entropy':entropy}, 'rrunner'+str(num)+'.pt2')
   

    if benchmark:

        print("Plotting Reward Traces")
        c1 = MPEAgent(d_in, d_act).to(dev)
        c2 = MPEAgent(d_in, d_act).to(dev)
        r = MPEAgent(d_in, d_act).to(dev)

        all_rewards = []

        risk_fns=['rrunner1.pt2','rrunner2.pt2','rrunner3.pt2','rrunner4.pt2','rrunner5.pt2','rrunner6.pt2','rrunner7.pt2','rrunner8.pt2','rrunner9.pt2','rrunner10.pt2','rrunner11.pt2','rrunner12.pt2','rrunner13.pt2','rrunner14.pt2','rrunner15.pt2','rrunner16.pt2','rrunner17.pt2','rrunner18.pt2','rrunner19.pt2','rrunner20.pt2','rrunner21.pt2','rrunner22.pt2','rrunner23.pt2','rrunner24.pt2','rrunner25.pt2','rrunner26.pt2','rrunner27.pt2','rrunner28.pt2','rrunner29.pt2','rrunner30.pt2']
        #np.random.shuffle(risk_fns)
        van_fns=['vrun1.pt2','vrun2.pt2','vrun3.pt2','vrun4.pt2','vrun5.pt2','vrun6.pt2','vrun7.pt2','vrun8.pt2','vrun9.pt2','vrun10.pt2','vrun11.pt2','vrun12.pt2','vrun13.pt2','vrun14.pt2','vrun15.pt2','vrun16.pt2','vrun17.pt2','vrun18.pt2','vrun19.pt2','vrun20.pt2','vrun21.pt2','vrun22.pt2','vrun23.pt2','vrun24.pt2','vrun25.pt2','vrun26.pt2','vrun27.pt2','vrun28.pt2','vrun29.pt2','vrun30.pt2']
 
        fnames=van_fns+risk_fns

        rs_rob=[np.array(upload_agent(c1,c2,r,'./agents/'+filename)[-1]) for filename in risk_fns]
        rs_van=[np.array(upload_agent(c1,c2,r,'./agents/'+filename)[-1])for filename in van_fns]

       
        if len(van_fns)>0:
            rs2=np.mean(np.array(rs_van),axis=0)
            st2=np.std(np.array(rs_van),axis=0)
            plt.plot(np.cumsum(5*np.ones(len(rs2))),rs2,'lightcoral', label="Vanilla PPO")
            plt.fill_between(np.cumsum(5*np.ones(len(rs2))),rs2-st2,rs2+st2,fc='lightcoral',alpha=0.2)

        if len(risk_fns)>0:
            rs1=np.mean(np.array(rs_rob),axis=0)
            st1=np.std(np.array(rs_rob),axis=0)
            plt.plot(np.cumsum(5*np.ones(len(rs1))),rs1,'royalblue',label='Risk-Averse PPO')
            plt.fill_between(np.cumsum(5*np.ones(len(rs1))),rs1-st1,rs1+st1,fc='royalblue',alpha=0.2)

        plt.legend()
        plt.xlabel('training steps')
        plt.ylabel('reward')
        plt.pause(0.1)
        

        c12 = copy.deepcopy(c1).to(device)
        r2=copy.deepcopy(r).to(device)
        c22 = copy.deepcopy(c2).to(device)
        comp1=np.zeros([len(fnames),len(fnames)])
        
        print("Running Round-Robin")
        i=0
        eval_env= ParallelEnv(100, 15, ENV_NAME)
        c12,c22,r2,_=upload_agent(c12,c22,r2,"./training_agents/ippo_tag0001_3.pt2")
        runner=copy.deepcopy(r2)
        for f1 in fnames:
            print(str(i+1)+"/"+str(len(fnames))+' rows')
            
            c1,c2,r,_=upload_agent(c1,c2,r,'./agents/'+f1)
            j=0
            for f2 in fnames:
                c12,c22,r2,_=upload_agent(c12,c22,r2,'./agents/'+f2)

                payoff=evaluate_team(eval_env, c1,c22,runner,max_steps=100)
                comp1[i,j]=payoff
                j+=1
            i+=1

        plt.matshow(comp1,cmap='inferno'); 
        plt.colorbar(); 

        plt.gca().set_xticks([-0.5,len(van_fns)-0.5,len(fnames)-0.5], labels=[],fontsize=8); 
        plt.gca().set_yticks([-0.5,len(van_fns)-0.5,len(fnames)-0.5], labels=[],fontsize=8); 
        plt.gca().tick_params(length=5, width=1, which='major',color='k',bottom=False)
        plt.gca().minorticks_on()
        plt.gca().tick_params(length=0, width=1, which='minor',color='k',bottom=False,labelsize=8)
        import math
        mp1=(-0.5+len(van_fns)-0.5)/2
        mp2=(len(van_fns)-0.5 + len(fnames)-0.5)/2+0.5
        plt.gca().xaxis.set_minor_locator(FixedLocator([mp1,mp2]))
        plt.gca().xaxis.set_minor_formatter(FixedFormatter(['PPO\n Agent 1','RQE \n Agent 1']))
        plt.gca().yaxis.set_minor_locator(FixedLocator([mp1,mp2]))
        plt.gca().yaxis.set_minor_formatter(FixedFormatter(['PPO\n Agent 2','RQE\n Agent 2']))
        for t in plt.gca().xaxis.get_minorticklabels():
            t.set_ha("center")              # horizontal alignment of the text box
            t.set_multialignment("center")

        for t in plt.gca().yaxis.get_minorticklabels():
            t.set_ha("right")              # horizontal alignment of the text box
            t.set_multialignment("center")

        plt.show()
