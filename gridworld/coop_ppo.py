import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import copy
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
from numba import njit

# ==========================================
# 1. JIT Compiled Environment Logic (Numba)
# ==========================================
@njit
def fast_step_logic(agent_pos, has_onion, onion_locs, onions_available, pot_loc, actions, grid_size=5):
    """
    Compiled version of the environment logic for high-speed execution.
    Strictly preserves the logic: Move=0.2 cost, Collision=2.0 cost, Pot is solid.
    """
    rewards = np.zeros(2, dtype=np.float32)
    # 0: Up, 1: Down, 2: Left, 3: Right, 4: Stay
    moves = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=np.int32)
    
    move_penalty = 0.2
    collision_penalty = 2.0 
    
    # Random permutation for agent order (0 or 1 first)
    p0, p1 = 0, 1
    if np.random.random() < 0.5:
        p0, p1 = 1, 0
    order = np.array([p0, p1])
    
    for i in order:
        action = actions[i]
        
        # 1. Apply Move Penalty
        if action != 4: 
            rewards[i] -= move_penalty

        # 2. Calculate Proposed Position
        move = moves[action]
        curr_r, curr_c = agent_pos[i][0], agent_pos[i][1]
        
        # Manual clip to ensure boundaries
        prop_r = curr_r + move[0]
        if prop_r < 0: prop_r = 0
        elif prop_r >= grid_size: prop_r = grid_size - 1
        
        prop_c = curr_c + move[1]
        if prop_c < 0: prop_c = 0
        elif prop_c >= grid_size: prop_c = grid_size - 1
        
        # 3. Check Interactions
        
        # A. Check Pot Collision (Bump to Deliver)
        # Pot is solid, so if we hit it, we stay put (curr_r, curr_c)
        if prop_r == pot_loc[0] and prop_c == pot_loc[1]:
            if has_onion[i] == 1:
                has_onion[i] = 0
                rewards[0] += 10.0
                rewards[1] += 10.0
            continue 

        # B. Check Agent Collision
        other = 1 - i
        if prop_r == agent_pos[other][0] and prop_c == agent_pos[other][1]:
            rewards[i] -= collision_penalty
            continue 
        
        # C. Update Position
        agent_pos[i][0] = prop_r
        agent_pos[i][1] = prop_c
        
        # 4. Check Pickup (Stepping ONTO onion)
        if has_onion[i] == 0:
            # Check onion loc 0
            if onions_available[0] == 1 and agent_pos[i][0] == onion_locs[0][0] and agent_pos[i][1] == onion_locs[0][1]:
                has_onion[i] = 1
                onions_available[0] = 0
                rewards[0] += 1.0 
                rewards[1] += 1.0 
            # Check onion loc 1
            elif onions_available[1] == 1 and agent_pos[i][0] == onion_locs[1][0] and agent_pos[i][1] == onion_locs[1][1]:
                has_onion[i] = 1
                onions_available[1] = 0
                rewards[0] += 1.0 
                rewards[1] += 1.0 
            
    # 5. Independent Respawn (0.2 prob)
    respawn_prob = 0.2
    if onions_available[0] == 0 and np.random.random() < respawn_prob:
        onions_available[0] = 1
    if onions_available[1] == 0 and np.random.random() < respawn_prob:
        onions_available[1] = 1
            
    return agent_pos, has_onion, onions_available, rewards

@njit
def fast_get_obs(agent_pos, has_onion, onions_available):
    # Flatten and concatenate manually for Numba speed
    obs = np.empty(8, dtype=np.float32)
    obs[0] = agent_pos[0][0]
    obs[1] = agent_pos[0][1]
    obs[2] = has_onion[0]
    obs[3] = agent_pos[1][0]
    obs[4] = agent_pos[1][1]
    obs[5] = has_onion[1]
    obs[6] = onions_available[0]
    obs[7] = onions_available[1]
    return obs

class CoopCookingEnv(gym.Env):
    def __init__(self):
        self.grid_size = 5
        self.agents = [0, 1]
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=5, shape=(8,), dtype=np.float32)
        self.max_steps = 50
        self.reset()

    def reset(self):
        # User specified random start
        if np.random.random() < 0.5:
            self.agent_pos = np.array([[1, 0], [0 ,1]], dtype=np.int32)
        else:
            self.agent_pos = np.array([[0, 1], [1 ,0]], dtype=np.int32)
            
        self.has_onion = np.array([0, 0], dtype=np.int32)
        self.onion_locs = np.array([[3, 4], [4, 3]], dtype=np.int32)
        self.onions_available = np.array([1, 1], dtype=np.int32)
        self.pot_loc = np.array([2, 2], dtype=np.int32)
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return fast_get_obs(self.agent_pos, self.has_onion, self.onions_available)

    def step(self, actions):
        self.agent_pos, self.has_onion, self.onions_available, rewards = fast_step_logic(
            self.agent_pos, 
            self.has_onion, 
            self.onion_locs, 
            self.onions_available, 
            self.pot_loc, 
            np.array(actions, dtype=np.int32)
        )
        
        self.steps += 1
        done = self.steps >= self.max_steps
        return self._get_obs(), rewards, done, {}

# ==========================================
# 2. Parallel Vector Env
# ==========================================
def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done: ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break

class CloudpickleWrapper(object):
    def __init__(self, x): self.x = x
    def __getstate__(self): import cloudpickle; return cloudpickle.dumps(self.x)
    def __setstate__(self, ob): import pickle; self.x = pickle.loads(ob)

class ParallelVectorEnv:
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True 
            p.start()
        for remote in self.work_remotes: remote.close()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return (np.stack(obs), np.stack(rews), np.stack(dones))

    def reset(self):
        for remote in self.remotes: remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for remote in self.remotes: remote.send(('close', None))
        for p in self.ps: p.join()

# ==========================================
# 3. PPO Agent
# ==========================================
class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None: action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), probs

def obs_for_seat(obs, fcp_seat):
    """
    Return obs reordered so FCP's seat info is in slots [0,1,2] and partner's in [3,4,5].
    obs:       shape (num_envs, 8) — [a0_r, a0_c, a0_has, a1_r, a1_c, a1_has, onion0, onion1]
    fcp_seat:  shape (num_envs,) int array of {0, 1}; which seat FCP plays in each env.
    Onion slots [6,7] are unchanged. Vectorised; works on numpy arrays or torch tensors.
    """
    if isinstance(obs, torch.Tensor):
        out = obs.clone()
        mask = torch.as_tensor(fcp_seat, dtype=torch.bool, device=obs.device)
        if mask.any():
            swapped = torch.cat([out[mask, 3:6], out[mask, 0:3], out[mask, 6:8]], dim=1)
            out[mask] = swapped
        return out
    else:
        out = obs.copy()
        mask = np.asarray(fcp_seat).astype(bool)
        if mask.any():
            tmp = out[mask, 0:3].copy()
            out[mask, 0:3] = out[mask, 3:6]
            out[mask, 3:6] = tmp
        return out

# ==========================================
# 4. Visualization & Utilities
# ==========================================
ITEM_MAP = {
    0: {'label': 'Empty', 'color': 'white'},
    1: {'label': 'Counter', 'color': 'gray'},
    2: {'label': 'Onion', 'color': 'gold'},
    3: {'label': 'Tomato', 'color': 'tomato'},
    4: {'label': 'Pot', 'color': 'black'},
}

def render_grid(env, ax, step_num, total_reward):
    base_env = getattr(env, 'unwrapped', env)
    grid = np.zeros((base_env.grid_size, base_env.grid_size), dtype=int)
    py, px = base_env.pot_loc
    grid[py, px] = 4
    for i, (oy, ox) in enumerate(base_env.onion_locs):
        if base_env.onions_available[i]:
            grid[oy, ox] = 2

    rows, cols = grid.shape
    ax.clear()
    ax.set_title(f"Step: {step_num} | Total Reward: {total_reward:.2f}")
    
    for r in range(rows):
        for c in range(cols):
            item_id = grid[r, c]
            if item_id in ITEM_MAP:
                color = ITEM_MAP[item_id]['color']
                rect = plt.Rectangle((c, rows - 1 - r), 1, 1, facecolor=color, edgecolor='lightgray')
                ax.add_patch(rect)
                if item_id != 0: 
                    ax.text(c + 0.5, rows - 1 - r + 0.5, ITEM_MAP[item_id]['label'][0], 
                            color='white' if color in ['black', 'darkred'] else 'black',
                            ha='center', va='center', weight='bold')

    for i in range(2): 
        r, c = base_env.agent_pos[i]
        agent_color = 'blue' if i == 0 else 'purple'
        circle = plt.Circle((c + 0.5, rows - 1 - r + 0.5), 0.3, color=agent_color, alpha=0.8)
        ax.add_patch(circle)
        ax.text(c + 0.5, rows - 1 - r + 0.5, f"A{i+1}", color='white', 
                ha='center', va='center', fontsize=9, weight='bold')
        if base_env.has_onion[i] == 1:
             ax.text(c + 0.8, rows - 1 - r + 0.8, "H", color='red', fontsize=10, weight='bold')

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')

def visualize_overcooked(env, policy_a, policy_b, k=3, delay=0.2):
    policy_a.eval()
    policy_b.eval()
    try:
        device = next(policy_a.parameters()).device
    except Exception:
        device = torch.device("cpu")
    print(f"Visualizing on device: {device}")

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.ion() 
    plt.show() 

    for episode in range(k):
        obs = env.reset()
        done = False
        step = 0
        episode_reward = 0.0
        
        while not done:
            render_grid(env, ax, step, episode_reward)
            plt.draw()
            plt.pause(delay)
            
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                if hasattr(policy_a, "get_action_and_value"):
                    output_a = policy_a.get_action_and_value(state_tensor)
                    output_b = policy_b.get_action_and_value(state_tensor)
                    action_a_tensor = output_a[0] if isinstance(output_a, tuple) else output_a
                    action_b_tensor = output_b[0] if isinstance(output_b, tuple) else output_b
                    action_a = action_a_tensor.cpu().item()
                    action_b = action_b_tensor.cpu().item()
                else:
                    logits_a = policy_a(state_tensor)
                    logits_b = policy_b(state_tensor)
                    action_a = torch.argmax(logits_a).cpu().item()
                    action_b = torch.argmax(logits_b).cpu().item()

            next_obs, reward, done, info = env.step((action_a, action_b))
            
            # --- FIX IS HERE ---
            # Handle list, numpy array, or scalar
            if isinstance(reward, (list, np.ndarray)):
                episode_reward += np.sum(reward)
            else:
                episode_reward += reward
                
            obs = next_obs
            step += 1
            if step > 100: break
        time.sleep(1.0)
    plt.ioff()
    plt.close()

def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    # Optimized to work on pre-allocated tensors
    values = values.clone().detach().to(torch.float32)
    next_value = torch.tensor(next_value, dtype=torch.float32) if not isinstance(next_value, torch.Tensor) else next_value.clone().detach().to(torch.float32)
    rewards = rewards.to(torch.float32)
    dones = dones.to(torch.float32)

    num_steps = len(rewards)
    adv = torch.zeros_like(rewards)
    lastgaelam = 0
    
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = values[t+1]
        
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        adv[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        
    returns = adv + values
    return adv, returns

def evaluate_coop_performance(agent1, agent2, device, num_episodes=5):
    env = ParallelVectorEnv([lambda: CoopCookingEnv() for _ in range(num_episodes)])
    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)
    num_steps = 128
    all_rewards = [] 
    
    for step in range(num_steps):
        with torch.no_grad():
            a1, _, _, _, _ = agent1.get_action_and_value(obs)
            a2, _, _, _, _ = agent2.get_action_and_value(obs)
            
            actions_tensor = torch.stack([a1, a2], dim=1)
            actions_np = actions_tensor.cpu().numpy()
            
            next_obs, rew, done = env.step(actions_np)
            
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            rewards_tensor = torch.as_tensor(rew, dtype=torch.float32, device=device)
            all_rewards.append(rewards_tensor)

    env.close()
    
    rewards_stacked = torch.stack(all_rewards) 
    r1 = rewards_stacked[:, :, 0].sum(dim=0).mean()
    r2 = rewards_stacked[:, :, 1].sum(dim=0).mean()
    agent1.train()
    agent2.train()
    
    return r1, r2

def compute_crossplay_matrix(agents_seat0, agents_seat1, num_episodes=20, swap_obs_seat1=False):
    """
    Run cross-play between two lists of frozen agents. Returns an
    (len(agents_seat0), len(agents_seat1)) numpy array of mean shared episode return
    (sum of both seats' rewards summed over a 128-step rollout, averaged across envs).
    swap_obs_seat1: if True, agents_seat1[j] is fed obs_for_seat(obs, ones) — used for FCP.
    Note for FCP: agents_seat0[i] (an FCP at seat 0) sees raw obs because its own seat info
    already lives in slots [0:3]. agents_seat1[j] (an FCP at seat 1) needs the slot swap so
    its own seat info appears in [0:3] like during training. obs_for_seat is a no-op for
    seat==0, so passing swap_obs_seat1=True only affects the seat-1 agent.
    """
    cpu_device = torch.device("cpu")
    M0, M1 = len(agents_seat0), len(agents_seat1)
    C = np.zeros((M0, M1), dtype=np.float32)
    num_steps = 128

    for i in range(M0):
        for j in range(M1):
            a0 = agents_seat0[i]; a0.eval()
            a1 = agents_seat1[j]; a1.eval()
            env = ParallelVectorEnv([lambda: CoopCookingEnv() for _ in range(num_episodes)])
            obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=cpu_device)
            total = torch.zeros(num_episodes, dtype=torch.float32)
            seat1_seats = np.ones(num_episodes, dtype=np.int64)  # for obs swap

            for _ in range(num_steps):
                with torch.no_grad():
                    a_0, _, _, _, _ = a0.get_action_and_value(obs)
                    obs_for_a1 = obs_for_seat(obs, seat1_seats) if swap_obs_seat1 else obs
                    a_1, _, _, _, _ = a1.get_action_and_value(obs_for_a1)
                actions = np.stack([a_0.numpy(), a_1.numpy()], axis=1)
                next_obs, rew, done = env.step(actions)
                total += torch.as_tensor(rew.sum(axis=1), dtype=torch.float32)
                obs = torch.as_tensor(next_obs, dtype=torch.float32, device=cpu_device)

            env.close()
            C[i, j] = total.mean().item()
            print(f"  crossplay [{i},{j}] = {C[i,j]:.2f}")
    return C

# ==========================================
# 5. Training Functions (Buffer Optimized)
# ==========================================
def train_independent_ppo(agents=None, steps=50000, seed=None):
    print("\n=== Training Independent PPO (Optimized) ===")
    if seed is not None:
        np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    
    num_envs = 8
    num_steps = 128
    env = ParallelVectorEnv([lambda: CoopCookingEnv() for _ in range(num_envs)])
    
    if agents is None:
        agent1 = Agent(8, 5).to(cpu_device)
        agent2 = Agent(8, 5).to(cpu_device)
    else:
        agent1, agent2 = agents
    opt1 = optim.Adam(agent1.parameters(), lr=6e-4)
    opt2 = optim.Adam(agent2.parameters(), lr=6e-4)
    
    # Pre-allocate buffers for speed
    obs_buf = torch.zeros((num_steps, num_envs, 8), dtype=torch.float32, device=cpu_device)
    a1_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    a2_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    lp1_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    lp2_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    v1_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    v2_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    rew_buf = torch.zeros((num_steps, num_envs, 2), dtype=torch.float32, device=cpu_device)
    done_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)

    rewards_log=[]
    total_updates = steps // (num_steps * num_envs) + 1
    
    # Reset ONCE outside the loop (State persists across updates)
    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=cpu_device)
    
    for update in range(1, total_updates):
        # Reset ONCE outside the loop (State persists across updates)
        obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=cpu_device)
        # --- Collection Phase (CPU + Buffers) ---
        for step in range(num_steps):
            obs_buf[step] = obs
            with torch.no_grad():
                a1, lp1, _, v1, _ = agent1.get_action_and_value(obs)
                a2, lp2, _, v2, _ = agent2.get_action_and_value(obs)
            
            a1_buf[step], a2_buf[step] = a1.float(), a2.float()
            lp1_buf[step], lp2_buf[step] = lp1, lp2
            v1_buf[step], v2_buf[step] = v1.flatten(), v2.flatten()
            
            actions = np.stack([a1.numpy(), a2.numpy()], axis=1)
            next_obs, rew, done = env.step(actions)
            
            rew_buf[step] = torch.as_tensor(rew, dtype=torch.float32)
            done_buf[step] = torch.as_tensor(done, dtype=torch.float32)
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=cpu_device)

        # --- Update Phase (GPU) ---
        agent1.to(device); agent2.to(device)
        
        with torch.no_grad():
            gpu_obs = obs.to(device)
            _, _, _, next_v1, _ = agent1.get_action_and_value(gpu_obs)
            _, _, _, next_v2, _ = agent2.get_action_and_value(gpu_obs)
            
        def update_agent(agent, opt, b_obs, b_a, b_lp, b_v, b_rew, b_done, next_v, agent_idx):
            dev_obs = b_obs.to(device)
            dev_a = b_a.to(device)
            dev_lp = b_lp.to(device)
            dev_v = b_v.to(device)
            dev_done = b_done.to(device)
            dev_rew = b_rew.to(device)
            
            agent_rewards = dev_rew[:, :, agent_idx]
            adv, ret = compute_gae(agent_rewards, dev_v, next_v.flatten(), dev_done)
            
            flat_obs = dev_obs.view(-1, 8)
            flat_a = dev_a.view(-1)
            flat_lp = dev_lp.view(-1)
            flat_adv = adv.view(-1)
            flat_ret = ret.view(-1)
            
            for _ in range(4):
                _, new_lp, ent, new_v, _ = agent.get_action_and_value(flat_obs, flat_a)
                ratio = torch.exp(new_lp - flat_lp)
                surr1 = ratio * flat_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * flat_adv
                loss_p = -torch.min(surr1, surr2).mean()
                loss_v = 0.5 * ((new_v.flatten() - flat_ret) ** 2).mean()
                loss = loss_p + loss_v - 0.1 * ent.mean()
                
                opt.zero_grad()
                loss.backward()
                opt.step()
            return agent_rewards.mean().item()

        r1 = update_agent(agent1, opt1, obs_buf, a1_buf, lp1_buf, v1_buf, rew_buf, done_buf, next_v1, 0)
        r2 = update_agent(agent2, opt2, obs_buf, a2_buf, lp2_buf, v2_buf, rew_buf, done_buf, next_v2, 1)
        
        agent1.to(cpu_device); agent2.to(cpu_device)
        
        if update % 10 == 0:
            r1,r2 = evaluate_coop_performance(agent1, agent2, cpu_device)
            print(f"Update {update}: Van PPO Agent 1 = {r1:.2f}, Van PPO Agent 2 = {r2:.2f}")
            rewards_log.append([r1,r2])

    if seed is not None:
        torch.save({
            'type': 'Vanilla',
            'seed': seed,
            'model_state_dict1': agent1.state_dict(),
            'model_state_dict2': agent2.state_dict(),
            'rewards': rewards_log,
        }, f'van_ppo{seed}.pt2')
        print(f"Saved van_ppo{seed}.pt2")
    env.close()
    return [agent1, agent2], rewards_log

def train_adversarial_marl(agents=None, risk=0.2, bnd_rationality=0.1, total_steps=50000, seed=None):
    print("\n=== Training Adversarial MARL (Optimized) ===")
    if seed is not None:
        np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    
    num_envs = 8
    num_steps = 128
    env = ParallelVectorEnv([lambda: CoopCookingEnv() for _ in range(num_envs)])

    if agents is None:
        p1 = Agent(8, 5).to(cpu_device)
        p2 = Agent(8, 5).to(cpu_device)
        adv1 = copy.deepcopy(p1).to(cpu_device)
        adv2 = copy.deepcopy(p2).to(cpu_device)
    else:
        p1, p2, adv1, adv2 = agents
    
    opts = {
        "p1": optim.Adam(p1.parameters(), lr=6e-4),
        "p2": optim.Adam(p2.parameters(), lr=6e-4),
        "adv1": optim.Adam(adv1.parameters(), lr=6e-4),
        "adv2": optim.Adam(adv2.parameters(), lr=6e-4)
    }

    kl_coef = risk
    entropy_coeff = bnd_rationality
    rewards_log = []
    
    # Common Buffers
    obs_buf = torch.zeros((num_steps, num_envs, 8), dtype=torch.float32, device=cpu_device)
    rew_buf = torch.zeros((num_steps, num_envs, 2), dtype=torch.float32, device=cpu_device)
    done_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    
    # Specific buffers - reused for both phases to save memory
    a1_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    a2_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    lp1_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    lp2_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    v1_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    v2_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)

    total_updates = total_steps // (num_steps * num_envs) + 1
    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=cpu_device)

    for update in range(1, total_updates):
        
        def update_step(agent, opt, b_obs, b_a, b_lp, b_v, b_rew, b_done, next_val, ref_model, reward_idx, maximize_reward=True):
            dev_obs = b_obs.to(device)
            dev_a = b_a.to(device)
            dev_lp = b_lp.to(device)
            dev_v = b_v.to(device)
            dev_done = b_done.to(device)
            dev_rew = b_rew.to(device)

            r_vec = dev_rew[:, :, reward_idx]
            if not maximize_reward: r_vec = -r_vec 
            
            adv, ret = compute_gae(r_vec, dev_v, next_val.flatten(), dev_done)
            
            flat_obs = dev_obs.view(-1, 8)
            flat_a = dev_a.view(-1)
            flat_lp = dev_lp.view(-1)
            flat_adv = adv.view(-1)
            flat_ret = ret.view(-1)
            
            for _ in range(4):
                _, new_lp, ent, new_v, curr_dist = agent.get_action_and_value(flat_obs, flat_a)
                ratio = torch.exp(new_lp - flat_lp)
                surr1 = ratio * flat_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * flat_adv
                loss_p = -torch.min(surr1, surr2).mean()
                loss_v = 0.5 * ((new_v.flatten() - flat_ret) ** 2).mean()
                if ref_model is None:
                    loss = loss_p + loss_v - entropy_coeff * ent.mean()
                else:
                    with torch.no_grad():
                        _, _, _, _, ref_dist = ref_model.get_action_and_value(flat_obs)
                    kl = torch.distributions.kl.kl_divergence(ref_dist, curr_dist).mean()
                    loss = loss_p + loss_v + kl_coef * kl
                
                opt.zero_grad()
                loss.backward()
                opt.step()

        # ==================================
        # PHASE A: P1 (Coop) + Adv1 (Adv)
        # ==================================
        obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=cpu_device) # Reset for phase A
        for step in range(num_steps):
            obs_buf[step] = obs
            with torch.no_grad():
                a_p1, lp_p1, _, v_p1, _ = p1.get_action_and_value(obs)
                a_adv1, lp_adv1, _, v_adv1, _ = adv1.get_action_and_value(obs)
            
            a1_buf[step] = a_p1.float(); a2_buf[step] = a_adv1.float()
            lp1_buf[step] = lp_p1; lp2_buf[step] = lp_adv1
            v1_buf[step] = v_p1.flatten(); v2_buf[step] = v_adv1.flatten()
            
            actions = np.stack([a_p1.numpy(), a_adv1.numpy()], axis=1)
            next_obs, rew, done = env.step(actions)
            
            rew_buf[step] = torch.as_tensor(rew, dtype=torch.float32)
            done_buf[step] = torch.as_tensor(done, dtype=torch.float32)
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=cpu_device)

        # Move to GPU
        p1.to(device); adv1.to(device); p2.to(device)
        with torch.no_grad():
            gpu_obs = obs.to(device)
            _, _, _, nv_p1, _ = p1.get_action_and_value(gpu_obs)
            _, _, _, nv_adv1, _ = adv1.get_action_and_value(gpu_obs)

        # Update
        update_step(p1, opts["p1"], obs_buf, a1_buf, lp1_buf, v1_buf, rew_buf, done_buf, nv_p1, None, 0, True)
        update_step(adv1, opts["adv1"], obs_buf, a2_buf, lp2_buf, v2_buf, rew_buf, done_buf, nv_adv1, p2, 0, False)
        
        # Back to CPU
        p1.to(cpu_device); adv1.to(cpu_device); p2.to(cpu_device)

        # ==================================
        # PHASE B: Adv2 (Adv) + P2 (Coop)
        # ==================================
        obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=cpu_device) # Reset for phase B
        for step in range(num_steps):
            obs_buf[step] = obs
            with torch.no_grad():
                a_adv2, lp_adv2, _, v_adv2, _ = adv2.get_action_and_value(obs)
                a_p2, lp_p2, _, v_p2, _ = p2.get_action_and_value(obs)
            
            a1_buf[step] = a_adv2.float(); a2_buf[step] = a_p2.float()
            lp1_buf[step] = lp_adv2; lp2_buf[step] = lp_p2
            v1_buf[step] = v_adv2.flatten(); v2_buf[step] = v_p2.flatten()
            
            actions = np.stack([a_adv2.numpy(), a_p2.numpy()], axis=1)
            next_obs, rew, done = env.step(actions)
            
            rew_buf[step] = torch.as_tensor(rew, dtype=torch.float32)
            done_buf[step] = torch.as_tensor(done, dtype=torch.float32)
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=cpu_device)

        # Move to GPU
        adv2.to(device); p2.to(device); p1.to(device)
        with torch.no_grad():
            gpu_obs = obs.to(device)
            _, _, _, nv_adv2, _ = adv2.get_action_and_value(gpu_obs)
            _, _, _, nv_p2, _ = p2.get_action_and_value(gpu_obs)

        # Update
        update_step(adv2, opts["adv2"], obs_buf, a1_buf, lp1_buf, v1_buf, rew_buf, done_buf, nv_adv2, p1, 1, False)
        update_step(p2, opts["p2"], obs_buf, a2_buf, lp2_buf, v2_buf, rew_buf, done_buf, nv_p2, None, 1, True)

        # Back to CPU
        adv2.to(cpu_device); p2.to(cpu_device); p1.to(cpu_device)

        if update % 10 == 0:
            r1,r2 = evaluate_coop_performance(p1, p2, cpu_device)
            print(f"Update {update}: Robust PPO Agent 1 = {r1:.2f}, Robust PPO Agent 2 = {r2:.2f}")
            rewards_log.append([r1,r2])

    if seed is not None:
        torch.save({
            'type': 'Robust',
            'seed': seed,
            'model_state_dict1': p1.state_dict(),
            'model_state_dict2': p2.state_dict(),
            'adv_state_dict1': adv1.state_dict(),
            'adv_state_dict2': adv2.state_dict(),
            'rewards': rewards_log,
        }, f'risk_ppo{seed}.pt2')
        print(f"Saved risk_ppo{seed}.pt2")
    env.close()
    return [p1,p2,adv1,adv2], rewards_log

def train_pure_adversarial(agents=None, total_steps=50000, seed=None):
    """
    Pure adversarial baseline: same two-phase structure as train_adversarial_marl
    but with NO KL regularization on adversaries and NO entropy bonus on cooperators.
    Saves to pure_adv{seed}.pt2 with the same checkpoint layout as risk_ppo*.pt2.
    """
    print("\n=== Training Pure Adversarial (no regularization) ===")
    if seed is not None:
        np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    num_envs = 8
    num_steps = 128
    env = ParallelVectorEnv([lambda: CoopCookingEnv() for _ in range(num_envs)])

    if agents is None:
        p1 = Agent(8, 5).to(cpu_device)
        p2 = Agent(8, 5).to(cpu_device)
        adv1 = copy.deepcopy(p1).to(cpu_device)
        adv2 = copy.deepcopy(p2).to(cpu_device)
    else:
        p1, p2, adv1, adv2 = agents

    opts = {
        "p1":   optim.Adam(p1.parameters(),   lr=6e-4),
        "p2":   optim.Adam(p2.parameters(),   lr=6e-4),
        "adv1": optim.Adam(adv1.parameters(), lr=6e-4),
        "adv2": optim.Adam(adv2.parameters(), lr=6e-4),
    }

    rewards_log = []

    obs_buf = torch.zeros((num_steps, num_envs, 8), dtype=torch.float32, device=cpu_device)
    rew_buf = torch.zeros((num_steps, num_envs, 2), dtype=torch.float32, device=cpu_device)
    done_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    a1_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    a2_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    lp1_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    lp2_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    v1_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    v2_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)

    total_updates = total_steps // (num_steps * num_envs) + 1
    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=cpu_device)

    for update in range(1, total_updates):

        def update_step(agent, opt, b_obs, b_a, b_lp, b_v, b_rew, b_done, next_val, reward_idx, maximize_reward=True):
            dev_obs = b_obs.to(device)
            dev_a = b_a.to(device)
            dev_lp = b_lp.to(device)
            dev_v = b_v.to(device)
            dev_done = b_done.to(device)
            dev_rew = b_rew.to(device)

            r_vec = dev_rew[:, :, reward_idx]
            if not maximize_reward:
                r_vec = -r_vec

            adv, ret = compute_gae(r_vec, dev_v, next_val.flatten(), dev_done)

            flat_obs = dev_obs.view(-1, 8)
            flat_a = dev_a.view(-1)
            flat_lp = dev_lp.view(-1)
            flat_adv = adv.view(-1)
            flat_ret = ret.view(-1)

            for _ in range(4):
                _, new_lp, ent, new_v, _ = agent.get_action_and_value(flat_obs, flat_a)
                ratio = torch.exp(new_lp - flat_lp)
                surr1 = ratio * flat_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * flat_adv
                loss_p = -torch.min(surr1, surr2).mean()
                loss_v = 0.5 * ((new_v.flatten() - flat_ret) ** 2).mean()
                loss = loss_p + loss_v

                opt.zero_grad()
                loss.backward()
                opt.step()

        # PHASE A: P1 (Coop, max reward 0) + Adv1 (Adv, min reward 0)
        obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=cpu_device)
        for step in range(num_steps):
            obs_buf[step] = obs
            with torch.no_grad():
                a_p1, lp_p1, _, v_p1, _ = p1.get_action_and_value(obs)
                a_adv1, lp_adv1, _, v_adv1, _ = adv1.get_action_and_value(obs)

            a1_buf[step] = a_p1.float();   a2_buf[step] = a_adv1.float()
            lp1_buf[step] = lp_p1;          lp2_buf[step] = lp_adv1
            v1_buf[step] = v_p1.flatten(); v2_buf[step] = v_adv1.flatten()

            actions = np.stack([a_p1.numpy(), a_adv1.numpy()], axis=1)
            next_obs, rew, done = env.step(actions)
            rew_buf[step] = torch.as_tensor(rew, dtype=torch.float32)
            done_buf[step] = torch.as_tensor(done, dtype=torch.float32)
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=cpu_device)

        p1.to(device); adv1.to(device)
        with torch.no_grad():
            gpu_obs = obs.to(device)
            _, _, _, nv_p1, _ = p1.get_action_and_value(gpu_obs)
            _, _, _, nv_adv1, _ = adv1.get_action_and_value(gpu_obs)

        update_step(p1,   opts["p1"],   obs_buf, a1_buf, lp1_buf, v1_buf, rew_buf, done_buf, nv_p1,   0, True)
        update_step(adv1, opts["adv1"], obs_buf, a2_buf, lp2_buf, v2_buf, rew_buf, done_buf, nv_adv1, 0, False)

        p1.to(cpu_device); adv1.to(cpu_device)

        # PHASE B: Adv2 (Adv, min reward 1) + P2 (Coop, max reward 1)
        obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=cpu_device)
        for step in range(num_steps):
            obs_buf[step] = obs
            with torch.no_grad():
                a_adv2, lp_adv2, _, v_adv2, _ = adv2.get_action_and_value(obs)
                a_p2, lp_p2, _, v_p2, _ = p2.get_action_and_value(obs)

            a1_buf[step] = a_adv2.float(); a2_buf[step] = a_p2.float()
            lp1_buf[step] = lp_adv2;        lp2_buf[step] = lp_p2
            v1_buf[step] = v_adv2.flatten(); v2_buf[step] = v_p2.flatten()

            actions = np.stack([a_adv2.numpy(), a_p2.numpy()], axis=1)
            next_obs, rew, done = env.step(actions)
            rew_buf[step] = torch.as_tensor(rew, dtype=torch.float32)
            done_buf[step] = torch.as_tensor(done, dtype=torch.float32)
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=cpu_device)

        adv2.to(device); p2.to(device)
        with torch.no_grad():
            gpu_obs = obs.to(device)
            _, _, _, nv_adv2, _ = adv2.get_action_and_value(gpu_obs)
            _, _, _, nv_p2, _ = p2.get_action_and_value(gpu_obs)

        update_step(adv2, opts["adv2"], obs_buf, a1_buf, lp1_buf, v1_buf, rew_buf, done_buf, nv_adv2, 1, False)
        update_step(p2,   opts["p2"],   obs_buf, a2_buf, lp2_buf, v2_buf, rew_buf, done_buf, nv_p2,   1, True)

        adv2.to(cpu_device); p2.to(cpu_device)

        if update % 10 == 0:
            r1, r2 = evaluate_coop_performance(p1, p2, cpu_device)
            print(f"Update {update}: PureAdv Agent 1 = {r1:.2f}, Agent 2 = {r2:.2f}")
            rewards_log.append([r1, r2])

    if seed is not None:
        torch.save({
            'type': 'PureAdv',
            'seed': seed,
            'model_state_dict1': p1.state_dict(),
            'model_state_dict2': p2.state_dict(),
            'adv_state_dict1':   adv1.state_dict(),
            'adv_state_dict2':   adv2.state_dict(),
            'rewards': rewards_log,
        }, f'pure_adv{seed}.pt2')
        print(f"Saved pure_adv{seed}.pt2")
    env.close()
    return [p1, p2, adv1, adv2], rewards_log

def train_self_play(seed=0, total_steps=2_000_000):
    """
    True self-play: one shared Agent network controls both seats.
    Both seats' trajectories are concatenated into a single buffer
    (shape (num_steps, 2*num_envs, ...)) and a single PPO update is performed.
    Saves K=3 checkpoints (early/mid/final) to sp_seed{seed}.pt2.
    """
    print(f"\n=== Training Self-Play seed={seed} ===")
    np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    num_envs = 8
    num_steps = 128
    env = ParallelVectorEnv([lambda: CoopCookingEnv() for _ in range(num_envs)])

    agent = Agent(8, 5).to(cpu_device)
    opt = optim.Adam(agent.parameters(), lr=6e-4)

    # Buffers: 2*num_envs because both seats produce trajectories
    B = 2 * num_envs
    obs_buf = torch.zeros((num_steps, B, 8), dtype=torch.float32, device=cpu_device)
    a_buf  = torch.zeros((num_steps, B), dtype=torch.float32, device=cpu_device)
    lp_buf = torch.zeros((num_steps, B), dtype=torch.float32, device=cpu_device)
    v_buf  = torch.zeros((num_steps, B), dtype=torch.float32, device=cpu_device)
    rew_buf = torch.zeros((num_steps, B), dtype=torch.float32, device=cpu_device)
    done_buf = torch.zeros((num_steps, B), dtype=torch.float32, device=cpu_device)

    rewards_log = []
    total_updates = total_steps // (num_steps * num_envs) + 1

    # Checkpoint schedule: early ~25%, mid ~50%, final = last update
    ckpt_updates = {
        max(1, total_updates // 4): 'early',
        max(1, total_updates // 2): 'mid',
        total_updates - 1:          'final',
    }
    checkpoints = []

    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=cpu_device)

    for update in range(1, total_updates):
        obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=cpu_device)

        for step in range(num_steps):
            # Forward both seats on the same obs (one batched call: stack [obs, obs] gives 2N rows
            # but we want different samples per seat, so do two independent forwards)
            with torch.no_grad():
                a0, lp0, _, v0, _ = agent.get_action_and_value(obs)
                a1, lp1, _, v1, _ = agent.get_action_and_value(obs)

            # Stack into 2*num_envs along env dim: rows [0..N) = seat 0, rows [N..2N) = seat 1
            obs_buf[step, :num_envs] = obs
            obs_buf[step, num_envs:] = obs
            a_buf[step, :num_envs] = a0.float(); a_buf[step, num_envs:] = a1.float()
            lp_buf[step, :num_envs] = lp0; lp_buf[step, num_envs:] = lp1
            v_buf[step, :num_envs] = v0.flatten(); v_buf[step, num_envs:] = v1.flatten()

            actions = np.stack([a0.numpy(), a1.numpy()], axis=1)
            next_obs, rew, done = env.step(actions)

            rew_buf[step, :num_envs] = torch.as_tensor(rew[:, 0], dtype=torch.float32)
            rew_buf[step, num_envs:] = torch.as_tensor(rew[:, 1], dtype=torch.float32)
            done_t = torch.as_tensor(done, dtype=torch.float32)
            done_buf[step, :num_envs] = done_t
            done_buf[step, num_envs:] = done_t
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=cpu_device)

        # Update phase on GPU
        agent.to(device)
        with torch.no_grad():
            gpu_obs = obs.to(device)
            _, _, _, next_v, _ = agent.get_action_and_value(gpu_obs)
        next_v_full = torch.cat([next_v.flatten(), next_v.flatten()], dim=0)

        dev_obs = obs_buf.to(device)
        dev_a = a_buf.to(device)
        dev_lp = lp_buf.to(device)
        dev_v = v_buf.to(device)
        dev_rew = rew_buf.to(device)
        dev_done = done_buf.to(device)

        adv, ret = compute_gae(dev_rew, dev_v, next_v_full, dev_done)

        flat_obs = dev_obs.view(-1, 8)
        flat_a = dev_a.view(-1)
        flat_lp = dev_lp.view(-1)
        flat_adv = adv.view(-1)
        flat_ret = ret.view(-1)

        for _ in range(4):
            _, new_lp, ent, new_v, _ = agent.get_action_and_value(flat_obs, flat_a)
            ratio = torch.exp(new_lp - flat_lp)
            surr1 = ratio * flat_adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * flat_adv
            loss_p = -torch.min(surr1, surr2).mean()
            loss_v = 0.5 * ((new_v.flatten() - flat_ret) ** 2).mean()
            loss = loss_p + loss_v - 0.1 * ent.mean()
            opt.zero_grad(); loss.backward(); opt.step()

        agent.to(cpu_device)

        # Capture checkpoint if scheduled
        if update in ckpt_updates:
            tier = ckpt_updates[update]
            sd = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}
            checkpoints.append({'tier': tier, 'update': update, 'state_dict': sd})

        if update % 10 == 0:
            r1, r2 = evaluate_coop_performance(agent, agent, cpu_device)
            print(f"Update {update}: SP seed={seed} r1={r1:.2f} r2={r2:.2f}")
            rewards_log.append([r1, r2])

    env.close()

    torch.save({
        'type': 'SelfPlay',
        'seed': seed,
        'checkpoints': checkpoints,
        'rewards': rewards_log,
    }, f'sp_seed{seed}.pt2')
    print(f"Saved sp_seed{seed}.pt2 ({len(checkpoints)} checkpoints)")
    return agent, rewards_log

def _load_partner_pool(pool_paths, device):
    """Load all checkpoints from a list of sp_seed*.pt2 files into frozen Agent instances."""
    partners = []
    for path in pool_paths:
        d = torch.load(path, weights_only=True)
        assert d['type'] == 'SelfPlay', f"Pool file {path} is not SelfPlay"
        for ckpt in d['checkpoints']:
            a = Agent(8, 5).to(device)
            a.load_state_dict(ckpt['state_dict'])
            a.eval()
            for p in a.parameters():
                p.requires_grad_(False)
            partners.append(a)
    return partners

def train_fcp(pool_paths, seed=0, total_steps=2_000_000):
    """
    Best-response PPO against a fixed pool of self-play partners.
    Each env independently samples (partner_idx, fcp_seat) on episode boundaries.
    FCP sees obs reordered so its own seat info is in slots [0,1,2]; partners see raw obs.
    Saves final policy to fcp_seed{seed}.pt2.
    """
    print(f"\n=== Training FCP seed={seed} (pool size {len(pool_paths)}) ===")
    np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    num_envs = 8
    num_steps = 128
    env = ParallelVectorEnv([lambda: CoopCookingEnv() for _ in range(num_envs)])

    partners = _load_partner_pool(pool_paths, cpu_device)
    P = len(partners)
    print(f"Loaded {P} partners")

    fcp = Agent(8, 5).to(cpu_device)
    opt = optim.Adam(fcp.parameters(), lr=6e-4)

    obs_buf = torch.zeros((num_steps, num_envs, 8), dtype=torch.float32, device=cpu_device)
    a_buf  = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    lp_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    v_buf  = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    rew_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)
    done_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=cpu_device)

    rewards_log = []
    total_updates = total_steps // (num_steps * num_envs) + 1

    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=cpu_device)
    partner_idx = np.random.randint(0, P, size=num_envs)
    fcp_seat = np.random.randint(0, 2, size=num_envs)

    for update in range(1, total_updates):
        obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=cpu_device)
        partner_idx = np.random.randint(0, P, size=num_envs)
        fcp_seat = np.random.randint(0, 2, size=num_envs)

        for step in range(num_steps):
            obs_fcp = obs_for_seat(obs, fcp_seat)
            obs_buf[step] = obs_fcp

            with torch.no_grad():
                a_fcp, lp_fcp, _, v_fcp, _ = fcp.get_action_and_value(obs_fcp)

            # Partner forwards: group envs by partner_idx
            a_partner = np.zeros(num_envs, dtype=np.int64)
            with torch.no_grad():
                for pid in np.unique(partner_idx):
                    mask = (partner_idx == pid)
                    sub_obs = obs[mask]
                    a_p, _, _, _, _ = partners[int(pid)].get_action_and_value(sub_obs)
                    a_partner[mask] = a_p.numpy()

            a_fcp_np = a_fcp.numpy()
            actions = np.zeros((num_envs, 2), dtype=np.int64)
            for e in range(num_envs):
                if fcp_seat[e] == 0:
                    actions[e, 0] = a_fcp_np[e]; actions[e, 1] = a_partner[e]
                else:
                    actions[e, 0] = a_partner[e]; actions[e, 1] = a_fcp_np[e]

            a_buf[step] = a_fcp.float()
            lp_buf[step] = lp_fcp
            v_buf[step] = v_fcp.flatten()

            next_obs, rew, done = env.step(actions)
            # FCP reward = reward of the seat FCP plays in each env
            fcp_reward = rew[np.arange(num_envs), fcp_seat]
            rew_buf[step] = torch.as_tensor(fcp_reward, dtype=torch.float32)
            done_buf[step] = torch.as_tensor(done, dtype=torch.float32)
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=cpu_device)

            # Resample partner+seat for envs that just finished an episode
            for e in range(num_envs):
                if done[e]:
                    partner_idx[e] = np.random.randint(0, P)
                    fcp_seat[e] = np.random.randint(0, 2)

        # Update phase on GPU
        fcp.to(device)
        with torch.no_grad():
            final_obs_fcp = obs_for_seat(obs, fcp_seat).to(device)
            _, _, _, next_v, _ = fcp.get_action_and_value(final_obs_fcp)

        dev_obs = obs_buf.to(device)
        dev_a = a_buf.to(device)
        dev_lp = lp_buf.to(device)
        dev_v = v_buf.to(device)
        dev_rew = rew_buf.to(device)
        dev_done = done_buf.to(device)

        adv, ret = compute_gae(dev_rew, dev_v, next_v.flatten(), dev_done)

        flat_obs = dev_obs.view(-1, 8)
        flat_a = dev_a.view(-1)
        flat_lp = dev_lp.view(-1)
        flat_adv = adv.view(-1)
        flat_ret = ret.view(-1)

        for _ in range(4):
            _, new_lp, ent, new_v, _ = fcp.get_action_and_value(flat_obs, flat_a)
            ratio = torch.exp(new_lp - flat_lp)
            surr1 = ratio * flat_adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * flat_adv
            loss_p = -torch.min(surr1, surr2).mean()
            loss_v = 0.5 * ((new_v.flatten() - flat_ret) ** 2).mean()
            loss = loss_p + loss_v - 0.1 * ent.mean()
            opt.zero_grad(); loss.backward(); opt.step()

        fcp.to(cpu_device)

        if update % 10 == 0:
            # Eval: sample 5 partners uniformly from pool, FCP in seat 0, mean shared return
            sampled = np.random.choice(P, size=min(5, P), replace=False)
            returns = []
            for pid in sampled:
                r1, r2 = evaluate_coop_performance(fcp, partners[int(pid)], cpu_device, num_episodes=5)
                returns.append(float(r1) + float(r2))
            mean_ret = float(np.mean(returns))
            print(f"Update {update}: FCP seed={seed} mean_pool_return={mean_ret:.2f}")
            # Note: rewards_log semantics differs from van/risk. Van/risk log
            # per-seat returns of one fixed self-play pair; FCP logs the mean
            # shared return averaged over a sampled set of pool partners.
            # We split it in half and duplicate into a 2-col shape so the
            # benchmark's rs[:,-1] plot code works uniformly across methods —
            # a curve point's value is mean_pool_return / 2 (per-seat-equivalent).
            rewards_log.append([mean_ret / 2.0, mean_ret / 2.0])

    env.close()

    torch.save({
        'type': 'FCP',
        'seed': seed,
        'pool_paths': list(pool_paths),
        'state_dict': fcp.state_dict(),
        'rewards': rewards_log,
    }, f'fcp_seed{seed}.pt2')
    print(f"Saved fcp_seed{seed}.pt2")
    return fcp, rewards_log

def upload_agent(p1,p2,filename):
    saved = torch.load(filename, weights_only=True)
    p1.load_state_dict(saved.get('model_state_dict1'))
    p2.load_state_dict(saved.get('model_state_dict2'))
    rewards=saved.get('rewards')
    return [p1,p2,rewards]

def upload_rob_agent(p1,p2,adv1,adv2,filename):
    saved = torch.load(filename, weights_only=True)
    p1.load_state_dict(saved.get('model_state_dict1'))
    p2.load_state_dict(saved.get('model_state_dict2'))
    if saved['type']=='Robust':
        adv1.load_state_dict(saved.get('adv_state_dict1'))
        adv2.load_state_dict(saved.get('adv_state_dict2'))
    rewards=saved.get('rewards')
    return [p1,p2,rewards]

# # ==========================================
# # Main Guard
# # ==========================================
# if __name__ == "__main__":
#     try:
#         mp.set_start_method('spawn')
#     except RuntimeError:
#         pass
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#     cpu_device = torch.device("cpu")
        
#     filename=None
#     benchmark=0
#     train=1

#     print("Starting Training...")
    
#     tau=10
#     risk=1/tau

#     p1 = Agent(8, 5).to(cpu_device)
#     p2 = Agent(8, 5).to(cpu_device)
#     adv1 = copy.deepcopy(p1).to(cpu_device)
#     adv2 = copy.deepcopy(p2).to(cpu_device)

#     if train:
#         ppo_agents,ppo_rewards = train_independent_ppo([p1,p2],steps=20)
#         #risk_agents,risk_rewards=train_adversarial_marl([p1,p2,adv1,adv2],risk,total_steps=2000000)

#     if benchmark:
#         risk_fns=['risk_ppo1.pt2', 'risk_ppo2.pt2', 'risk_ppo3.pt2', 'risk_ppo4.pt2', 'risk_ppo5.pt2']
#         van_fns=['van_ppo1.pt2', 'van_ppo2.pt2', 'van_ppo3.pt2', 'van_ppo4.pt2', 'van_ppo5.pt2']
#         fnames=van_fns+risk_fns

#         rs_rob=[np.array(upload_rob_agent(p1,p2,adv1,adv2,filename)[2]) for filename in risk_fns]
#         rs_van=[np.array(upload_agent(p1,p2,filename)[2]) for filename in van_fns]

#         rs2=np.mean(np.array(rs_van),axis=0)
#         st2=np.std(np.array(rs_van),axis=0)
#         plt.plot(rs2[:,-1],'r')
#         plt.fill_between(range(len(rs2[:,-1])),rs2[:,-1]-st2[:,-1],rs2[:,-1]+st2[:,-1],fc='r',alpha=0.1)

#         rs1=np.mean(np.array(rs_rob),axis=0)
#         st1=np.std(np.array(rs_rob),axis=0)
#         plt.plot(rs1[:,-1],'b')
#         plt.fill_between(range(len(rs1[:,-1])),rs1[:,-1]-st1[:,-1],rs1[:,-1]+st1[:,-1],fc='b',alpha=0.1)
#         plt.pause(0.1)

#         a1 = copy.deepcopy(p1).to(cpu_device)
#         a2 = copy.deepcopy(p2).to(cpu_device)
#         comp1=np.zeros([len(fnames),len(fnames)])
#         comp2=np.zeros([len(fnames),len(fnames)])

#         i=0
#         for f1 in fnames:
#             print(i)
#             if f1[0]=='r':
#                 p1,p2,_=upload_rob_agent(p1,p2,adv1,adv2,f1)
#             else:
#                 p1,p2,_=upload_agent(p1,p2,f1)
#             j=0
#             for f2 in fnames:
#                 if f1[0]=='r':
#                     a1,a2,_=upload_rob_agent(a1,a2,adv1,adv2,f2)
#                 else:
#                     a1,a2,_=upload_agent(a1,a2,f2)

#                 c1=evaluate_coop_performance(p1, a2, cpu_device,5)
#                 comp1[i,j]=c1[0]
#                 comp2[i,j]=c1[1]
#                 j+=1
#             i+=1