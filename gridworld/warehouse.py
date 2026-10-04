"""
RWARE IPPO — Vanilla (parameter sharing) + Risk-Averse (adversarial partners)

Design choices:
  - Raw RWARE observations passed through unchanged.
  - Separate actor and critic MLPs: no shared trunk, no gradient interference.
  - Flat buffer (T*E*A,) from the start — no per-agent loops.
  - Always bootstrap V(s') — no special truncation handling.
  - Minibatch PPO with shuffle.
  - No RunningNormalize — RWARE obs are small bounded binary/integer values.
  - Adversarial: random slot assignment each rollout.
  - Reverse KL: KL(adv ∥ true) keeps adversary in support of true policy.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import copy
import time
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import random as random

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
NUM_WORKERS           = 8       # match typical core count; each worker runs AsyncVectorEnv
GLOBAL_ENV_NAME       = "rware-tiny-4ag-v2"
GLOBAL_PICKUP_SHAPING = 0.01


# ---------------------------------------------------------------------------
# Observation processing
# ---------------------------------------------------------------------------
# Raw RWARE obs (fast_obs=True, msg_bits=0):
#   [x, y, carrying_shelf, dir_onehot(4), on_highway, sensor_cells(63)] = 71 dims
# Passed through unchanged — obs_dim computed at runtime.

def process_obs_batch(obs_tuple) -> np.ndarray:
    return np.stack(obs_tuple).astype(np.float32)   # (A, obs_dim)


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------
class RWAREWrapper:
    def __init__(self, env_name: str = GLOBAL_ENV_NAME, pickup_shaping: float = 0.01):
        import rware  # noqa: F401
        self.env             = gym.make(env_name)
        self.num_agents      = self.env.unwrapped.n_agents
        self.action_dim      = self.env.action_space[0].n
        self.pickup_shaping  = pickup_shaping
        self._prev_carrying  = np.zeros(self.num_agents, dtype=bool)
        _raw, _              = self.env.reset()
        self._prev_carrying[:] = False
        self.obs_dim         = np.array(_raw[0], dtype=np.float32).shape[0]

    def reset(self):
        obs_tuple, _ = self.env.reset()
        self._prev_carrying[:] = False
        return process_obs_batch(obs_tuple)       # (A, obs_dim)

    def step(self, actions: np.ndarray):
        obs_tuple, rewards, done, truncated, _ = self.env.step(
            tuple(int(a) for a in actions)
        )
        obs = process_obs_batch(obs_tuple)
        rew = np.array(rewards, dtype=np.float32)

        if self.pickup_shaping > 0.0:
            w            = self.env.unwrapped
            req          = set(w.request_queue)
            carrying_req = np.array([
                a.carrying_shelf is not None and a.carrying_shelf in req
                for a in w.agents
            ])
            rew += (carrying_req & ~self._prev_carrying).astype(np.float32) * self.pickup_shaping
            self._prev_carrying = carrying_req

        return obs, rew, done or truncated, done and not truncated, truncated

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Parallel workers — rollout-in-worker design
# ---------------------------------------------------------------------------
# Each worker owns its slice of envs AND runs the full T-step rollout locally.
# Main sends policy weights once per update; workers fill shared-memory buffers
# with the complete (T, E_w, A, ...) rollout; main reads once.
# IPC cost: 1 round-trip per update instead of T round-trips.
# ---------------------------------------------------------------------------

def _make_env():
    return RWAREWrapper(GLOBAL_ENV_NAME, pickup_shaping=GLOBAL_PICKUP_SHAPING)


class _CloudpickleWrapper:
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def _rollout_worker(worker_id, env_fn_wrapper, envs_per_worker,
                    shm_names, shapes, dtypes,
                    sd_keys, sd_offsets, layer_shapes,
                    cmd_pipe, barrier):
    """
    Owns envs_per_worker envs. Reads policy weights from shm (weights_true/adv),
    runs full T-step rollout, writes results into shm rollout buffers, hits barrier.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch as _torch
    import torch.nn as _nn
    from torch.distributions import Categorical as _Cat
    from multiprocessing.shared_memory import SharedMemory

    shms   = {k: SharedMemory(name=shm_names[k]) for k in shm_names}
    arrays = {k: np.ndarray(shapes[k], dtype=dtypes[k], buffer=shms[k].buf)
              for k in shm_names}

    Ew = envs_per_worker
    sl = slice(worker_id * Ew, (worker_id + 1) * Ew)

    obs_dim = shapes["obs_buf"][3]
    A       = shapes["obs_buf"][2]
    hidden  = 64
    act_dim = None  # received per-rollout in the message

    def _load_weights(model, flat, keys, offsets, lshapes):
        """Copy weights from flat shm array into model params in-place."""
        sd = model.state_dict()
        for key, (start, end), shape in zip(keys, offsets, lshapes):
            sd[key].copy_(_torch.from_numpy(flat[start:end]).view(shape))
        model.load_state_dict(sd)

    def _build_policy(obs_dim, act_dim):
        def _mlp(out_dim, gain):
            net = _nn.Sequential(
                _nn.Linear(obs_dim, hidden), _nn.Tanh(),
                _nn.Linear(hidden, hidden),  _nn.Tanh(),
                _nn.Linear(hidden, out_dim),
            )
            return net
        class _P(_nn.Module):
            def __init__(self):
                super().__init__()
                self.actor  = _mlp(act_dim, 0.01)
                self.critic = _mlp(1, 1.0)
            def forward(self, x):
                return self.actor(x), self.critic(x).squeeze(-1)
            def act_val(self, x, action=None):
                logits, val = self.forward(x)
                dist = _Cat(logits=logits)
                if action is None: action = dist.sample()
                return action, dist.log_prob(action), val, dist
        return _P()

    true_pol = None
    adv_pol  = None

    try:
        envs    = [env_fn_wrapper.x() for _ in range(Ew)]
        cur_obs = np.stack([env.reset() for env in envs])   # (Ew, A, obs_dim)

        while True:
            msg = cmd_pipe.recv()
            if msg[0] == "close":
                break

            elif msg[0] == "reset":
                cur_obs = np.stack([env.reset() for env in envs])
                arrays["obs_init"][sl] = cur_obs
                barrier.wait()

            elif msg[0] in ("rollout_vanilla", "rollout_robust"):
                vanilla = (msg[0] == "rollout_vanilla")
                if vanilla:
                    _, act_dim_v, T, gamma, lam = msg
                else:
                    _, act_dim_v, is_true_ea, T, gamma, lam = msg

                # Build policies lazily — check each independently
                if true_pol is None:
                    true_pol = _build_policy(obs_dim, act_dim_v)
                if not vanilla and adv_pol is None:
                    adv_pol = _build_policy(obs_dim, act_dim_v)

                # Load weights from shm — zero IPC cost, no pickle
                _load_weights(true_pol, arrays["weights_true"], sd_keys, sd_offsets, layer_shapes)
                true_pol.eval()
                if not vanilla:
                    _load_weights(adv_pol, arrays["weights_adv"], sd_keys, sd_offsets, layer_shapes)
                    adv_pol.eval()

                Ew_A = Ew * A

                if not vanilla:
                    is_true_flat = _torch.from_numpy(is_true_ea.reshape(Ew_A))
                    arrays["is_true_buf"][:, sl] = is_true_ea[np.newaxis]

                # Pre-allocate step buffers — reused every step
                new_obs  = np.empty((Ew, A, obs_dim), dtype=np.float32)
                new_rew  = np.empty((Ew, A),           dtype=np.float32)
                new_done = np.empty(Ew,                dtype=bool)
                obs_t    = _torch.empty(Ew_A, obs_dim)

                for t in range(T):
                    obs_t.copy_(_torch.from_numpy(cur_obs.reshape(Ew_A, obs_dim)))

                    with _torch.no_grad():
                        if vanilla:
                            acts, lps, vals, _ = true_pol.act_val(obs_t)
                        else:
                            acts_t, lps_t, vals_t, _ = true_pol.act_val(obs_t)
                            acts_a, lps_a, vals_a, _ = adv_pol.act_val(obs_t)
                            acts = _torch.where(is_true_flat, acts_t, acts_a)
                            lps  = _torch.where(is_true_flat, lps_t,  lps_a)
                            vals = _torch.where(is_true_flat, vals_t, vals_a)

                    acts_np = acts.numpy().reshape(Ew, A)

                    arrays["obs_buf"][t, sl] = cur_obs
                    arrays["act_buf"][t, sl] = acts_np
                    arrays["lp_buf"] [t, sl] = lps.numpy().reshape(Ew, A)
                    arrays["val_buf"][t, sl] = vals.numpy().reshape(Ew, A)

                    for i, (env, a) in enumerate(zip(envs, acts_np)):
                        o, r, done, _, _ = env.step(a)
                        if done:
                            o = env.reset()
                        new_obs[i]  = o
                        new_rew[i]  = r
                        new_done[i] = done

                    arrays["rew_buf"] [t, sl] = new_rew
                    arrays["done_buf"][t, sl] = new_done
                    cur_obs[:] = new_obs

                arrays["obs_init"][sl] = cur_obs
                barrier.wait()

    finally:
        for env in envs:
            env.close()
        for shm in shms.values():
            shm.close()


class ParallelEnv:
    """
    Rollout-in-worker vectorised env.
    Workers run the full T-step rollout locally and fill shared-memory buffers.
    Main process sends policy weights once per update; IPC = 1 round-trip.
    """

    def __init__(self, total_envs: int, num_workers: int = NUM_WORKERS,
                 env_name: str = GLOBAL_ENV_NAME, pickup_shaping: float = 0.01):
        from multiprocessing.shared_memory import SharedMemory

        global GLOBAL_ENV_NAME, GLOBAL_PICKUP_SHAPING
        GLOBAL_ENV_NAME       = env_name
        GLOBAL_PICKUP_SHAPING = pickup_shaping
        assert total_envs % num_workers == 0

        self.total_envs      = total_envs
        self.num_workers     = num_workers
        self.envs_per_worker = total_envs // num_workers

        _dummy = RWAREWrapper(env_name, pickup_shaping)
        self.obs_dim  = _dummy.obs_dim
        self.n_agents = _dummy.num_agents
        self.act_dim  = _dummy.action_dim
        _dummy.close()

        E, A, D = total_envs, self.n_agents, self.obs_dim
        self._T       = None
        self._shms    = {}
        self._arrays  = {}
        self._E, self._A, self._D = E, A, D

        # Compute total parameter count for a SharedPolicy so we can
        # allocate shm for weights — avoids pickling state_dict every update
        hidden = 64
        def _param_count(in_d, out_d):
            # 2-layer MLP: in->hidden->hidden->out
            return (in_d*hidden+hidden + hidden*hidden+hidden + hidden*out_d+out_d)
        p_actor  = _param_count(D, self.act_dim)
        p_critic = _param_count(D, 1)
        self._n_params = p_actor + p_critic

        # Build param offset map matching SharedPolicy layer order:
        # actor: [0.w, 0.b, 2.w, 2.b, 4.w, 4.b]  critic: same
        def _offsets(in_d, out_d):
            sizes = [in_d*hidden, hidden, hidden*hidden, hidden, hidden*out_d, out_d]
            offs  = []
            pos   = 0
            for s in sizes:
                offs.append((pos, pos+s))
                pos += s
            return offs
        self._actor_offsets  = _offsets(D, self.act_dim)
        self._critic_offsets = _offsets(D, 1)
        # Key order must match SharedPolicy state_dict order
        self._sd_keys = (
            ["actor.0.weight","actor.0.bias","actor.2.weight","actor.2.bias",
             "actor.4.weight","actor.4.bias"] +
            ["critic.0.weight","critic.0.bias","critic.2.weight","critic.2.bias",
             "critic.4.weight","critic.4.bias"]
        )
        self._sd_offsets = []
        base = 0
        for o in self._actor_offsets + self._critic_offsets:
            size = o[1] - o[0]
            self._sd_offsets.append((base, base+size))
            base += size

        # Static shm: obs_init + weights for up to 2 policies (true + adv)
        self._shm_static_specs = {
            "obs_init":    ((E, A, D),             np.float32),
            "weights_true": ((self._n_params,),     np.float32),
            "weights_adv":  ((self._n_params,),     np.float32),
        }
        self._allocate_shm(self._shm_static_specs)

        self._barrier = mp.Barrier(num_workers + 1)
        self._pipes   = []
        self._procs   = []

        # We'll pass shm_names/shapes/dtypes when we know T
        # Workers are spawned lazily on first collect_rollout call
        self._workers_ready = False
        self._num_workers   = num_workers

    def _allocate_shm(self, specs):
        from multiprocessing.shared_memory import SharedMemory
        for k, (shape, dtype) in specs.items():
            if k in self._shms:
                try:
                    self._shms[k].close()
                except Exception:
                    pass
                try:
                    self._shms[k].unlink()
                except Exception:
                    pass
            nbytes = max(int(np.prod(shape)) * np.dtype(dtype).itemsize, 1)
            shm    = SharedMemory(create=True, size=nbytes)
            arr    = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            self._shms[k]   = shm
            self._arrays[k] = arr

    def _spawn_workers(self, T, has_is_true=False):
        E, A, D = self._E, self._A, self._D
        rollout_specs = {
            "obs_buf":  ((T, E, A, D), np.float32),
            "act_buf":  ((T, E, A),    np.int32),
            "lp_buf":   ((T, E, A),    np.float32),
            "val_buf":  ((T, E, A),    np.float32),
            "rew_buf":  ((T, E, A),    np.float32),
            "done_buf": ((T, E),       bool),
        }
        if has_is_true:
            rollout_specs["is_true_buf"] = ((T, E, A), bool)

        self._allocate_shm(rollout_specs)
        self._T = T

        all_specs = {**self._shm_static_specs, **rollout_specs}
        shm_names = {k: self._shms[k].name   for k in all_specs}
        shapes    = {k: all_specs[k][0]       for k in all_specs}
        dtypes    = {k: all_specs[k][1]       for k in all_specs}

        # Build layer shape list matching sd_keys order for workers
        hid = 64
        actor_shapes  = [(hid, D), (hid,), (hid,hid), (hid,), (self.act_dim, hid), (self.act_dim,)]
        critic_shapes = [(hid, D), (hid,), (hid,hid), (hid,), (1,       hid), (1,)]
        layer_shapes  = actor_shapes + critic_shapes

        for wid in range(self._num_workers):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=_rollout_worker,
                args=(wid, _CloudpickleWrapper(_make_env), self.envs_per_worker,
                      shm_names, shapes, dtypes,
                      self._sd_keys, self._sd_offsets, layer_shapes,
                      child_conn, self._barrier),
                daemon=True,
            )
            p.start()
            child_conn.close()
            self._pipes.append(parent_conn)
            self._procs.append(p)

        # Initial reset
        for pipe in self._pipes:
            pipe.send(("reset",))
        self._barrier.wait()
        self._workers_ready = True

    def _pack_weights(self, sd, target_array):
        """Flatten state_dict tensors into a pre-allocated shm float32 array."""
        for key, (start, end) in zip(self._sd_keys, self._sd_offsets):
            target_array[start:end] = sd[key].numpy().ravel()

    def _stop_workers(self):
        """Kill worker processes only — leaves shm intact for reuse."""
        for pipe in self._pipes:
            try:
                pipe.send(("close",))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=3.0)
        self._pipes = []
        self._procs = []
        self._workers_ready = False

    def collect_rollout(self, policy_sd, T, gamma, lam, act_dim):
        if not self._workers_ready or self._T != T:
            if self._workers_ready:
                self._stop_workers()
            self._spawn_workers(T, has_is_true=False)

        self._pack_weights(policy_sd, self._arrays["weights_true"])
        for pipe in self._pipes:
            pipe.send(("rollout_vanilla", act_dim, T, gamma, lam))
        self._barrier.wait()

        return {
            "obs":      torch.as_tensor(self._arrays["obs_buf"]),
            "act":      torch.as_tensor(self._arrays["act_buf"]),
            "lp":       torch.as_tensor(self._arrays["lp_buf"]),
            "val":      torch.as_tensor(self._arrays["val_buf"]),
            "rew":      torch.as_tensor(self._arrays["rew_buf"]),
            "done":     torch.as_tensor(self._arrays["done_buf"].astype(np.float32)),
            "obs_next": torch.as_tensor(self._arrays["obs_init"]),
        }

    def collect_rollout_robust(self, true_sd, adv_sd, is_true_ea, T, gamma, lam, act_dim):
        if not self._workers_ready or self._T != T or "is_true_buf" not in self._arrays:
            if self._workers_ready:
                self._stop_workers()
            self._spawn_workers(T, has_is_true=True)

        self._pack_weights(true_sd, self._arrays["weights_true"])
        self._pack_weights(adv_sd,  self._arrays["weights_adv"])

        Ew = self.envs_per_worker
        for wid, pipe in enumerate(self._pipes):
            sl = slice(wid * Ew, (wid + 1) * Ew)
            pipe.send(("rollout_robust", act_dim, is_true_ea[sl], T, gamma, lam))
        self._barrier.wait()

        return {
            "obs":      torch.as_tensor(self._arrays["obs_buf"]),
            "act":      torch.as_tensor(self._arrays["act_buf"]),
            "lp":       torch.as_tensor(self._arrays["lp_buf"]),
            "val":      torch.as_tensor(self._arrays["val_buf"]),
            "rew":      torch.as_tensor(self._arrays["rew_buf"]),
            "done":     torch.as_tensor(self._arrays["done_buf"].astype(np.float32)),
            "obs_next": torch.as_tensor(self._arrays["obs_init"]),
            "is_true":  torch.as_tensor(self._arrays["is_true_buf"]),
        }

    # Keep a simple step() for evaluate_policy compatibility
    def reset(self):
        if not self._workers_ready:
            self._spawn_workers(1, has_is_true=False)
        for pipe in self._pipes:
            pipe.send(("reset",))
        self._barrier.wait()
        return self._arrays["obs_init"].copy()   # (E, A, obs_dim)

    def step(self, actions: np.ndarray):
        raise NotImplementedError(
            "ParallelEnv.step() removed — use collect_rollout() for training. "
            "For eval, use a separate EvalEnv.")

    def close(self):
        """Full teardown — stop workers and release all shm."""
        self._stop_workers()
        for shm in self._shms.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# EvalEnv — lightweight single-step env for evaluate_policy
# ---------------------------------------------------------------------------
class EvalEnv:
    """Simple non-parallel env wrapper for evaluation — no shared memory needed."""

    def __init__(self, n_envs: int, env_name: str = GLOBAL_ENV_NAME,
                 pickup_shaping: float = 0.01):
        self.envs = [RWAREWrapper(env_name, pickup_shaping) for _ in range(n_envs)]
        self.n_envs    = n_envs
        self.n_agents  = self.envs[0].num_agents
        self.obs_dim   = self.envs[0].obs_dim

    def reset(self):
        return np.stack([e.reset() for e in self.envs])   # (E, A, obs_dim)

    def step(self, actions: np.ndarray):
        """actions: (E, A) int"""
        obs_l, rew_l, done_l = [], [], []
        for e, env in enumerate(self.envs):
            o, r, done, _, _ = env.step(actions[e])
            if done:
                o = env.reset()
            obs_l.append(o); rew_l.append(r); done_l.append(done)
        return (np.stack(obs_l), np.stack(rew_l),
                np.array(done_l, dtype=bool),
                np.array(done_l, dtype=bool),
                np.array(done_l, dtype=bool))

    def close(self):
        for e in self.envs:
            e.close()



# ---------------------------------------------------------------------------
# Policy: completely separate actor and critic MLPs
# ---------------------------------------------------------------------------
class SharedPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        def _mlp(out_dim, out_gain):
            layers = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, out_dim),
            )
            for m in layers.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
            nn.init.orthogonal_(layers[-1].weight, gain=out_gain)
            return layers

        self.actor  = _mlp(action_dim, out_gain=0.01)
        self.critic = _mlp(1,          out_gain=1.0)

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, dist

    @torch.no_grad()
    def act(self, obs, greedy=False):
        logits = self.actor(obs)
        return logits.argmax(-1) if greedy else Categorical(logits=logits).sample()


# ---------------------------------------------------------------------------
# GAE — flat, always bootstraps
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """
    Fully vectorised GAE — no Python loop over T.
    Works by scanning backwards using torch ops only.
    rewards, values, dones : (T, N)
    next_value             : (N,)
    """
    T      = rewards.shape[0]
    gl     = gamma * lam
    masks  = 1.0 - dones                              # (T, N)

    # Build next-values: shift values by 1, plug next_value at the end
    # next_vals[t] = values[t+1] for t<T-1, next_value for t=T-1
    next_vals = torch.empty_like(values)
    next_vals[:-1] = values[1:]
    next_vals[-1]  = next_value

    deltas = rewards + gamma * next_vals * masks - values  # (T, N)

    # Backwards scan: adv[t] = delta[t] + gl*mask[t]*adv[t+1]
    # Unroll with cumulative product trick:
    #   adv[t] = sum_{k=0}^{T-1-t} (gl*mask)^k * delta[t+k]
    # Implemented as a reverse cumsum with decaying weights.
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_value)
    for t in range(T - 1, -1, -1):
        gae    = deltas[t] + gl * masks[t] * gae
        adv[t] = gae
    return adv, adv + values


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_policy(eval_env: "ParallelEnv", policies, n_agents: int = None,
                    device=None, n_steps: int = 200):
    """
    Evaluate using the rollout workers — fully parallel, no serial env loop.

    eval_env must be a ParallelEnv. n_agents is ignored if eval_env is a
    ParallelEnv (agent count comes from the env itself).

    policies : SharedPolicy | str | dict | list thereof
               Single policy = self-play. List = cross-play.
    """
    if device is None:
        device = torch.device("cpu")

    # Always read agent count from the actual env — ignores n_agents arg
    n_agents = eval_env.n_agents
    def _load(p):
        if isinstance(p, torch.nn.Module):
            return p.to(device)
        if isinstance(p, dict):
            # Raw state dict snapshot from AsyncEvaluator — need obs/act dim from eval_env
            pol = SharedPolicy(eval_env.obs_dim, eval_env.act_dim).to(device)
            pol.load_state_dict(_strip_compiled(p))
            return pol
        ckpt = torch.load(p, map_location=device, weights_only=False)
        key  = "true_policy" if "true_policy" in ckpt else "policy"
        pol  = SharedPolicy(ckpt["obs_dim"], ckpt["act_dim"]).to(device)
        pol.load_state_dict(_strip_compiled(ckpt[key]))
        return pol

    def _is_single(p):
        return isinstance(p, dict) or isinstance(p, (SharedPolicy, str)) or (
            isinstance(p, torch.nn.Module) and not isinstance(p, list))

    if _is_single(policies):
        policy_list = [_load(policies)] * n_agents
    else:
        loaded = [_load(p) for p in policies]
        assert len(loaded) == n_agents, f"Expected {n_agents} policies, got {len(loaded)}"
        policy_list = loaded


    for p in set(id(p) for p in policy_list):
        pass  # policies already on device from _load

    E   = eval_env.total_envs
    act_dim = eval_env.act_dim

    same_policy = len(set(id(p) for p in policy_list)) == 1
    eval_env.reset()
    if same_policy:
        # Self-play: single collect_rollout with the shared policy
        pol = policy_list[0]
        pol.eval()
        sd  = _strip_compiled(pol.state_dict())
        buf = eval_env.collect_rollout(sd, n_steps, gamma=0.99, lam=0.95, act_dim=act_dim)
        # rew: (T, E, A) — sum over steps and agents, mean over envs
        rew = buf["rew"]   # (T, E, A)
    else:
        # Cross-play: two distinct policies — use collect_rollout_robust.
        # Assign first unique policy as "true", second as "adv".
        # For >2 unique policies, assign slots by majority.
        unique_pols = list(dict.fromkeys(id(p) for p in policy_list))
        pol_by_id   = {id(p): p for p in policy_list}

        true_pol = pol_by_id[unique_pols[0]]
        adv_pol  = pol_by_id[unique_pols[-1]]  # last unique = adversary
        true_pol.eval(); adv_pol.eval()

        true_sd = _strip_compiled(true_pol.state_dict())
        adv_sd  = _strip_compiled(adv_pol.state_dict())

        # is_true_ea: agent a is "true" if policy_list[a] is true_pol
        true_id = unique_pols[0]
        is_true_row = np.array([id(policy_list[a]) == true_id for a in range(n_agents)], dtype=bool)
        is_true_ea  = np.tile(is_true_row, (E, 1))   # (E, A)

        buf = eval_env.collect_rollout_robust(
            true_sd, adv_sd, is_true_ea, n_steps, gamma=0.99, lam=0.95, act_dim=act_dim
        )
        rew = buf["rew"]   # (T, E, A)

    # rew shape: (T, E, A)
    total_rew = rew.sum(dim=0).sum(dim=-1)          # (E,) — sum over T and agents
    total_del = rew.round().sum(dim=0).sum(dim=-1)  # (E,) — deliveries (reward≈1 per delivery)
    delivery_rate=np.cumsum(rew.round(),axis=0).sum(dim=-1).mean(axis=1)
    return float(total_rew.mean()), float(total_del.mean()), delivery_rate/np.linspace(1,n_steps,n_steps)


# ---------------------------------------------------------------------------
# Async evaluator — runs eval in a background thread so training never blocks
# ---------------------------------------------------------------------------
class AsyncEvaluator:
    """
    Wraps evaluate_policy in a daemon thread so training never blocks.
    evaluate_policy now delegates to collect_rollout, so the thread just
    triggers a worker rollout and reads the reward buffer — no policy
    inference in the thread, no deepcopy needed.

    submit() snapshots weights, launches background eval.
    result() returns last completed (score, deliveries) instantly.
    """
    def __init__(self, eval_env, policy_ref, n_agents, device, n_steps=200):
        import threading
        self._eval_env   = eval_env
        self._policy_ref = policy_ref
        self._n_agents   = n_agents
        self._device     = device
        self._n_steps    = n_steps
        self._thread     = None
        self._last       = (0.0, 0.0, None)
        self._lock       = threading.Lock()
        self._threading  = threading

    def submit(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            # Snapshot weights before the next update overwrites them
            sd = _strip_compiled({k: v.cpu().clone() for k, v in self._policy_ref.state_dict().items()})
        def _run():
            result = evaluate_policy(self._eval_env, sd, self._n_agents,
                                     self._device, self._n_steps)
            with self._lock:
                self._last = result
        t = self._threading.Thread(target=_run, daemon=True)
        t.start()
        with self._lock:
            self._thread = t

    def result(self):
        with self._lock:
            return self._last

    def wait(self):
        with self._lock:
            t = self._thread
        if t is not None:
            t.join()



def _strip_compiled(sd):
    """torch.compile prefixes all keys with '_orig_mod.'. Strip it so the
    plain worker-side module can load the state dict without errors."""
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in sd):
        return {k[len(prefix):] if k.startswith(prefix) else k: v
                for k, v in sd.items()}
    return sd


# ---------------------------------------------------------------------------
# Vanilla IPPO
# ---------------------------------------------------------------------------
def train_vanilla_ippo(
    env_name:       str   = "rware-tiny-4ag-easy-v2",
    policy                = None,
    steps:          int   = 5_000_000,
    lr:             float = 6e-4,
    entropy_coef:   float = 0.1,
    num_envs:       int   = 32,
    num_steps:      int   = 500,
    k_epochs:       int   = 4,
    minibatch_size: int   = 2048,
    device                = None,
):
    if device is None:
        device = torch.device("cpu")
    if rewards_log is None:
        rewards_log = []

    print(f"=== Vanilla IPPO | {env_name} ===")

    env      = ParallelEnv(num_envs, NUM_WORKERS, env_name, pickup_shaping)
    eval_env = ParallelEnv(min(num_envs, 8), NUM_WORKERS, env_name, pickup_shaping)
    dummy    = RWAREWrapper(env_name)
    obs_dim, act_dim, n_agents = dummy.obs_dim, dummy.action_dim, dummy.num_agents
    dummy.close()
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  n_agents={n_agents}")

    if policy is None:
        policy = SharedPolicy(obs_dim, act_dim, hidden_dim).to(device)
    opt = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    policy = torch.compile(policy)

    T, E, A = num_steps, num_envs, n_agents
    N       = E * A
    total_updates = steps // (E * T) + 1
    start = time.time()

    # Async evaluator — eval runs in a background thread, never blocks training
    aeval = AsyncEvaluator(eval_env, policy, n_agents, device)

    try:
        for update in range(1, total_updates + 1):

            # ── workers run full T-step rollout, return filled buffers ────
            buf = env.collect_rollout(
                _strip_compiled(policy.state_dict()), T, gamma, gae_lambda, act_dim
            )

            obs_flat  = buf["obs"].reshape(T, N, obs_dim).to(device)
            act_flat  = buf["act"].reshape(T, N).to(device=device, dtype=torch.long)
            lp_flat   = buf["lp"].reshape(T, N).to(device)
            val_flat  = buf["val"].reshape(T, N).to(device)
            rew_flat  = buf["rew"].reshape(T, N).to(device)
            done_flat = buf["done"].unsqueeze(-1).expand(T, E, A).reshape(T, N).to(device)

            with torch.no_grad():
                obs_next = buf["obs_next"].reshape(N, obs_dim).to(device)
                _, next_val, _, _, _ = policy.get_action_and_value(obs_next)

            adv, ret = compute_gae(rew_flat, val_flat, next_val, done_flat, gamma, gae_lambda)
            adv_flat2 = adv.view(-1)
            ret_flat2 = ret.view(-1)
            adv_flat2 = (adv_flat2 - adv_flat2.mean()) / (adv_flat2.std() + 1e-8)

            obs_2d  = obs_flat.view(-1, obs_dim)
            act_1d  = act_flat.view(-1)
            lp_1d   = lp_flat.view(-1)

            policy.train()
            B = obs_2d.shape[0]
            for _ in range(k_epochs):
                for mb in torch.randperm(B, device=device).split(minibatch_size):
                    _, new_lp, ent, new_val, _ = policy.get_action_and_value(
                        obs_2d[mb], act_1d[mb]
                    )
                    ratio = torch.exp(new_lp - lp_1d[mb])
                    loss  = (
                        -torch.min(ratio * adv_flat2[mb],
                                   torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_flat2[mb]).mean()
                        + 0.5 * ((new_val - ret_flat2[mb]) ** 2).mean()
                        - entropy_coef * ent.mean()
                    )
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    opt.step()

            if update % 10 == 0:
                # Submit eval to background thread (non-blocking)
                aeval.submit()
                # Read last completed result (instant)
                score, deliveries, _ = aeval.result()
                elapsed = time.time() - start
                sps     = update * E * T / elapsed
                print(f"Update {update:5d} | Score {score:8.3f} | Deliveries {deliveries:6.2f} | "
                      f"Time {elapsed:6.0f}s | SPS {sps:.0f}")
                rewards_log.append((score, deliveries))

            if update % 500 == 0:
                _save_vanilla(filename, policy, rewards_log, obs_dim, act_dim, n_agents)

    finally:
        aeval.wait()
        env.close()
        eval_env.close()
        _save_vanilla(filename, policy, rewards_log, obs_dim, act_dim, n_agents)

    return policy, np.array(rewards_log)


def _save_vanilla(filename, policy, rewards_log, obs_dim, act_dim, n_agents):
    torch.save({
        "policy":   policy.state_dict(),
        "rewards":  rewards_log,
        "obs_dim":  obs_dim,
        "act_dim":  act_dim,
        "n_agents": n_agents,
    }, filename)


# ---------------------------------------------------------------------------
# Risk-Averse IPPO
# ---------------------------------------------------------------------------
def train_risk_averse_ippo(
    env_name:       str   = "rware-tiny-4ag-easy-v2",
    m:              int   = 2,
    true_policy           = None,
    adv_policy            = None,
    steps:          int   = 5_000_000,
    lr:             float = 6e-4,
    entropy_coef:   float = 0.1,
    risk_factor:    float = 0.5,
    num_envs:       int   = 32,
    num_steps:      int   = 500,
    k_epochs:       int   = 4,
    minibatch_size: int   = 2048,
    hidden_dim:     int   = 64,
    gamma:          float = 0.99,
    gae_lambda:     float = 0.95,
    clip_eps:       float = 0.2,
    pickup_shaping: float = 0.01,
    rewards_log           = None,
    filename:       str   = "rware_robust.pt",
    device                = None,
):
    """
    m cooperative agents share true_policy; (n-m) adversarial share adv_policy.
    Slot assignment randomised per rollout.
    Adversarial objective: maximize −team_reward − risk_factor · KL(adv ∥ true)
    Evaluation uses all-true team (deployment scenario).
    """
    if device is None:
        device = torch.device("cpu")
    if rewards_log is None:
        rewards_log = []

    print(f"=== Risk-Averse IPPO | {env_name} | m={m} | risk={risk_factor} ===")

    env      = ParallelEnv(num_envs, NUM_WORKERS, env_name, pickup_shaping)
    eval_env = ParallelEnv(min(num_envs, 8), NUM_WORKERS, env_name, pickup_shaping)
    dummy    = RWAREWrapper(env_name)
    obs_dim, act_dim, n_agents = dummy.obs_dim, dummy.action_dim, dummy.num_agents
    dummy.close()

    assert n_agents - m > 0, "need at least one adversarial agent"
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  "
          f"n_agents={n_agents}  coop={m}  adv={n_agents - m}")

    if true_policy is None:
        true_policy = SharedPolicy(obs_dim, act_dim, hidden_dim)
    if adv_policy is None:
        adv_policy  = copy.deepcopy(true_policy)
    true_policy = true_policy.to(device)
    adv_policy  = adv_policy.to(device)

    true_policy = torch.compile(true_policy)
    adv_policy  = torch.compile(adv_policy)
    true_opt = optim.Adam(true_policy.parameters(), lr=lr, eps=1e-5)
    adv_opt  = optim.Adam(adv_policy.parameters(),  lr=lr, eps=1e-5)

    T, E, A = num_steps, num_envs, n_agents
    N       = E * A
    total_updates = steps // (E * T) + 1
    start = time.time()

    aeval = AsyncEvaluator(eval_env, true_policy, n_agents, device)

    try:
        for update in range(1, total_updates + 1):

            # Random slot assignment for this rollout
            is_true_ea = np.zeros((E, A), dtype=bool)
            for e in range(E):
                is_true_ea[e, np.random.choice(A, size=m, replace=False)] = True

            # workers run full T-step rollout
            buf = env.collect_rollout_robust(
                _strip_compiled(true_policy.state_dict()),
                _strip_compiled(adv_policy.state_dict()),
                is_true_ea, T, gamma, gae_lambda, act_dim
            )

            obs_flat    = buf["obs"].reshape(T, N, obs_dim).to(device)
            act_flat    = buf["act"].reshape(T, N).to(device=device, dtype=torch.long)
            lp_flat     = buf["lp"].reshape(T, N).to(device)
            val_flat    = buf["val"].reshape(T, N).to(device)
            rew_flat    = buf["rew"].reshape(T, N).to(device)
            done_flat   = buf["done"].unsqueeze(-1).expand(T, E, A).reshape(T, N).to(device)
            is_true_all = buf["is_true"].reshape(T, N).to(device)

            # Bootstrap
            with torch.no_grad():
                obs_next     = buf["obs_next"].reshape(N, obs_dim).to(device)
                is_true_last = torch.as_tensor(is_true_ea.reshape(N), device=device)
                _, tv, _, _, _ = true_policy.get_action_and_value(obs_next)
                _, av, _, _, _ = adv_policy.get_action_and_value(obs_next)
                next_val = torch.where(is_true_last, tv, av)

            adv_gae, ret_arr = compute_gae(rew_flat, val_flat, next_val, done_flat, gamma, gae_lambda)
            adv_gae_f = adv_gae.view(-1)
            ret_arr_f = ret_arr.view(-1)

            obs_2d     = obs_flat.view(-1, obs_dim)
            act_1d     = act_flat.view(-1)
            lp_1d      = lp_flat.view(-1)
            is_true_1d = is_true_all.view(-1)

            # true policy update
            true_idx = is_true_1d.nonzero(as_tuple=True)[0]
            obs_t = obs_2d[true_idx]; act_t = act_1d[true_idx]
            lp_t  = lp_1d[true_idx];  adv_t = adv_gae_f[true_idx]
            ret_t = ret_arr_f[true_idx]
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            true_policy.train()
            for _ in range(k_epochs):
                for mb in torch.randperm(len(true_idx), device=device).split(minibatch_size):
                    _, nlp, ent, nval, _ = true_policy.get_action_and_value(obs_t[mb], act_t[mb])
                    ratio = torch.exp(nlp - lp_t[mb])
                    loss  = (
                        -torch.min(ratio * adv_t[mb],
                                   torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[mb]).mean()
                        + 0.5 * ((nval - ret_t[mb]) ** 2).mean()
                        - entropy_coef * ent.mean()
                    )
                    true_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(true_policy.parameters(), 0.5)
                    true_opt.step()

            # adversarial policy update
            adv_idx = (~is_true_1d).nonzero(as_tuple=True)[0]
            obs_a = obs_2d[adv_idx]; act_a = act_1d[adv_idx]
            lp_a  = lp_1d[adv_idx]
            adv_r = -(adv_gae_f[adv_idx])
            ret_a = -(ret_arr_f[adv_idx])
            adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)

            adv_policy.train()
            for _ in range(k_epochs):
                for mb in torch.randperm(len(adv_idx), device=device).split(minibatch_size):
                    _, nlp, ent, nval, curr_dist = adv_policy.get_action_and_value(
                        obs_a[mb], act_a[mb]
                    )
                    with torch.no_grad():
                        _, _, _, _, ref_dist = true_policy.get_action_and_value(obs_a[mb])
                    kl    = torch.distributions.kl_divergence(curr_dist, ref_dist).mean()
                    ratio = torch.exp(nlp - lp_a[mb])
                    loss  = (
                        -torch.min(ratio * adv_r[mb],
                                   torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_r[mb]).mean()
                        + 0.5 * ((nval - ret_a[mb]) ** 2).mean()
                        - entropy_coef * ent.mean()
                        + risk_factor * kl
                    )
                    adv_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(adv_policy.parameters(), 0.5)
                    adv_opt.step()

            if update % 10 == 0:
                aeval.submit()
                score, deliveries, _ = aeval.result()
                elapsed = time.time() - start
                print(f"Update {update:5d} | Score {score:8.3f} | Deliveries {deliveries:6.2f} | Time {elapsed:6.0f}s")
                rewards_log.append((score, deliveries))

            if update % 500 == 0:
                _save_robust(filename, true_policy, adv_policy, rewards_log,
                             obs_dim, act_dim, n_agents, m, risk_factor)

    finally:
        aeval.wait()
        env.close()
        eval_env.close()
        _save_robust(filename, true_policy, adv_policy, rewards_log,
                     obs_dim, act_dim, n_agents, m, risk_factor)

    return true_policy, adv_policy, np.array(rewards_log)



def _save_robust(filename, true_policy, adv_policy, rewards_log,
                 obs_dim, act_dim, n_agents, m, risk_factor):
    torch.save({
        "true_policy": true_policy.state_dict(),
        "adv_policy":  adv_policy.state_dict(),
        "rewards":     rewards_log,
        "obs_dim":     obs_dim,
        "act_dim":     act_dim,
        "n_agents":    n_agents,
        "m":           m,
        "risk_factor": risk_factor,
    }, filename)


# ---------------------------------------------------------------------------
# Checkpoint loaders
# ---------------------------------------------------------------------------
def load_vanilla(filename: str, device=None):
    if device is None:
        device = torch.device("cpu")
    ckpt   = torch.load(filename, map_location=device, weights_only=False)
    policy = SharedPolicy(ckpt["obs_dim"], ckpt["act_dim"]).to(device)
    policy.load_state_dict(_strip_compiled(ckpt["policy"]))
    return policy, ckpt.get("rewards", [])


def load_robust(filename: str, device=None):
    if device is None:
        device = torch.device("cpu")
    ckpt   = torch.load(filename, map_location=device, weights_only=False)
    true_p = SharedPolicy(ckpt["obs_dim"], ckpt["act_dim"]).to(device)
    adv_p  = SharedPolicy(ckpt["obs_dim"], ckpt["act_dim"]).to(device)
    true_p.load_state_dict(_strip_compiled(ckpt["true_policy"]))
    adv_p.load_state_dict(_strip_compiled(ckpt["adv_policy"]))
    return true_p, adv_p, ckpt.get("rewards", [])


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------
ACTION_NAMES = ["NOOP", "FORWARD", "LEFT", "RIGHT", "TOGGLE"]


# @torch.no_grad()
# def visualize_policy(
#     policies,
#     env_name:   str   = "rware-tiny-4ag-easy-v2",
#     episodes:   int   = 3,
#     max_steps:  int   = 500,
#     delay:      float = 0.08,
#     stochastic: bool  = True,
#     device            = None,
# ):
#     """
#     policies : SharedPolicy | str | list[SharedPolicy | str]
#                Mix freely for cross-play. Strings are checkpoint paths.
#     """
#     if device is None:
#         device = torch.device("cpu")

#     def _load(p):
#         if isinstance(p, torch.nn.Module):
#             return p.to(device)
#         ckpt = torch.load(p, map_location=device, weights_only=False)
#         key  = "true_policy" if "true_policy" in ckpt else "policy"
#         pol  = SharedPolicy(ckpt["obs_dim"], ckpt["act_dim"]).to(device)
#         pol.load_state_dict(_strip_compiled(ckpt[key]))
#         if ckpt.get("rewards"):
#             r  = ckpt["rewards"]
#             sc = [x[0] if isinstance(x, (list, tuple)) else x for x in r]
#             print(f"  [{p}] final={sc[-1]:.3f}  best={max(sc):.3f}  over {len(sc)*10} updates")
#         return pol

#     env_probe = gym.make(env_name)
#     n_agents  = env_probe.unwrapped.n_agents
#     env_probe.close()

#     if isinstance(policies, (SharedPolicy, str)) or (
#             isinstance(policies, torch.nn.Module) and not isinstance(policies, list)):
#         pol         = _load(policies)
#         policy_list = [pol] * n_agents
#     else:
#         policy_list = [_load(p) for p in policies]
#         assert len(policy_list) == n_agents, f"Expected {n_agents} policies, got {len(policy_list)}"

#     for p in policy_list:
#         p.eval()

#     cross_play = len(set(id(p) for p in policy_list)) > 1
#     print(f"Visualizing '{env_name}' | {n_agents} agents | "
#           f"{'cross-play' if cross_play else 'self-play'} | "
#           f"{'stochastic' if stochastic else 'greedy'} | "
#           f"{episodes} ep | delay={delay}s")

#     try:
#         import rware.rendering as _rr
#         _orig = _rr.Viewer._draw_shelfs
#         def _safe(self, env):
#             s = getattr(env, "shelfs", None) or getattr(env, "shelves", None)
#             if s:
#                 _orig(self, env)
#         _rr.Viewer._draw_shelfs = _safe
#     except Exception:
#         pass

#     env = gym.make(env_name, render_mode="human")

#     ep_returns = []
#     try:
#         for ep in range(1, episodes + 1):
#             obs_tuple, _ = env.reset()
#             obs    = torch.as_tensor(
#                 process_obs_batch(obs_tuple), dtype=torch.float32, device=device
#             )
#             ep_rew   = np.zeros(n_agents, dtype=np.float32)
#             act_hist = np.zeros(5, dtype=int)

#             for step in range(max_steps):
#                 env.render()
#                 time.sleep(delay)

#                 acts_list = []
#                 for a in range(n_agents):
#                     logits_a = policy_list[a](obs[a].unsqueeze(0))[0].squeeze(0)
#                     act_a    = (Categorical(logits=logits_a).sample()
#                                 if stochastic else logits_a.argmax(-1))
#                     acts_list.append(act_a)
#                 actions = torch.stack(acts_list)
#                 acts_np = actions.cpu().numpy().astype(int)
#                 for a in acts_np:
#                     act_hist[a] += 1

             

#                 obs_tuple, rewards, done, truncated, _ = env.step(tuple(int(a) for a in acts_np))
#                 ep_rew += np.array(rewards, dtype=np.float32)
#                 obs     = torch.as_tensor(
#                     process_obs_batch(obs_tuple), dtype=torch.float32, device=device
#                 )
#                 if (all(done) if hasattr(done, "__iter__") else done) or \
#                         (all(truncated) if hasattr(truncated, "__iter__") else truncated):
#                     break

#             ep_returns.append(ep_rew)
#             total   = act_hist.sum()
#             act_str = "  ".join(
#                 f"{ACTION_NAMES[i]}:{act_hist[i]/total:.0%}"
#                 for i in range(5) if act_hist[i] > 0
#             )
#             print(f"Ep {ep} | steps={step+1} | deliveries={ep_rew.sum():.1f} | "
#                   f"per_agent={np.round(ep_rew,2)} | acts: {act_str}")

#     except KeyboardInterrupt:
#         print("Stopped.")
#     finally:
#         try:
#             env.close()
#         except AttributeError:
#             pass

#     if ep_returns:
#         arr        = np.stack(ep_returns)
#         deliveries = arr.sum(axis=1)
#         print(f"Mean deliveries/ep: {deliveries.mean():.2f}  "
#               f"(min {deliveries.min():.2f}  max {deliveries.max():.2f})")

@torch.no_grad()
def visualize_policy(
    policies,
    env_name:   str   = "rware-tiny-4ag-easy-v2",
    episodes:   int   = 3,
    max_steps:  int   = 500,
    delay:      float = 0.08,
    stochastic: bool  = True,
    device            = None,
    save_gif:   str   = None,
    gif_fps:    int   = 10,
):
    """
    policies : SharedPolicy | str | list[SharedPolicy | str]
               Mix freely for cross-play. Strings are checkpoint paths.
    save_gif : str | None
               If provided, saves all episodes to a gif at this path.
    gif_fps  : int
               Frames per second for the saved gif.
    """
    if device is None:
        device = torch.device("cpu")

    def _load(p):
        if isinstance(p, torch.nn.Module):
            return p.to(device)
        ckpt = torch.load(p, map_location=device, weights_only=False)
        key  = "true_policy" if "true_policy" in ckpt else "policy"
        pol  = SharedPolicy(ckpt["obs_dim"], ckpt["act_dim"]).to(device)
        pol.load_state_dict(_strip_compiled(ckpt[key]))
        if ckpt.get("rewards"):
            r  = ckpt["rewards"]
            sc = [x[0] if isinstance(x, (list, tuple)) else x for x in r]
            print(f"  [{p}] final={sc[-1]:.3f}  best={max(sc):.3f}  over {len(sc)*10} updates")
        return pol

    env_probe = gym.make(env_name)
    n_agents  = env_probe.unwrapped.n_agents
    env_probe.close()

    if isinstance(policies, (SharedPolicy, str)) or (
            isinstance(policies, torch.nn.Module) and not isinstance(policies, list)):
        pol         = _load(policies)
        policy_list = [pol] * n_agents
    else:
        policy_list = [_load(p) for p in policies]
        assert len(policy_list) == n_agents, f"Expected {n_agents} policies, got {len(policy_list)}"

    for p in policy_list:
        p.eval()

    cross_play = len(set(id(p) for p in policy_list)) > 1
    print(f"Visualizing '{env_name}' | {n_agents} agents | "
          f"{'cross-play' if cross_play else 'self-play'} | "
          f"{'stochastic' if stochastic else 'greedy'} | "
          f"{episodes} ep | delay={delay}s"
          + (f" | saving to {save_gif}" if save_gif else ""))

    try:
        import rware.rendering as _rr
        _orig = _rr.Viewer._draw_shelfs
        def _safe(self, env):
            s = getattr(env, "shelfs", None) or getattr(env, "shelves", None)
            if s:
                _orig(self, env)
        _rr.Viewer._draw_shelfs = _safe
    except Exception:
        pass

    env = gym.make(env_name, render_mode="human")

    def _capture_frame():
        """Grab current pyglet window contents as numpy rgb array."""
        renderer = getattr(env.unwrapped, 'renderer', None)
        if renderer is None:
            return None
        win = getattr(renderer, 'window', None)
        if win is None:
            return None
        try:
            import ctypes
            win.switch_to()
            win.dispatch_events()
            w, h = win.get_framebuffer_size()
            arr = np.empty((h, w, 4), dtype=np.uint8)
            from pyglet.gl import glReadPixels, GL_RGBA, GL_UNSIGNED_BYTE
            glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE,
                         arr.ctypes.data_as(ctypes.c_void_p))
            return np.ascontiguousarray(arr[::-1, :, :3])
        except Exception as e:
            print(f"Frame capture error: {e}")
            return None

    ep_returns          = []
    all_step_deliveries = []
    frames              = []

    try:
        for ep in range(1, episodes + 1):
            obs_tuple, _ = env.reset()
            obs    = torch.as_tensor(
                process_obs_batch(obs_tuple), dtype=torch.float32, device=device
            )
            ep_rew          = np.zeros(n_agents, dtype=np.float32)
            act_hist        = np.zeros(5, dtype=int)
            step_deliveries = []

            for step in range(max_steps):
                env.render()
                if save_gif:
                    frame = _capture_frame()
                    if frame is not None:
                        if len(frames) == 0:
                            print(f"First frame captured: shape={frame.shape}")
                        frames.append(frame)
                time.sleep(delay)

                acts_list = []
                for a in range(n_agents):
                    logits_a = policy_list[a](obs[a].unsqueeze(0))[0].squeeze(0)
                    act_a    = (Categorical(logits=logits_a).sample()
                                if stochastic else logits_a.argmax(-1))
                    acts_list.append(act_a)
                actions = torch.stack(acts_list)
                acts_np = actions.cpu().numpy().astype(int)
                for a in acts_np:
                    act_hist[a] += 1

                obs_tuple, rewards, done, truncated, _ = env.step(tuple(int(a) for a in acts_np))
                step_rew  = np.array(rewards, dtype=np.float32)
                ep_rew   += step_rew
                step_deliveries.append(np.round(step_rew).sum())
                obs = torch.as_tensor(
                    process_obs_batch(obs_tuple), dtype=torch.float32, device=device
                )
                if (all(done) if hasattr(done, "__iter__") else done) or \
                        (all(truncated) if hasattr(truncated, "__iter__") else truncated):
                    break

            ep_returns.append(ep_rew)
            all_step_deliveries.append(step_deliveries)
            total   = act_hist.sum()
            act_str = "  ".join(
                f"{ACTION_NAMES[i]}:{act_hist[i]/total:.0%}"
                for i in range(5) if act_hist[i] > 0
            )
            print(f"Ep {ep} | steps={step+1} | deliveries={ep_rew.sum():.1f} | "
                  f"per_agent={np.round(ep_rew,2)} | acts: {act_str}")

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        try:
            env.close()
        except AttributeError:
            pass

    if ep_returns:
        arr        = np.stack(ep_returns)
        deliveries = arr.sum(axis=1)
        print(f"Mean deliveries/ep: {deliveries.mean():.2f}  "
              f"(min {deliveries.min():.2f}  max {deliveries.max():.2f})")

    # Save gif
    if save_gif and frames:
        try:
            from PIL import Image
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(
                save_gif, save_all=True, append_images=imgs[1:],
                duration=int(1000 / gif_fps), loop=0,
            )
            print(f"Saved {len(frames)} frames to {save_gif}")
        except ImportError:
            print("PIL not installed — run `pip install Pillow` to save gifs.")
    elif save_gif and not frames:
        print("No frames captured — ensure pyglet is available and a display is present.")

    # Delivery rate plot
    if all_step_deliveries:
        max_len = max(len(s) for s in all_step_deliveries)
        padded  = []
        for s in all_step_deliveries:
            cum = np.cumsum(s)
            if len(cum) < max_len:
                cum = np.concatenate([cum, np.full(max_len - len(cum), cum[-1])])
            padded.append(cum)

        padded        = np.array(padded)
        mean_cum      = padded.mean(axis=0)
        steps_arr     = np.arange(1, max_len + 1)
        mean_per_step = mean_cum / steps_arr

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        for cum in padded:
            ax1.plot(steps_arr, cum, alpha=0.3, lw=1)
        ax1.plot(steps_arr, mean_cum, 'k-', lw=2, label='Mean')
        ax1.set_xlabel("Step"); ax1.set_ylabel("Cumulative deliveries")
        ax1.set_title(f"Cumulative deliveries — {env_name}"); ax1.legend()

        for cum in padded:
            ax2.plot(steps_arr, cum / steps_arr, alpha=0.3, lw=1)
        ax2.plot(steps_arr, mean_per_step, 'k-', lw=2, label='Mean')
        ax2.set_xlabel("Step"); ax2.set_ylabel("Deliveries / step (running avg)")
        ax2.set_title(f"Delivery rate — {env_name}"); ax2.legend()

        plt.tight_layout()
        plt.pause(0.1)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(van_files, rob_files, env_name: str, n_agents: int,
                 num_envs: int = 8, device=None):
    if device is None:
        device = torch.device("cpu")

    fig, ax = plt.subplots(figsize=(8, 4))
    van_rewards, rob_rewards = [], []

    for fn in van_files:
        _, r = load_vanilla(fn, device)
        r = np.array([x[0] if isinstance(x, (list,tuple)) else x for x in r])
        ax.plot(np.arange(len(r)) * 10, r, c="red", alpha=0.2)
        van_rewards.append(r)
    for fn in rob_files:
        _, _, r = load_robust(fn, device)
        r = np.array([x[0] if isinstance(x, (list,tuple)) else x for x in r])
        ax.plot(np.arange(len(r)) * 10, r, c="blue", alpha=0.2)
        rob_rewards.append(r)

    if van_rewards:
        n = min(len(x) for x in van_rewards)
        ax.plot(np.arange(n)*10, np.mean([x[:n] for x in van_rewards], 0),
                c="red", label="Vanilla IPPO")
    if rob_rewards:
        n = min(len(x) for x in rob_rewards)
        ax.plot(np.arange(n)*10, np.mean([x[:n] for x in rob_rewards], 0),
                c="blue", label="Risk-Averse IPPO")

    ax.set_xlabel("Updates"); ax.set_ylabel("Score"); ax.legend()
    ax.set_title(f"Learning curves — {env_name}")
    plt.tight_layout(); plt.savefig("rware_curves.png", dpi=150); plt.pause(0.1)

    all_files = van_files + rob_files
    n = len(all_files)
    if n < 2:
        return

    eval_env = ParallelEnv(32, 8, env_name)
    policies = ([load_vanilla(fn, device)[0] for fn in van_files] +
                [load_robust(fn, device)[0]  for fn in rob_files])

    half = n_agents // 2
    matrix = np.zeros((n, n))
    for i, pi in enumerate(policies):
        for j, pj in enumerate(policies):
            # Build per-agent policy list: first half=pi, second half=pj
            policy_list = [pi] * half + [pj] * (n_agents - half)
            random.shuffle(policy_list)
            score, _ = evaluate_policy(eval_env,policy_list, n_agents, device, n_steps=500)
            matrix[i, j] = score
        print(f"  row {i+1}/{n} done")
    eval_env.close()


    print('Plotting...')
    plt.close('all'); 
    plt.matshow(matrix,cmap='inferno'); 

    plt.colorbar(); 
    plt.gca().set_xticks([-0.5,len(van_files)-0.5,len(all_files)-0.5], labels=[],fontsize=8); 
    plt.gca().set_yticks([-0.5,len(van_files)-0.5,len(all_files)-0.5], labels=[],fontsize=8); 
    plt.gca().tick_params(length=5, width=1, which='major',color='k',bottom=False)
    plt.gca().minorticks_on()
    plt.gca().tick_params(length=0, width=1, which='minor',color='k',bottom=False,labelsize=8)
    import math
    mp1=(-0.5+len(van_files)-0.5)/2
    mp2=(len(van_files)-0.5+len(all_files)-0.5)/2+0.5
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

    plt.pause(0.1)
    return matrix

def eval_scaling(fn, n_agents_list, device, n_steps=500):
    results = []
    for i in n_agents_list:
        env = ParallelEnv(16, 8, f"rware-tiny-{i}ag-v2")
        score = evaluate_policy(env, fn, n_agents=i, device=device, n_steps=n_steps)[0]
        env.close()
        results.append(score)
        print('File: '+fn + '--------------' +str(i)+': '+str(score))
    return results


#----------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    ENV_NAME       = "rware-tiny-4ag-easy-v2"
    M              = 3
    N_RUNS         = 5
    STEPS          = 20_000_000
    LR             = 5e-4
    ENTROPY        = 0.05
    RISK           = 0.2
    NUM_ENVS       = 16
    NUM_STEPS      = 500
    MINIBATCH      = 2048
    PICKUP_SHAPING = 0.01
    DEVICE         = torch.device("cpu")

    TRAIN_VAN = 0
    TRAIN_ROB = 1
    BENCHMARK = 0

    dummy = RWAREWrapper(ENV_NAME)
    OBS_DIM, ACT_DIM, N_AGENTS = dummy.obs_dim, dummy.action_dim, dummy.num_agents
    dummy.close()
    van_files, rob_files = [], []

    if TRAIN_VAN:
        for idx in range(N_RUNS):
            fn = f"rware_vanilla_{idx+6}.pt"
            print(f"\n--- Vanilla run {idx} ---")
            policy = SharedPolicy(OBS_DIM, ACT_DIM).to(DEVICE)
            train_vanilla_ippo(
                env_name=ENV_NAME, policy=policy,
                steps=STEPS, lr=LR, entropy_coef=ENTROPY,
                num_envs=NUM_ENVS, num_steps=NUM_STEPS, minibatch_size=MINIBATCH,
                pickup_shaping=PICKUP_SHAPING, filename=fn, device=DEVICE,
            )
            van_files.append(fn)
    else:
        van_files = [f"rware_vanilla_{i}.pt" for i in range(10)]

    if TRAIN_ROB:
        for idx in range(N_RUNS):
            fn = f"./robust_agents_easy_risk_02/rware_robust_02_{M}_{idx+5}.pt"
            print(f"\n--- Risk-averse run {idx} ---")
            policy = SharedPolicy(OBS_DIM, ACT_DIM).to(DEVICE)
            a_pol  = copy.deepcopy(policy)
            train_risk_averse_ippo(
                env_name=ENV_NAME, m=M,
                true_policy=policy, adv_policy=a_pol,
                steps=STEPS, lr=LR, entropy_coef=ENTROPY,
                risk_factor=RISK, num_envs=NUM_ENVS,
                num_steps=NUM_STEPS, minibatch_size=MINIBATCH,
                pickup_shaping=PICKUP_SHAPING, filename=fn, device=DEVICE,
            )
            rob_files.append(fn)
    else:
        #rob_files = [f'rware_robust_05_3_{i}.pt' for i in range(5)] +[f'rware_robust_02_3_{i}.pt' for i in range(5)]+[f'rware_robust_005_3_{i}.pt' for i in range(5)] +[f'rware_robust_001_3_{i}.pt' for i in range(5)]
        rob_files = [f'rware_robust_005_3_{i}.pt' for i in range(10)]

    if BENCHMARK:

        ##Makes the cross play plots in an environment with N agents
        #N=10
        #matrix=plot_results(van_files, rob_files, f"rware-tiny-{N}ag-easy-v2",10,num_envs=NUM_ENVS, device=DEVICE)

        # 

        # Plots the scaling with number of agents in environment "rware-tiny-NUM_AGENTS-ag-v2"

        n_agents_list = list(range(2, 19, 2))
        print(n_agents_list)
        num_ags=3
        rob_fns0 = [f'./robust_agents_easy_risk_05/rware_robust_05_3_{i}.pt'  for i in range(num_ags)] 
        rob_fns1 = [f'./robust_agents_easy_risk_02/rware_robust_02_3_{i}.pt'  for i in range(num_ags)] 
        rob_fns2 = [f'./robust_agents_easy_risk_005/rware_robust_005_3_{i}.pt'for i in range(num_ags)]
        rob_fns3 = [f'./robust_agents_easy_risk_001/rware_robust_001_3_{i}.pt'for i in range(num_ags)]
        van_files = [f"./vanilla_agents_easy/rware_vanilla_{i}.pt" for i in range(num_ags)]

        scaled_van   = [eval_scaling(fn, n_agents_list, DEVICE) for fn in van_files]
        scaled_robs0 = [eval_scaling(fn, n_agents_list, DEVICE) for fn in rob_fns0]
        scaled_robs1 = [eval_scaling(fn, n_agents_list, DEVICE) for fn in rob_fns1]
        scaled_robs2 = [eval_scaling(fn, n_agents_list, DEVICE) for fn in rob_fns2]
        scaled_robs3 = [eval_scaling(fn, n_agents_list, DEVICE) for fn in rob_fns3]
       

        plt.plot(n_agents_list , np.mean(np.array(scaled_van),axis=0),'indianred', lw=2, label='PPO Self-Play');
        plt.plot(n_agents_list , np.mean(np.array(scaled_robs0),axis=0),'dodgerblue', linestyle='--', lw=2, label='SRPO (0.5) Self-Play');
        plt.plot(n_agents_list , np.mean(np.array(scaled_robs1),axis=0),'dodgerblue', linestyle=':', lw=2, label='SRPO (0.2) Self-Play');
        plt.plot(n_agents_list , np.mean(np.array(scaled_robs2),axis=0),'dodgerblue', linestyle='-', lw=2, label='SRPO (0.05) Self-Play'); 
        plt.plot(n_agents_list , np.mean(np.array(scaled_robs3),axis=0),'dodgerblue', linestyle='-.', lw=2, label='SRPO (0.01) Self-Play'); 
        plt.legend(); plt.plot([4,4],[0,92],'k--',lw=0.5, alpha=0.5); 
        plt.xlabel('Number of Agents'); 
        plt.ylabel('Number of Deliveries');
        plt.show()
