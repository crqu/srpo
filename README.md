# SRPO: Training Generalizable Collaborative Agents via Strategic Risk Aversion

Official code for the NeurIPS 2026 paper *Training Generalizable Collaborative Agents via Strategic Risk Aversion*.

SRPO (Strategically Risk-Averse Policy Optimization) trains a **hero** policy against a KL-constrained **adversary** that replaces its partner during training. The adversary maximizes the negative of the hero's return while being penalized by KL(adversary || hero), keeping the worst-case partner realistic. The resulting hero policy is robust to partner variation at deployment time.

## Environments

| Environment | Description | Location |
|---|---|---|
| LLM debate (GSM8K) | Multi-round math debate, verl/Ray, multi-GPU | `verl/trainer/` |
| Cooperative grid-world | Overcooked-like, single-GPU PyTorch | `gridworld/coop_ppo.py` |
| Tag | PettingZoo MPE `simple_tag` | `gridworld/tag_decentralized.py` |
| RWARE | Multi-robot warehouse | `gridworld/warehouse.py` |

## Setup

```bash
# Create a Python 3.10+ env (conda or venv) and install:
pip install -r requirements.txt

# For gridworld environments only:
pip install gymnasium pettingzoo==1.24.3 rware numba matplotlib imageio psutil
```

## LLM / GSM8K

The IPPO baseline and SRPO algorithm both run from the same Hydra entry point; `multi_agent.trainer_type` selects the trainer class.

| | Path |
|---|---|
| Entry point | `verl/trainer/main_mappo.py` |
| Algorithm | `verl/trainer/ppo/mappo_trainer.py` |
| Default config | `verl/trainer/config/mappo_trainer.yaml` |

Agent 0 is the **adversary** and agent 1 is the **hero**. The adversary's reward is the negative of the hero's discounted future return, regularized by the SRPO KL term.

### Local debug (2 GPU)

```bash
bash debug_q05b_local.sh             # SRPO (default)
METHOD=ippo bash debug_q05b_local.sh # IPPO
```

### Multi-node SLURM

Edit `train_q05b.slurm` / `train.slurm` to fill in your SLURM account and partition, then:

```bash
sbatch train_q05b.slurm              # IPPO (default)
METHOD=srpo sbatch train_q05b.slurm  # SRPO
```

### Configuring SRPO

```yaml
multi_agent:
  trainer_type: risk_averse       # select RayRiskAverseTrainer
  adversary_kl_to_hero: true      # enable the SRPO KL(adv || hero) regularizer
  risk_coef: 1.0                  # 1/tau: higher = tighter adversary constraint
algorithm:
  kl_ctrl:
    kl_coef: 0.1                  # hero -> ref KL penalty (smaller = weaker)
```

## Grid-world environments

Three single-GPU PyTorch scripts, each self-contained (no verl dependency).

```bash
cd gridworld

# Cooperative grid-world (Overcooked-like, IPPO / SRPO / FCP)
python run_coop_ppo.py

# FCP pipeline (self-play partner pool then FCP best-response)
python train_fcp_full.py

# Tag (PettingZoo MPE simple_tag, IPPO and adversarial variants)
python tag_decentralized.py

# RWARE (multi-robot warehouse, IPPO and adversarial variants)
python warehouse.py
```

Each script writes checkpoints (`*.pt2`) and reward logs to the current working directory. Rerun with different `seed` values to sweep.

## Citation

```bibtex
@inproceedings{qu2026training,
  title={Training Generalizable Collaborative Agents via Strategic Risk Aversion},
  author={Qu, Chengrui and Zhang, Yizhou and Lanzetti, Nicolas and Mazumdar, Eric},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License

Built on the open-source [verl](https://github.com/volcengine/verl) framework (Apache-2.0). See `LICENSE` and `Notice.txt`.
