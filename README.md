# Proyek Ketahanan Pangan - RL Core

## 🧪 RL-Only Quickstart

1) Install deps (example):
```bash
pip install numpy tensorflow==2.*
```
*(Optional later: gymnasium, but not required for this minimal env.)*

2) Train DQN:
```bash
python train_dqn.py
```

Artifacts:
- `models/dqn_best.h5`, `models/dqn_last.h5`
- `runs/rewards.csv`

3) Next (after you have real models):
- Replace `default_predictive_runner` in `envs/farm_env.py` with calls to your predictive models (hasil panen, harga, risiko).
- Extend action meanings and reward shaping with domain parameters.
