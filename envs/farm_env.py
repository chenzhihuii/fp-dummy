
# envs/farm_env.py
# Minimal RL environment tailored for DQN training (no hard dependency on gym)
# You can later replace dummy predictive runner with real models.

from dataclasses import dataclass
import numpy as np
from typing import Callable, Dict, Any

class _Discrete:
    def __init__(self, n: int):
        self.n = n

@dataclass
class EnvConfig:
    max_steps: int = 52            # 1 step = 1 minggu
    stok_awal: float = 0.3
    uang_awal: float = 0.5
    cluster_id: int = 0
    n_clusters: int = 3
    random_seed: int = 42

def default_feature_builder(ctx: Dict[str, Any] = None, warmup: bool = False) -> np.ndarray:
    """
    Builds a fixed-length state vector in [0,1].
    When warmup=True, returns a zero vector to define state size before first reset.
    """
    if warmup or ctx is None:
        # [stok, uang, bulan_sin, bulan_cos, dummy, hasil_pred, harga_pred, proba_ketahanan] + 3 one-hot clusters = 11
        return np.zeros(11, dtype=np.float32)

    stok = float(np.clip(ctx.get("stok", 0.0), 0.0, 1.0))
    uang = float(np.clip(ctx.get("uang", 0.0), 0.0, 1.0))
    preds = ctx.get("preds", {"hasil_pred_norm":0.0,"harga_pred_norm":0.0,"proba_ketahanan":0.5})
    cluster_id = int(ctx.get("cluster_id", 0))
    n_clusters = int(ctx.get("n_clusters", 3))

    # seasonal encoding (placeholder: constant; plug actual month later)
    bulan_sin, bulan_cos = 0.0, 1.0
    dummy = 0.5

    # cluster one-hot
    cluster_oh = np.zeros(n_clusters, dtype=np.float32)
    cluster_oh[min(cluster_id, n_clusters-1)] = 1.0

    s = np.array([
        stok, uang, bulan_sin, bulan_cos, dummy,
        float(preds.get("hasil_pred_norm", 0.0)),
        float(preds.get("harga_pred_norm", 0.0)),
        float(preds.get("proba_ketahanan", 0.5)),
        *cluster_oh
    ], dtype=np.float32)
    return s

def default_predictive_runner(rng: np.random.Generator) -> Callable[[], Dict[str, float]]:
    """
    Returns a callable that produces normalized predictions in [0,1].
    Replace with real model inference later.
    """
    def _run():
        # slightly stochastic but bounded signals
        hasil_pred = float(np.clip(rng.normal(0.35, 0.05), 0.0, 1.0))
        harga_pred = float(np.clip(rng.normal(0.6, 0.08), 0.0, 1.0))
        proba_ketahanan = float(np.clip(rng.normal(0.75, 0.1), 0.0, 1.0))
        return {
            "hasil_pred": hasil_pred,
            "harga_pred": harga_pred,
            "proba_ketahanan": proba_ketahanan,
            "hasil_pred_norm": hasil_pred,
            "harga_pred_norm": harga_pred
        }
    return _run

class FarmEnv:
    """
    Minimal environment:
    - state: features + predictive outputs
    - actions (0..5):
        0: tunda tanam
        1: tanam komoditas A
        2: tanam komoditas B
        3: tambah pupuk
        4: irigasi
        5: jual sebagian stok
    - reward: shaped mix of margin, availability, risk, and cost
    """

    def __init__(
        self,
        feature_builder: Callable[..., np.ndarray] = default_feature_builder,
        predictive_runner: Callable[[], Dict[str, float]] = None,
        config: EnvConfig = None
    ):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.random_seed)
        self.feature_builder = feature_builder
        self.predictive_runner = predictive_runner or default_predictive_runner(self.rng)

        # public-like attributes used by training loop
        self.action_space = _Discrete(6)
        self.max_steps = self.cfg.max_steps

        # internal state
        self.step_count = 0
        self.cluster_id = self.cfg.cluster_id
        self.stok = self.cfg.stok_awal
        self.uang = self.cfg.uang_awal

        self.state_size = len(self.feature_builder(None, warmup=True))
        self.state = self.feature_builder(None, warmup=True)

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.stok = self.cfg.stok_awal
        self.uang = self.cfg.uang_awal
        preds = self.predictive_runner()
        self.state = self.feature_builder({
            "stok": self.stok,
            "uang": self.uang,
            "cluster_id": self.cluster_id,
            "n_clusters": self.cfg.n_clusters,
            "preds": preds
        })
        return self.state.copy()

    def step(self, action: int):
        self.step_count += 1
        preds = self.predictive_runner()

        biaya = 0.0
        hasil = 0.0
        pendapatan = 0.0

        # action dynamics
        if action == 0:   # tunda tanam
            biaya += 0.01
        elif action == 1: # tanam A
            biaya += 0.05
            hasil += preds["hasil_pred"] * 0.8
        elif action == 2: # tanam B
            biaya += 0.06
            hasil += preds["hasil_pred"] * 0.9
        elif action == 3: # pupuk
            biaya += 0.03
            hasil += 0.05
        elif action == 4: # irigasi
            biaya += 0.02
            hasil += 0.03
        elif action == 5: # jual stok
            jual = min(self.stok, 0.1)
            pendapatan += jual * preds["harga_pred"]
            self.stok -= jual

        # natural decay/consumption
        self.stok = float(np.clip(self.stok + hasil - 0.02, 0.0, 1.0))
        self.uang  = float(np.clip(self.uang  + pendapatan - biaya, 0.0, 1.0))

        # reward shaping
        margin_norm = (np.clip(pendapatan - biaya, -0.1, 0.1) + 0.1) / 0.2 # 0..1
        ketersediaan = self.stok
        risiko = 1.0 - preds["proba_ketahanan"]
        biaya_norm = np.clip(biaya / 0.1, 0.0, 1.0)

        w1, w2, w3, w4 = 0.5, 0.3, 0.1, 0.1
        reward = (w1 * margin_norm) + (w2 * ketersediaan) - (w3 * risiko) - (w4 * biaya_norm)

        self.state = self.feature_builder({
            "stok": self.stok,
            "uang": self.uang,
            "cluster_id": self.cluster_id,
            "n_clusters": self.cfg.n_clusters,
            "preds": preds
        })

        done = (self.step_count >= self.max_steps)
        info = {"biaya": biaya, "pendapatan": pendapatan, "hasil": hasil}
        return self.state.copy(), float(reward), bool(done), info
