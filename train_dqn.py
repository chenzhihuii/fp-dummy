
# train_dqn.py
# Minimal end-to-end training loop using the DQNAgent already in /agents

import os
import numpy as np
from agents.dqn_agent import DQNAgent
from envs.farm_env import FarmEnv, EnvConfig, default_feature_builder, default_predictive_runner

RUN_DIR = "runs"
os.makedirs(RUN_DIR, exist_ok=True)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train(episodes=400, batch_size=32, seed=42):
    cfg = EnvConfig(random_seed=seed, max_steps=52, cluster_id=1)
    env = FarmEnv(
        feature_builder=default_feature_builder,
        predictive_runner=default_predictive_runner(np.random.default_rng(seed)),
        config=cfg
    )

    state = env.reset()
    state = np.reshape(state, [1, env.state_size])
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_space.n)

    best_avg = -1e9
    rewards_hist = []

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        total_reward = 0.0

        for t in range(env.max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, env.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                break

        rewards_hist.append(total_reward)
        avg100 = np.mean(rewards_hist[-50:])  # moving average window 50 episodes
        print(f"Episode {e+1:4d}/{episodes} | Reward: {total_reward: .3f} | Avg50: {avg100: .3f} | eps={agent.epsilon: .3f}")

        # save best weights
        if avg100 > best_avg and len(rewards_hist) >= 50:
            best_avg = float(avg100)
            agent.model.save(os.path.join(MODEL_DIR, "dqn_best.h5"))

    # final save
    agent.model.save(os.path.join(MODEL_DIR, "dqn_last.h5"))
    np.savetxt(os.path.join(RUN_DIR, "rewards.csv"), np.array(rewards_hist), delimiter=",")
    print("Training finished. Weights and rewards saved.")

if __name__ == "__main__":
    train()
