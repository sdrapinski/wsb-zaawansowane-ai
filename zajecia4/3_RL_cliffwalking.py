from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Laboratorium: uczenie ze wzmocnieniem
# Temat: porównanie Q-learning oraz SARSA w środowisku Cliff Walking
# Plik jest samowystarczalny: nie wymaga biblioteki gym/gymnasium.
# Wystarczą: Python 3.x, numpy, matplotlib.
# ============================================================


@dataclass
class Config:
    rows: int = 4
    cols: int = 12
    episodes: int = 800
    max_steps: int = 200
    alpha: float = 0.10
    gamma: float = 0.95
    epsilon_start: float = 1.00
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 42
    step_penalty: float = -1.0
    cliff_penalty: float = -100.0
    goal_reward: float = -1.0
    eval_episodes: int = 200
    moving_avg_window: int = 25
    output_dir: str = "wyniki_rl"


class CliffWalkingEnv:
    """Klasyczne środowisko 4x12 znane z Sutton & Barto.

    S = start, G = goal, C = cliff
    Agent startuje w lewym dolnym rogu i próbuje dotrzeć do celu.
    Wejście na klif daje dużą karę i powrót do startu.
    """

    ACTIONS = {
        0: (-1, 0),  # gora
        1: (0, 1),   # prawo
        2: (1, 0),   # dol
        3: (0, -1),  # lewo
    }
    ACTION_NAMES = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    def __init__(
        self,
        rows: int = 4,
        cols: int = 12,
        step_penalty: float = -1.0,
        cliff_penalty: float = -100.0,
        goal_reward: float = -1.0,
    ):
        self.rows = rows
        self.cols = cols
        self.n_states = rows * cols
        self.n_actions = 4
        self.step_penalty = step_penalty
        self.cliff_penalty = cliff_penalty
        self.goal_reward = goal_reward
        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)
        self.cliff = {(rows - 1, c) for c in range(1, cols - 1)}
        self.agent_pos = self.start

    def state_to_index(self, pos: Tuple[int, int]) -> int:
        r, c = pos
        return r * self.cols + c

    def index_to_state(self, idx: int) -> Tuple[int, int]:
        return idx // self.cols, idx % self.cols

    def reset(self) -> int:
        self.agent_pos = self.start
        return self.state_to_index(self.agent_pos)

    def step(self, action: int) -> Tuple[int, float, bool]:
        dr, dc = self.ACTIONS[action]
        r, c = self.agent_pos
        nr = min(max(r + dr, 0), self.rows - 1)
        nc = min(max(c + dc, 0), self.cols - 1)
        next_pos = (nr, nc)

        if next_pos in self.cliff:
            self.agent_pos = self.start
            return self.state_to_index(self.agent_pos), self.cliff_penalty, False

        if next_pos == self.goal:
            self.agent_pos = next_pos
            return self.state_to_index(self.agent_pos), self.goal_reward, True

        self.agent_pos = next_pos
        return self.state_to_index(self.agent_pos), self.step_penalty, False

    def render_ascii(self, policy: np.ndarray | None = None) -> str:
        lines = []
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                pos = (r, c)
                if pos == self.start:
                    row_chars.append(" S ")
                elif pos == self.goal:
                    row_chars.append(" G ")
                elif pos in self.cliff:
                    row_chars.append(" C ")
                else:
                    if policy is None:
                        row_chars.append(" . ")
                    else:
                        idx = self.state_to_index(pos)
                        row_chars.append(f" {self.ACTION_NAMES[int(policy[idx])]} ")
            lines.append("".join(row_chars))
        return "\n".join(lines)


def epsilon_greedy(q_table: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_table.shape[1]))
    return int(np.argmax(q_table[state]))


def moving_average(values: List[float], window: int) -> np.ndarray:
    if window <= 1:
        return np.array(values, dtype=float)
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    prefix = np.array([arr[: i + 1].mean() for i in range(window - 1)])
    return np.concatenate([prefix, smoothed])


def derive_greedy_policy(q_table: np.ndarray) -> np.ndarray:
    return np.argmax(q_table, axis=1)


def train_q_learning(env: CliffWalkingEnv, cfg: Config) -> Dict[str, np.ndarray | List[float] | float]:
    rng = np.random.default_rng(cfg.seed)
    q_table = np.zeros((env.n_states, env.n_actions), dtype=float)
    rewards = []
    eps_history = []
    epsilon = cfg.epsilon_start

    for _ in range(cfg.episodes):
        state = env.reset()
        episode_reward = 0.0

        for _step in range(cfg.max_steps):
            action = epsilon_greedy(q_table, state, epsilon, rng)
            next_state, reward, done = env.step(action)

            best_next = np.max(q_table[next_state])
            td_target = reward + cfg.gamma * best_next * (0 if done else 1)
            td_error = td_target - q_table[state, action]
            q_table[state, action] += cfg.alpha * td_error

            state = next_state
            episode_reward += reward
            if done:
                break

        rewards.append(episode_reward)
        eps_history.append(epsilon)
        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

    return {
        "q_table": q_table,
        "episode_rewards": rewards,
        "epsilon_history": eps_history,
        "final_epsilon": epsilon,
    }


def train_sarsa(env: CliffWalkingEnv, cfg: Config) -> Dict[str, np.ndarray | List[float] | float]:
    rng = np.random.default_rng(cfg.seed)
    q_table = np.zeros((env.n_states, env.n_actions), dtype=float)
    rewards = []
    eps_history = []
    epsilon = cfg.epsilon_start

    for _ in range(cfg.episodes):
        state = env.reset()
        action = epsilon_greedy(q_table, state, epsilon, rng)
        episode_reward = 0.0

        for _step in range(cfg.max_steps):
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(q_table, next_state, epsilon, rng)

            td_target = reward + cfg.gamma * q_table[next_state, next_action] * (0 if done else 1)
            td_error = td_target - q_table[state, action]
            q_table[state, action] += cfg.alpha * td_error

            state, action = next_state, next_action
            episode_reward += reward
            if done:
                break

        rewards.append(episode_reward)
        eps_history.append(epsilon)
        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

    return {
        "q_table": q_table,
        "episode_rewards": rewards,
        "epsilon_history": eps_history,
        "final_epsilon": epsilon,
    }


def evaluate_policy(env: CliffWalkingEnv, q_table: np.ndarray, episodes: int = 200, max_steps: int = 200) -> Dict[str, float]:
    rewards = []
    steps_taken = []
    successes = 0

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        for step_idx in range(1, max_steps + 1):
            action = int(np.argmax(q_table[state]))
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                successes += 1
                steps_taken.append(step_idx)
                break
        else:
            steps_taken.append(max_steps)

        rewards.append(total_reward)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(successes / episodes),
        "mean_steps": float(np.mean(steps_taken)),
    }


def plot_learning_curves(results: Dict[str, Dict], cfg: Config, output_path: Path) -> None:
    plt.figure(figsize=(10, 5.5))
    for name, result in results.items():
        rewards = result["episode_rewards"]
        ma = moving_average(rewards, cfg.moving_avg_window)
        plt.plot(ma, label=f"{name} (średnia krocząca)")

    plt.title("Porównanie przebiegu uczenia")
    plt.xlabel("Epizod")
    plt.ylabel("Skumulowana nagroda w epizodzie")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_policy(env: CliffWalkingEnv, q_table: np.ndarray, title: str, output_path: Path) -> None:
    policy = derive_greedy_policy(q_table)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_xticks(np.arange(0, env.cols + 1, 1))
    ax.set_yticks(np.arange(0, env.rows + 1, 1))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    arrows = {0: (0, -0.25), 1: (0.25, 0), 2: (0, 0.25), 3: (-0.25, 0)}
    symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    for r in range(env.rows):
        for c in range(env.cols):
            state = env.state_to_index((r, c))
            x, y = c + 0.5, r + 0.5
            pos = (r, c)

            if pos == env.start:
                ax.text(x, y, 'S', ha='center', va='center', fontsize=16, fontweight='bold')
            elif pos == env.goal:
                ax.text(x, y, 'G', ha='center', va='center', fontsize=16, fontweight='bold')
            elif pos in env.cliff:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, alpha=0.25))
                ax.text(x, y, 'CL', ha='center', va='center', fontsize=11, fontweight='bold')
            else:
                action = int(policy[state])
                dx, dy = arrows[action]
                ax.arrow(x, y, dx, dy, head_width=0.10, head_length=0.08, length_includes_head=True)
                ax.text(x, y + 0.28, symbols[action], ha='center', va='center', fontsize=10)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_metrics_csv(metrics: Dict[str, Dict[str, float]], cfg: Config, output_path: Path) -> None:
    rows = []
    for algo_name, metric_values in metrics.items():
        row = {"algorytm": algo_name, **metric_values}
        row.update({f"cfg_{k}": v for k, v in asdict(cfg).items() if k != "output_dir"})
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(metrics: Dict[str, Dict[str, float]]) -> None:
    print("\n=== PODSUMOWANIE OCENY POLITYK (tryb zachłanny) ===")
    for algo_name, metric_values in metrics.items():
        print(f"\n{algo_name}")
        print(f"  średnia nagroda  : {metric_values['mean_reward']:.2f}")
        print(f"  odchylenie std   : {metric_values['std_reward']:.2f}")
        print(f"  success rate     : {metric_values['success_rate']:.2%}")
        print(f"  średnia liczba kroków: {metric_values['mean_steps']:.2f}")


def main() -> None:
    cfg = Config()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = CliffWalkingEnv(
        rows=cfg.rows,
        cols=cfg.cols,
        step_penalty=cfg.step_penalty,
        cliff_penalty=cfg.cliff_penalty,
        goal_reward=cfg.goal_reward,
    )

    q_learning_result = train_q_learning(env, cfg)
    sarsa_result = train_sarsa(env, cfg)

    results = {
        "Q-learning": q_learning_result,
        "SARSA": sarsa_result,
    }

    metrics = {
        name: evaluate_policy(env, result["q_table"], episodes=cfg.eval_episodes, max_steps=cfg.max_steps)
        for name, result in results.items()
    }

    plot_learning_curves(results, cfg, output_dir / "wykres_uczenia.png")
    plot_policy(env, q_learning_result["q_table"], "Polityka zachłanna - Q-learning", output_dir / "polityka_q_learning.png")
    plot_policy(env, sarsa_result["q_table"], "Polityka zachłanna - SARSA", output_dir / "polityka_sarsa.png")
    save_metrics_csv(metrics, cfg, output_dir / "metryki.csv")

    with open(output_dir / "polityka_q_learning.txt", "w", encoding="utf-8") as f:
        f.write(env.render_ascii(derive_greedy_policy(q_learning_result["q_table"])))

    with open(output_dir / "polityka_sarsa.txt", "w", encoding="utf-8") as f:
        f.write(env.render_ascii(derive_greedy_policy(sarsa_result["q_table"])))

    print_summary(metrics)
    print("\nPliki wynikowe zapisano w katalogu:", output_dir.resolve())


if __name__ == "__main__":
    main()
