"""Train and evaluate a Stable-Baselines3 policy on Robot Skills.

Examples:
    python training/train_sb3_skills.py train --timesteps 100000
    python training/train_sb3_skills.py evaluate --model artifacts/models/sb3_skills.zip
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Allow both `python training/train_sb3_skills.py` and
# `python -m training.train_sb3_skills`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.sb3_skills_env import SkillsSB3Env


ALGORITHMS = {
    "a2c": "A2C",
    "ppo": "PPO",
    "sac": "SAC",
}


def _algorithm(name: str):
    from stable_baselines3 import A2C, PPO, SAC

    return {"a2c": A2C, "ppo": PPO, "sac": SAC}[name]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SB3 VEX Robot Skills training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Validate the Gymnasium adapter")

    train = subparsers.add_parser("train", help="Train an SB3 policy")
    train.add_argument("--algorithm", choices=ALGORITHMS, default="ppo")
    train.add_argument("--timesteps", type=int, default=100_000)
    train.add_argument("--model", default="artifacts/models/sb3_skills")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="auto")
    train.add_argument("--envs", type=int, default=4,
                       help="Parallel skills runs per PPO rollout")
    train.add_argument("--delay", type=float, default=0.0,
                       help="Optional delay in seconds per callback update")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved policy")
    evaluate.add_argument("--algorithm", choices=ALGORITHMS, default="ppo")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--episodes", type=int, default=5)
    evaluate.add_argument("--render", action="store_true")
    evaluate.add_argument("--seed", type=int, default=0)
    return parser


def train(args: argparse.Namespace) -> None:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    if args.envs < 1:
        raise ValueError("--envs must be at least 1")

    def make_env(rank: int):
        def factory():
            return Monitor(SkillsSB3Env(seed=args.seed + rank))

        return factory

    env = (
        DummyVecEnv([make_env(0)])
        if args.envs == 1
        else SubprocVecEnv([make_env(i) for i in range(args.envs)])
    )

    class ProgressCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.best_score = float("-inf")
            self.last_log = 0.0

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                score = info.get("skills_score")
                if score is not None and score > self.best_score:
                    self.best_score = float(score)
            now = time.monotonic()
            if now - self.last_log >= 1.0:
                print(
                    f"steps={self.num_timesteps:,} "
                    f"best_score={max(0.0, self.best_score):.0f}"
                )
                self.last_log = now
            if args.delay > 0:
                time.sleep(args.delay)
            return True

    model_class = _algorithm(args.algorithm)
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        tensorboard_log = None
    else:
        tensorboard_log = "artifacts/logs/sb3_skills"
    model = model_class(
        "MlpPolicy",
        env,
        verbose=1,
        device=args.device,
        seed=args.seed,
        tensorboard_log=tensorboard_log,
    )
    model.learn(total_timesteps=args.timesteps, callback=ProgressCallback())
    output = args.model if args.model.endswith(".zip") else f"{args.model}.zip"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    model.save(output[:-4] if output.endswith(".zip") else output)
    env.close()
    print(f"Saved SB3 skills policy to {output}")


def evaluate(args: argparse.Namespace) -> None:
    env = SkillsSB3Env(
        render_mode="human" if args.render else None,
        seed=args.seed,
    )
    model = _algorithm(args.algorithm).load(args.model, env=env)
    scores = []
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
        scores.append(float(info["skills_score"]))
        print(f"Episode {episode + 1}: {scores[-1]:.0f}")
    print(
        f"Mean skills score: {sum(scores) / len(scores):.2f} "
        f"(min={min(scores):.0f}, max={max(scores):.0f})"
    )
    env.close()


def check() -> None:
    from stable_baselines3.common.env_checker import check_env

    check_env(SkillsSB3Env(), warn=True)
    print("SB3 skills environment check passed")


def main() -> None:
    args = _parser().parse_args()
    if args.command == "check":
        check()
    elif args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
