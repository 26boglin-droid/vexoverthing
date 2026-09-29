"""Stable-Baselines3 adapter for VEX Robot Skills.

The simulator exposes a low-level action dictionary and the existing training
environment is multi-agent.  SB3 expects a single Gymnasium environment, so
this adapter presents the lone skills robot as one agent with a continuous
nine-value action vector:

``[left_drive, right_drive, intake, score_pin, score_cup, toggle,
  flip_pin, flip_cup, match_load]``

Drive values are used continuously.  The remaining values are treated as
buttons and fire when greater than zero.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config.game_rules import SKILLS_SECONDS
from config.hyperparameters import CONTROL_DT
from simulation.simulator import OverrideSimulator
from utils.observation_builder import OBS_DIM, build_observation


class SkillsSB3Env(gym.Env):
    """Single-agent Gymnasium environment for SB3 skills strategy training."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")

        self.render_mode = render_mode
        self._seed = seed
        self.sim = OverrideSimulator(
            headless=render_mode != "human",
            skills_mode=True,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )
        self._episode_score = 0.0
        self._episode_steps = 0

    def _observation(self) -> np.ndarray:
        if len(self.sim.robots) != 1:
            raise RuntimeError(
                f"Skills environment requires exactly one robot, got {len(self.sim.robots)}"
            )
        robot = self.sim.robots[0]
        return build_observation(
            robot=robot,
            all_robots=self.sim.robots,
            pins=self.sim.pins,
            cups=self.sim.cups,
            goals=self.sim.goals,
            toggles=self.sim.toggles,
            rules_engine=self.sim.rules_engine,
            simulator=self.sim,
        )

    @staticmethod
    def _action(action: np.ndarray) -> Dict[str, Any]:
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (9,):
            raise ValueError(f"Expected action shape (9,), got {values.shape}")
        values = np.clip(values, -1.0, 1.0)
        return {
            "left": float(values[0]),
            "right": float(values[1]),
            "intake": bool(values[2] > 0.0),
            "score_pin": bool(values[3] > 0.0),
            "score_cup": bool(values[4] > 0.0),
            "toggle": bool(values[5] > 0.0),
            "flip_pin": bool(values[6] > 0.0),
            "flip_cup": bool(values[7] > 0.0),
            "match_load": bool(values[8] > 0.0),
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self.sim.reset()
        self.sim.timer_started = True
        self._episode_score = float(self.sim.rules_engine.red_score)
        self._episode_steps = 0
        return self._observation(), self._telemetry()

    def _telemetry(self) -> Dict[str, Any]:
        robot = self.sim.robots[0]
        return {
            "skills_score": float(self._episode_score),
            "time_remaining": max(0.0, float(self.sim.time_remaining)),
            "episode_steps": self._episode_steps,
            "robot_x": float(robot.body.position.x),
            "robot_y": float(robot.body.position.y),
        }

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        previous_score = float(self.sim.rules_engine.red_score)
        self.sim.step(CONTROL_DT, [self._action(action)])
        self._episode_steps += 1
        current_score = float(self.sim.rules_engine.red_score)
        score_delta = current_score - previous_score
        terminated = bool(self.sim.match_over)
        truncated = False

        # Score changes are the primary objective.  A small terminal bonus
        # keeps evaluation and training aligned without changing simulator
        # scoring.
        reward = float(score_delta)
        if terminated:
            reward += current_score
        self._episode_score = current_score

        if self.render_mode == "human":
            self.sim.render()

        info = self._telemetry()
        info["score_delta"] = score_delta
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> None:
        self.sim.render()

    def close(self) -> None:
        if self.render_mode == "human":
            import pygame

            pygame.quit()

    @property
    def skills_seconds(self) -> float:
        return SKILLS_SECONDS
