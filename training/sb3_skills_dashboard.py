"""Live PPO dashboard for parallel VEX Robot Skills training.

Run with::

    python -m training.sb3_skills_dashboard

The dashboard runs headless simulator workers in parallel, displays the best
completed skills score and its robot path, and allows training to be stopped
without killing the worker processes.
"""

from __future__ import annotations

import queue
import argparse
import os
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Dict, List

from training.sb3_skills_env import SkillsSB3Env


FIELD_SIZE = 144.0
CANVAS_SIZE = 520


class DashboardCallback:
    """SB3 callback that publishes scores and paths to the Tk main thread."""

    def __init__(self, updates: queue.Queue, stop_event: threading.Event,
                 delay: float, env_count: int):
        from stable_baselines3.common.callbacks import BaseCallback

        owner = self

        class Callback(BaseCallback):
            def __init__(callback):
                super().__init__(verbose=0)
                owner.best_score = 0.0
                owner.best_path = []
                owner.paths = [[] for _ in range(env_count)]
                owner.episodes = 0
                owner.started = time.monotonic()

            def _on_step(callback) -> bool:
                infos = callback.locals.get("infos", [])
                dones = callback.locals.get("dones", [])
                for index, info in enumerate(infos):
                    point = (float(info.get("robot_x", 0.0)),
                             float(info.get("robot_y", 0.0)))
                    if index < len(owner.paths):
                        owner.paths[index].append(point)
                    if index < len(dones) and dones[index]:
                        score = float(info.get("skills_score", 0.0))
                        owner.episodes += 1
                        if score >= owner.best_score:
                            owner.best_score = score
                            owner.best_path = list(
                                owner.paths[index]
                            )
                        owner.paths[index] = []

                now = time.monotonic()
                if now - owner.started >= 0.1:
                    owner.started = now
                    owner.updates.put({
                        "kind": "progress",
                        "steps": callback.num_timesteps,
                        "episodes": owner.episodes,
                        "best_score": owner.best_score,
                        "path": owner.best_path[-500:],
                    })
                if delay > 0:
                    time.sleep(delay)
                return not stop_event.is_set()

            def _on_training_end(callback) -> None:
                owner.updates.put({
                    "kind": "stopped" if stop_event.is_set() else "finished",
                    "best_score": owner.best_score,
                    "path": owner.best_path[-500:],
                })

        self.updates = updates
        self.callback = Callback()


def run_training(env_count: int, timesteps: int, delay: float,
                 updates: queue.Queue, stop_event: threading.Event,
                 model_path: str) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    def make_env(rank: int):
        def factory():
            return Monitor(SkillsSB3Env(seed=rank))

        return factory

    env = (
        DummyVecEnv([make_env(0)])
        if env_count == 1
        else SubprocVecEnv([make_env(i) for i in range(env_count)])
    )
    callback = DashboardCallback(updates, stop_event, delay, env_count)
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        tensorboard_log = None
    else:
        tensorboard_log = "artifacts/logs/sb3_skills"
    model = PPO(
        "MlpPolicy", env, verbose=0, seed=0,
        tensorboard_log=tensorboard_log,
    )
    try:
        model.learn(total_timesteps=timesteps, callback=callback.callback)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
    except Exception as error:
        updates.put({"kind": "error", "message": str(error)})
    finally:
        env.close()


class SkillsDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VEX Skills PPO Training")
        self.root.geometry("1000x650")
        self.updates: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.best_path: List[tuple[float, float]] = []
        self._build_ui()
        self.root.after(100, self._poll_updates)

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill="x")
        self.envs = tk.IntVar(value=4)
        self.timesteps = tk.IntVar(value=100_000)
        self.delay = tk.DoubleVar(value=0.0)
        self.status = tk.StringVar(value="Ready")
        self.score = tk.StringVar(value="Best score: 0")
        self.progress = tk.StringVar(value="Steps: 0   Episodes: 0")
        for label, variable, width in (
            ("Parallel runs", self.envs, 6),
            ("Timesteps", self.timesteps, 10),
            ("Delay (sec)", self.delay, 6),
        ):
            ttk.Label(controls, text=label).pack(side="left", padx=(0, 4))
            ttk.Entry(controls, textvariable=variable, width=width).pack(
                side="left", padx=(0, 12)
            )
        self.start_button = ttk.Button(controls, text="Start", command=self.start)
        self.start_button.pack(side="left", padx=4)
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop,
                                      state="disabled")
        self.stop_button.pack(side="left", padx=4)
        ttk.Label(self.root, textvariable=self.status).pack(anchor="w", padx=12)
        ttk.Label(self.root, textvariable=self.score,
                  font=("TkDefaultFont", 16, "bold")).pack(anchor="w", padx=12)
        ttk.Label(self.root, textvariable=self.progress).pack(anchor="w", padx=12)
        self.canvas = tk.Canvas(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                background="#263746", highlightthickness=0)
        self.canvas.pack(side="left", padx=12, pady=12)
        self.log = tk.Text(self.root, width=55, height=28, state="disabled")
        self.log.pack(side="left", fill="both", expand=True, padx=12, pady=12)
        self._draw_field()

    def _draw_field(self) -> None:
        self.canvas.delete("field")
        self.canvas.create_rectangle(10, 10, CANVAS_SIZE - 10, CANVAS_SIZE - 10,
                                    outline="#d9e2ec", width=2, tags="field")
        for x, y in ((48, 120), (24, 96), (96, 24), (120, 48), (72, 72),
                     (48, 24), (24, 48), (96, 120), (120, 96)):
            px, py = self._point(x, y)
            self.canvas.create_oval(px - 5, py - 5, px + 5, py + 5,
                                    fill="#efb366", outline="", tags="field")
        if len(self.best_path) > 1:
            points = [value for point in self.best_path for value in self._point(*point)]
            self.canvas.create_line(*points, fill="#5eead4", width=2, tags="path")

    @staticmethod
    def _point(x: float, y: float) -> tuple[float, float]:
        scale = (CANVAS_SIZE - 20) / FIELD_SIZE
        return 10 + x * scale, CANVAS_SIZE - 10 - y * scale

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        env_count = max(1, int(self.envs.get()))
        timesteps = max(1, int(self.timesteps.get()))
        delay = max(0.0, float(self.delay.get()))
        self.stop_event.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.set("Training...")
        self.worker = threading.Thread(
            target=run_training,
            args=(env_count, timesteps, delay, self.updates, self.stop_event,
                  "artifacts/models/sb3_skills_dashboard"),
            daemon=True,
        )
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.status.set("Stopping after the current PPO update...")

    def _poll_updates(self) -> None:
        try:
            while True:
                update: Dict[str, Any] = self.updates.get_nowait()
                if update["kind"] == "progress":
                    self.best_path = update["path"]
                    self.score.set(f"Best score: {update['best_score']:.0f}")
                    self.progress.set(
                        f"Steps: {update['steps']:,}   Episodes: {update['episodes']}"
                    )
                    self._append_log(
                        f"steps={update['steps']:,} "
                        f"episodes={update['episodes']} "
                        f"best_score={update['best_score']:.0f}"
                    )
                    self._draw_field()
                else:
                    if update["kind"] == "error":
                        self.status.set(f"Error: {update['message']}")
                    else:
                        self.status.set(update["kind"].capitalize())
                    if update["kind"] != "error":
                        self._append_log(
                            f"{update['kind']}: "
                            f"best_score={update.get('best_score', 0):.0f}"
                        )
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_updates)

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def run(self) -> None:
        self.root.mainloop()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VEX Skills PPO dashboard")
    parser.add_argument("--headless", action="store_true",
                        help="Run PPO with console progress instead of Tkinter")
    parser.add_argument("--envs", type=int, default=4,
                        help="Parallel skills runs")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay in seconds per training step")
    return parser


def _run_headless(args: argparse.Namespace) -> None:
    updates: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    run_training(
        max(1, args.envs), max(1, args.timesteps), max(0.0, args.delay),
        updates, stop_event, "artifacts/models/sb3_skills_dashboard",
    )
    while True:
        try:
            update = updates.get_nowait()
        except queue.Empty:
            break
        if update["kind"] == "progress":
            print(
                f"steps={update['steps']:,} "
                f"episodes={update['episodes']} "
                f"best_score={update['best_score']:.0f}"
            )
        elif update["kind"] == "error":
            print(f"error: {update['message']}")
        else:
            print(
                f"{update['kind']}: "
                f"best_score={update.get('best_score', 0):.0f}"
            )


if __name__ == "__main__":
    arguments = _parser().parse_args()
    if arguments.headless:
        _run_headless(arguments)
    elif not os.environ.get("DISPLAY"):
        print(
            "No graphical display is available. Run with --headless in this "
            "Codespace, or launch from a desktop session to use the Tkinter UI."
        )
    else:
        SkillsDashboard().run()