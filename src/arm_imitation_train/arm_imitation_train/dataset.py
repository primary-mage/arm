from __future__ import annotations

import csv
import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


DOF = 5


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _nearest_index(sorted_times: list[float], target_time: float) -> int:
    if not sorted_times:
        raise ValueError("Cannot align against an empty timestamp list.")
    insert_index = bisect_left(sorted_times, target_time)
    if insert_index <= 0:
        return 0
    if insert_index >= len(sorted_times):
        return len(sorted_times) - 1

    left_index = insert_index - 1
    right_index = insert_index
    if abs(sorted_times[left_index] - target_time) <= abs(sorted_times[right_index] - target_time):
        return left_index
    return right_index


def _extract_vector(row: dict[str, str], prefix: str) -> np.ndarray:
    values = [float(row[f"{prefix}{index}"]) for index in range(1, DOF + 1)]
    return np.asarray(values, dtype=np.float32)


@dataclass
class SampleIndex:
    episode_dir: Path
    frame_index: int
    frame_time: float
    state_index: int
    action_index: int


@dataclass
class QualityFilterConfig:
    enabled: bool = False
    min_frames: int = 0
    min_states: int = 0
    min_actions: int = 0
    min_duration_sec: float = 0.0
    min_action_total_motion: float = 0.0
    allowed_close_reasons: tuple[str, ...] = tuple()


class EpisodeAlignment:
    def __init__(self, episode_dir: Path, action_time_offset_sec: float) -> None:
        self.episode_dir = episode_dir
        self.meta_path = episode_dir / "meta.json"
        self.video_path = episode_dir / "rgb.mp4"
        self.rgb_timestamps_path = episode_dir / "rgb_timestamps.csv"
        self.states_path = episode_dir / "states.csv"
        self.actions_path = episode_dir / "actions.csv"

        if not self.video_path.exists():
            raise FileNotFoundError(f"Missing episode video: {self.video_path}")

        with self.meta_path.open("r", encoding="utf-8") as handle:
            self.meta = json.load(handle)

        self.frame_rows = _read_csv_rows(self.rgb_timestamps_path)
        self.state_rows = _read_csv_rows(self.states_path)
        self.action_rows = _read_csv_rows(self.actions_path)

        self.frame_times = [float(row["t"]) for row in self.frame_rows]
        self.state_times = [float(row["t"]) for row in self.state_rows]
        self.action_times = [float(row["t"]) for row in self.action_rows]
        self.action_time_offset_sec = action_time_offset_sec
        self.duration_sec = self._compute_meta_duration_sec()
        self.action_total_motion = self._compute_action_total_motion()

        self.samples: list[SampleIndex] = []
        self._build_sample_index()

    def _compute_meta_duration_sec(self) -> float:
        start_time = float(self.meta.get("start_time_sec", 0.0))
        end_time = float(self.meta.get("end_time_sec", 0.0))
        return max(0.0, end_time - start_time)

    def _compute_action_total_motion(self) -> float:
        total_motion = 0.0
        previous_q: np.ndarray | None = None
        for row in self.action_rows:
            current_q = _extract_vector(row, "q")
            if previous_q is not None:
                total_motion += float(np.linalg.norm(current_q - previous_q))
            previous_q = current_q
        return total_motion

    def _build_sample_index(self) -> None:
        if not self.frame_times or not self.state_times or not self.action_times:
            return

        for row in self.frame_rows:
            frame_index = int(row["frame_idx"])
            frame_time = float(row["t"])
            state_index = _nearest_index(self.state_times, frame_time)
            action_index = _nearest_index(
                self.action_times, frame_time + self.action_time_offset_sec
            )
            self.samples.append(
                SampleIndex(
                    episode_dir=self.episode_dir,
                    frame_index=frame_index,
                    frame_time=frame_time,
                    state_index=state_index,
                    action_index=action_index,
                )
            )


class VideoFrameReader:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        self.current_index = -1

    def read_frame(self, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            raise ValueError(f"Invalid frame index: {frame_index}")

        if frame_index != self.current_index + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            self.current_index = frame_index - 1

        success, frame = self.capture.read()
        if not success or frame is None:
            raise RuntimeError(
                f"Failed to read frame {frame_index} from {self.video_path}"
            )
        self.current_index = frame_index
        return frame

    def close(self) -> None:
        self.capture.release()


class ArmEpisodeDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        train_split: float = 0.8,
        image_size: tuple[int, int] = (224, 224),
        state_features: tuple[str, ...] = ("q", "dq"),
        target_mode: str = "q",
        action_time_offset_sec: float = 0.0,
        max_episodes: int = 0,
        quality_filter: QualityFilterConfig | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.train_split = train_split
        self.image_size = image_size
        self.state_features = state_features
        self.target_mode = target_mode
        self.action_time_offset_sec = action_time_offset_sec
        self.max_episodes = max_episodes
        self.quality_filter = quality_filter or QualityFilterConfig()

        self.alignments: list[EpisodeAlignment] = []
        self.sample_index: list[SampleIndex] = []
        self.frame_readers: dict[Path, VideoFrameReader] = {}

        self._load_alignments()

    def _load_alignments(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

        episode_dirs = sorted(
            path for path in self.root_dir.iterdir() if path.is_dir() and path.name.startswith("episode_")
        )
        if self.max_episodes > 0:
            episode_dirs = episode_dirs[: self.max_episodes]

        if not episode_dirs:
            raise RuntimeError(f"No episode directories found under {self.root_dir}")

        split_index = max(1, int(len(episode_dirs) * self.train_split))
        if self.split == "train":
            selected_dirs = episode_dirs[:split_index]
        elif self.split == "val":
            selected_dirs = episode_dirs[split_index:] or episode_dirs[-1:]
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        for episode_dir in selected_dirs:
            alignment = EpisodeAlignment(episode_dir, self.action_time_offset_sec)
            if not alignment.samples:
                continue

            filter_reasons = self._quality_filter_reasons(alignment)
            if filter_reasons:
                print(
                    f"[dataset:{self.split}] skip {episode_dir.name} because {', '.join(filter_reasons)}",
                    flush=True,
                )
                continue

            self.alignments.append(alignment)
            self.sample_index.extend(alignment.samples)

        if not self.sample_index:
            raise RuntimeError(f"No aligned samples were built for split={self.split}")

    def _quality_filter_reasons(self, alignment: EpisodeAlignment) -> list[str]:
        cfg = self.quality_filter
        if not cfg.enabled:
            return []

        reasons: list[str] = []
        if cfg.allowed_close_reasons:
            close_reason = str(alignment.meta.get("close_reason", ""))
            if close_reason not in cfg.allowed_close_reasons:
                reasons.append(f"close_reason={close_reason}")
        if len(alignment.frame_rows) < cfg.min_frames:
            reasons.append(f"frames<{cfg.min_frames}")
        if len(alignment.state_rows) < cfg.min_states:
            reasons.append(f"states<{cfg.min_states}")
        if len(alignment.action_rows) < cfg.min_actions:
            reasons.append(f"actions<{cfg.min_actions}")
        if alignment.duration_sec < cfg.min_duration_sec:
            reasons.append(f"duration<{cfg.min_duration_sec}")
        if alignment.action_total_motion < cfg.min_action_total_motion:
            reasons.append(f"action_motion<{cfg.min_action_total_motion}")
        return reasons

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.sample_index[index]
        alignment = next(
            alignment for alignment in self.alignments if alignment.episode_dir == sample.episode_dir
        )

        frame = self._read_episode_frame(sample.episode_dir, sample.frame_index)
        image_tensor = self._preprocess_frame(frame)

        state_row = alignment.state_rows[sample.state_index]
        action_row = alignment.action_rows[sample.action_index]

        state_vector = self._build_state_vector(state_row)
        action_vector = self._build_action_vector(action_row)

        return {
            "image": image_tensor,
            "state": torch.from_numpy(state_vector),
            "action": torch.from_numpy(action_vector),
        }

    def _read_episode_frame(self, episode_dir: Path, frame_index: int) -> np.ndarray:
        reader = self.frame_readers.get(episode_dir)
        if reader is None:
            reader = VideoFrameReader(episode_dir / "rgb.mp4")
            self.frame_readers[episode_dir] = reader
        return reader.read_frame(frame_index)

    def _preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(frame_bgr, self.image_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return torch.from_numpy(chw)

    def _build_state_vector(self, row: dict[str, str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for feature_name in self.state_features:
            if feature_name not in {"q", "dq", "ddq", "tau"}:
                raise ValueError(f"Unsupported state feature: {feature_name}")
            chunks.append(_extract_vector(row, feature_name))
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def _build_action_vector(self, row: dict[str, str]) -> np.ndarray:
        if self.target_mode == "q":
            return _extract_vector(row, "q")
        if self.target_mode == "dq":
            return _extract_vector(row, "dq")
        if self.target_mode == "q_dq":
            return np.concatenate([_extract_vector(row, "q"), _extract_vector(row, "dq")], axis=0)
        raise ValueError(f"Unsupported target_mode: {self.target_mode}")

    def close(self) -> None:
        for reader in self.frame_readers.values():
            reader.close()
        self.frame_readers.clear()

    def __del__(self) -> None:
        self.close()


def build_dataloaders(config: dict[str, Any]):
    from torch.utils.data import DataLoader

    dataset_cfg = config["dataset"]
    training_cfg = config["training"]

    image_size = tuple(dataset_cfg["image_size"])
    state_features = tuple(dataset_cfg["state_features"])

    quality_cfg = dataset_cfg.get("quality_filter", {})
    quality_filter = QualityFilterConfig(
        enabled=bool(quality_cfg.get("enabled", False)),
        min_frames=int(quality_cfg.get("min_frames", 0)),
        min_states=int(quality_cfg.get("min_states", 0)),
        min_actions=int(quality_cfg.get("min_actions", 0)),
        min_duration_sec=float(quality_cfg.get("min_duration_sec", 0.0)),
        min_action_total_motion=float(quality_cfg.get("min_action_total_motion", 0.0)),
        allowed_close_reasons=tuple(quality_cfg.get("allowed_close_reasons", [])),
    )

    train_dataset = ArmEpisodeDataset(
        root_dir=dataset_cfg["root_dir"],
        split="train",
        train_split=float(dataset_cfg["train_split"]),
        image_size=image_size,
        state_features=state_features,
        target_mode=dataset_cfg["target_mode"],
        action_time_offset_sec=float(dataset_cfg["action_time_offset_sec"]),
        max_episodes=int(dataset_cfg["max_episodes"]),
        quality_filter=quality_filter,
    )
    val_dataset = ArmEpisodeDataset(
        root_dir=dataset_cfg["root_dir"],
        split="val",
        train_split=float(dataset_cfg["train_split"]),
        image_size=image_size,
        state_features=state_features,
        target_mode=dataset_cfg["target_mode"],
        action_time_offset_sec=float(dataset_cfg["action_time_offset_sec"]),
        max_episodes=int(dataset_cfg["max_episodes"]),
        quality_filter=quality_filter,
    )

    common_kwargs = {
        "batch_size": int(training_cfg["batch_size"]),
        "num_workers": int(training_cfg["num_workers"]),
        "pin_memory": False,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    return train_dataset, val_dataset, train_loader, val_loader
