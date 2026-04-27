from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_FEATURES = ("q", "dq", "ddq", "tau")


@dataclass(frozen=True)
class EpisodePaths:
    episode_dir: Path
    meta_path: Path
    video_path: Path
    rgb_timestamps_path: Path
    states_path: Path
    actions_path: Path


@dataclass(frozen=True)
class EpisodeSample:
    frame_index: int
    state_index: int
    action_index: int


class EpisodeVideoReader:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        self.current_index = -1

    def read_frame_rgb(self, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            raise ValueError(f"Invalid frame index: {frame_index}")

        if frame_index != self.current_index + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            self.current_index = frame_index - 1

        success, frame_bgr = self.capture.read()
        if not success or frame_bgr is None:
            raise RuntimeError(f"Failed to read frame {frame_index} from {self.video_path}")

        self.current_index = frame_index
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the local ARM episode dataset into LeRobotDataset format."
    )
    parser.add_argument(
        "--input-root",
        default="/home/primarymage/WorkFile/arm/dataset",
        help="Directory that contains episode_* folders.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Destination directory for the converted LeRobot dataset.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="LeRobot repo_id, for example 'primarymage/arm-food-task'.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=0,
        help="Target LeRobot FPS. If omitted, infer from meta.json output_fps or timestamps.",
    )
    parser.add_argument(
        "--robot-type",
        default="arm2",
        help="robot_type metadata stored in the LeRobot dataset.",
    )
    parser.add_argument(
        "--camera-key",
        default="observation.images.front",
        help="LeRobot image/video feature key for the RGB stream.",
    )
    parser.add_argument(
        "--state-features",
        nargs="+",
        choices=SUPPORTED_FEATURES,
        default=["q", "dq"],
        help="State fields concatenated into observation.state.",
    )
    parser.add_argument(
        "--action-mode",
        choices=["q", "dq", "q_dq"],
        default="q",
        help="How to map actions.csv into the LeRobot action vector.",
    )
    parser.add_argument(
        "--action-time-offset-sec",
        type=float,
        default=0.0,
        help="Optional action alignment offset in seconds. Matches the local training loader semantics.",
    )
    parser.add_argument(
        "--task-name",
        default="",
        help="Override the task name for every episode. Defaults to meta.json task.",
    )
    parser.add_argument(
        "--allowed-close-reason",
        action="append",
        default=[],
        help="Only convert episodes whose meta.json close_reason matches this value. Repeat the flag to allow multiple values.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Optional limit for the number of episodes to convert.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Number of LeRobot image writer threads used during conversion.",
    )
    parser.add_argument(
        "--vcodec",
        default="libsvtav1",
        help="Video codec passed to LeRobotDataset.create().",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_root first if it already exists.",
    )
    parser.add_argument(
        "--skip-invalid-episodes",
        action="store_true",
        help="Skip unreadable or malformed episodes instead of aborting the whole conversion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the dataset and print the inferred schema without writing output files.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def nearest_index(sorted_times: list[float], target_time: float) -> int:
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


def infer_axis_count(field_names: list[str], prefix: str) -> int:
    indices = []
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for name in field_names:
        match = pattern.match(name)
        if match:
            indices.append(int(match.group(1)))

    if not indices:
        return 0

    indices = sorted(indices)
    expected = list(range(1, indices[-1] + 1))
    if indices != expected:
        raise ValueError(
            f"Columns for prefix '{prefix}' are not contiguous: found {indices}, expected {expected}"
        )
    return indices[-1]


def extract_vector(row: dict[str, str], prefix: str, dof: int) -> np.ndarray:
    values = [float(row[f"{prefix}{index}"]) for index in range(1, dof + 1)]
    return np.asarray(values, dtype=np.float32)


def build_axis_names(dof: int, feature_names: list[str]) -> list[str]:
    names: list[str] = []
    for feature_name in feature_names:
        names.extend(f"{feature_name}{index}" for index in range(1, dof + 1))
    return names


def build_action_axis_names(dof: int, action_mode: str) -> list[str]:
    if action_mode == "q":
        return [f"q{index}" for index in range(1, dof + 1)]
    if action_mode == "dq":
        return [f"dq{index}" for index in range(1, dof + 1)]
    return [f"q{index}" for index in range(1, dof + 1)] + [
        f"dq{index}" for index in range(1, dof + 1)
    ]


def build_state_vector(row: dict[str, str], state_features: list[str], dof: int) -> np.ndarray:
    chunks = [extract_vector(row, feature_name, dof) for feature_name in state_features]
    return np.concatenate(chunks, axis=0).astype(np.float32)


def build_action_vector(row: dict[str, str], action_mode: str, dof: int) -> np.ndarray:
    if action_mode == "q":
        return extract_vector(row, "q", dof)
    if action_mode == "dq":
        return extract_vector(row, "dq", dof)
    return np.concatenate([extract_vector(row, "q", dof), extract_vector(row, "dq", dof)], axis=0)


def discover_episodes(input_root: Path) -> list[EpisodePaths]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset root does not exist: {input_root}")

    episodes: list[EpisodePaths] = []
    for episode_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if not episode_dir.name.startswith("episode_"):
            continue

        paths = EpisodePaths(
            episode_dir=episode_dir,
            meta_path=episode_dir / "meta.json",
            video_path=episode_dir / "rgb.mp4",
            rgb_timestamps_path=episode_dir / "rgb_timestamps.csv",
            states_path=episode_dir / "states.csv",
            actions_path=episode_dir / "actions.csv",
        )
        missing = [
            str(path.name)
            for path in (
                paths.meta_path,
                paths.video_path,
                paths.rgb_timestamps_path,
                paths.states_path,
                paths.actions_path,
            )
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Episode {episode_dir.name} is missing required files: {', '.join(missing)}"
            )
        episodes.append(paths)

    if not episodes:
        raise RuntimeError(f"No episode_* directories found under {input_root}")
    return episodes


def load_meta(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def filter_episodes(
    episodes: list[EpisodePaths],
    allowed_close_reasons: set[str],
    max_episodes: int,
) -> tuple[list[EpisodePaths], list[str]]:
    selected: list[EpisodePaths] = []
    skipped_reasons: list[str] = []

    for episode in episodes:
        meta = load_meta(episode.meta_path)
        close_reason = str(meta.get("close_reason", ""))
        if allowed_close_reasons and close_reason not in allowed_close_reasons:
            skipped_reasons.append(f"skip {episode.episode_dir.name}: close_reason={close_reason}")
            continue
        selected.append(episode)
        if max_episodes > 0 and len(selected) >= max_episodes:
            break

    if not selected:
        raise RuntimeError("No episodes left after filtering.")
    return selected, skipped_reasons


def infer_fps(selected_episodes: list[EpisodePaths], fps_arg: int) -> int:
    if fps_arg > 0:
        return fps_arg

    meta_fps_values: list[float] = []
    for episode in selected_episodes:
        meta = load_meta(episode.meta_path)
        value = meta.get("output_fps")
        if isinstance(value, (float, int)) and math.isfinite(float(value)) and float(value) > 0.0:
            meta_fps_values.append(float(value))

    if meta_fps_values:
        inferred = int(round(float(np.median(np.asarray(meta_fps_values, dtype=np.float64)))))
        if inferred <= 0:
            raise ValueError(f"Invalid inferred FPS from meta.json: {meta_fps_values}")
        return inferred

    frame_rows = read_csv_rows(selected_episodes[0].rgb_timestamps_path)
    if len(frame_rows) < 2:
        raise RuntimeError("Cannot infer FPS from timestamps with fewer than two frames.")

    frame_times = np.asarray([float(row["t"]) for row in frame_rows], dtype=np.float64)
    frame_deltas = np.diff(frame_times)
    frame_deltas = frame_deltas[frame_deltas > 0.0]
    if frame_deltas.size == 0:
        raise RuntimeError("Cannot infer FPS because all frame timestamp deltas are non-positive.")

    inferred = int(round(1.0 / float(np.median(frame_deltas))))
    if inferred <= 0:
        raise RuntimeError(f"Failed to infer a positive FPS from timestamps in {selected_episodes[0].episode_dir}")
    return inferred


def infer_dataset_schema(
    sample_episode: EpisodePaths,
    state_features: list[str],
    action_mode: str,
    camera_key: str,
) -> tuple[int, tuple[int, int, int], dict]:
    state_rows = read_csv_rows(sample_episode.states_path)
    action_rows = read_csv_rows(sample_episode.actions_path)
    if not state_rows:
        raise RuntimeError(f"Episode has no state rows: {sample_episode.episode_dir}")
    if not action_rows:
        raise RuntimeError(f"Episode has no action rows: {sample_episode.episode_dir}")

    state_header = list(state_rows[0].keys())
    action_header = list(action_rows[0].keys())
    dof = infer_axis_count(state_header, "q")
    if dof <= 0:
        raise RuntimeError(f"Could not infer joint count from {sample_episode.states_path}")

    for feature_name in state_features:
        feature_dof = infer_axis_count(state_header, feature_name)
        if feature_dof != dof:
            raise RuntimeError(
                f"State feature '{feature_name}' has {feature_dof} axes, expected {dof} in {sample_episode.states_path}"
            )

    required_action_features = ["q", "dq"] if action_mode == "q_dq" else [action_mode]
    for feature_name in required_action_features:
        feature_dof = infer_axis_count(action_header, feature_name)
        if feature_dof != dof:
            raise RuntimeError(
                f"Action feature '{feature_name}' has {feature_dof} axes, expected {dof} in {sample_episode.actions_path}"
            )

    video_reader = EpisodeVideoReader(sample_episode.video_path)
    try:
        frame_rgb = video_reader.read_frame_rgb(0)
    finally:
        video_reader.close()

    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise RuntimeError(
            f"Expected RGB image shape (H, W, 3), got {frame_rgb.shape} in {sample_episode.video_path}"
        )

    image_shape = tuple(int(dimension) for dimension in frame_rgb.shape)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_features) * dof,),
            "names": {"axes": build_axis_names(dof, state_features)},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(build_action_axis_names(dof, action_mode)),),
            "names": {"axes": build_action_axis_names(dof, action_mode)},
        },
        camera_key: {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
    }
    return dof, image_shape, features


def build_episode_samples(
    episode: EpisodePaths,
    action_time_offset_sec: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[EpisodeSample]]:
    frame_rows = read_csv_rows(episode.rgb_timestamps_path)
    state_rows = read_csv_rows(episode.states_path)
    action_rows = read_csv_rows(episode.actions_path)

    if not frame_rows or not state_rows or not action_rows:
        raise RuntimeError(
            f"Episode {episode.episode_dir.name} is missing frame/state/action rows and cannot be converted."
        )

    frame_times = [float(row["t"]) for row in frame_rows]
    state_times = [float(row["t"]) for row in state_rows]
    action_times = [float(row["t"]) for row in action_rows]

    samples: list[EpisodeSample] = []
    for row in frame_rows:
        frame_index = int(row["frame_idx"])
        frame_time = float(row["t"])
        state_index = nearest_index(state_times, frame_time)
        action_index = nearest_index(action_times, frame_time + action_time_offset_sec)
        samples.append(
            EpisodeSample(
                frame_index=frame_index,
                state_index=state_index,
                action_index=action_index,
            )
        )

    if samples[-1].frame_index < 0:
        raise RuntimeError(f"Episode {episode.episode_dir.name} has invalid frame indices.")

    return state_rows, action_rows, samples


def resolve_task_name(meta: dict, episode_dir: Path, task_override: str) -> str:
    if task_override:
        return task_override

    task_name = str(meta.get("task", "")).strip()
    if task_name:
        return task_name
    return episode_dir.name


def ensure_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)


def import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except ImportError:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

            return LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import LeRobotDataset. Install a recent lerobot package, for example: pip install lerobot"
            ) from exc


def clear_pending_episode_buffer(dataset) -> None:
    clear_fn = getattr(dataset, "clear_episode_buffer", None)
    if callable(clear_fn):
        clear_fn()


def convert_episode(
    dataset,
    episode: EpisodePaths,
    dof: int,
    image_shape: tuple[int, int, int],
    state_features: list[str],
    action_mode: str,
    action_time_offset_sec: float,
    camera_key: str,
    task_override: str,
) -> tuple[str, int]:
    meta = load_meta(episode.meta_path)
    task_name = resolve_task_name(meta, episode.episode_dir, task_override)
    state_rows, action_rows, samples = build_episode_samples(
        episode,
        action_time_offset_sec=action_time_offset_sec,
    )

    video_reader = EpisodeVideoReader(episode.video_path)
    try:
        for sample in samples:
            frame_rgb = video_reader.read_frame_rgb(sample.frame_index)
            if tuple(int(dimension) for dimension in frame_rgb.shape) != image_shape:
                raise RuntimeError(
                    f"Frame shape mismatch in {episode.video_path}: got {frame_rgb.shape}, expected {image_shape}"
                )

            state_vector = build_state_vector(state_rows[sample.state_index], state_features, dof)
            action_vector = build_action_vector(action_rows[sample.action_index], action_mode, dof)

            dataset.add_frame(
                {
                    "task": task_name,
                    "observation.state": state_vector,
                    "action": action_vector,
                    camera_key: frame_rgb,
                }
            )
    finally:
        video_reader.close()

    dataset.save_episode()
    return task_name, len(samples)


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    allowed_close_reasons = set(args.allowed_close_reason)
    state_features = list(args.state_features)

    episodes = discover_episodes(input_root)
    selected_episodes, skipped_reasons = filter_episodes(
        episodes,
        allowed_close_reasons=allowed_close_reasons,
        max_episodes=args.max_episodes,
    )
    fps = infer_fps(selected_episodes, args.fps)
    dof, image_shape, features = infer_dataset_schema(
        selected_episodes[0],
        state_features=state_features,
        action_mode=args.action_mode,
        camera_key=args.camera_key,
    )

    print(f"input_root={input_root}")
    print(f"selected_episodes={len(selected_episodes)}")
    print(f"inferred_fps={fps}")
    print(f"dof={dof}")
    print(f"image_shape={image_shape}")
    print(f"state_shape={features['observation.state']['shape']}")
    print(f"action_shape={features['action']['shape']}")
    if skipped_reasons:
        print("filtered_episodes:")
        for reason in skipped_reasons:
            print(f"  {reason}")

    if args.dry_run:
        print("dry_run=true, no LeRobot dataset was written.")
        return

    ensure_output_root(output_root, overwrite=args.overwrite)
    LeRobotDataset = import_lerobot_dataset()
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=features,
        root=output_root,
        robot_type=args.robot_type or None,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        vcodec=args.vcodec,
    )

    converted_episodes = 0
    converted_frames = 0
    try:
        for episode in selected_episodes:
            try:
                task_name, frame_count = convert_episode(
                    dataset=dataset,
                    episode=episode,
                    dof=dof,
                    image_shape=image_shape,
                    state_features=state_features,
                    action_mode=args.action_mode,
                    action_time_offset_sec=args.action_time_offset_sec,
                    camera_key=args.camera_key,
                    task_override=args.task_name,
                )
            except Exception as exc:
                clear_pending_episode_buffer(dataset)
                if args.skip_invalid_episodes:
                    print(f"skip {episode.episode_dir.name}: {exc}")
                    continue
                raise

            converted_episodes += 1
            converted_frames += frame_count
            print(
                f"converted {episode.episode_dir.name}: task={task_name} frames={frame_count}"
            )
    finally:
        dataset.finalize()

    print(
        f"conversion_complete output_root={output_root} episodes={converted_episodes} frames={converted_frames}"
    )


if __name__ == "__main__":
    main()
