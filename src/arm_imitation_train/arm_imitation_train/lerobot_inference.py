from __future__ import annotations

import argparse
import os
import site
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from robot_msgs.msg import MotorState, RobotState


DOF = 5


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    if msg.height <= 0 or msg.width <= 0:
        raise ValueError("invalid image size")

    encoding = msg.encoding.lower()
    if encoding not in {"bgr8", "rgb8", "bgra8", "rgba8"}:
        raise ValueError(f"unsupported encoding: {msg.encoding}")

    channels = 3 if encoding in {"bgr8", "rgb8"} else 4
    expected_row_bytes = msg.width * channels
    if msg.step < expected_row_bytes:
        raise ValueError(f"invalid step {msg.step}, expected >= {expected_row_bytes}")

    frame = np.frombuffer(msg.data, dtype=np.uint8)
    frame = frame[: msg.step * msg.height].reshape((msg.height, msg.step))
    frame = frame[:, :expected_row_bytes].reshape((msg.height, msg.width, channels))

    if encoding == "rgb8":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame.copy()


def build_state_vector(msg: RobotState, state_features: Sequence[str]) -> np.ndarray:
    if len(msg.motor_state) < DOF:
        raise ValueError(f"expected at least {DOF} joints, got {len(msg.motor_state)}")

    chunks: list[np.ndarray] = []
    for feature in state_features:
        values = []
        for index in range(DOF):
            motor = msg.motor_state[index]
            if feature == "q":
                values.append(motor.q)
            elif feature == "dq":
                values.append(motor.dq)
            elif feature == "ddq":
                values.append(motor.ddq)
            elif feature == "tau":
                values.append(motor.tau_est)
            else:
                raise ValueError(f"unsupported state feature: {feature}")
        chunks.append(np.asarray(values, dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def split_paths(raw_value: str) -> list[str]:
    return [part.strip() for part in raw_value.split(";") if part.strip()]


def ensure_python_paths(extra_paths: Sequence[str]) -> None:
    for path_str in extra_paths:
        path = str(path_str).strip()
        if path and path not in sys.path:
            sys.path.insert(0, path)


def ensure_library_paths(extra_paths: Sequence[str]) -> None:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    current_paths = [path for path in existing.split(":") if path]
    prepend_paths: list[str] = []
    for path_str in extra_paths:
        path = str(path_str).strip()
        if path and path not in current_paths and path not in prepend_paths:
            prepend_paths.append(path)
    if prepend_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(prepend_paths + current_paths)


def remove_user_site_packages() -> None:
    user_site = site.getusersitepackages()
    if not user_site:
        return
    normalized_user_site = os.path.normpath(user_site)
    sys.path[:] = [
        path for path in sys.path if os.path.normpath(path) != normalized_user_site
    ]


def import_torch():
    import torch

    return torch


class LeRobotPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("arm_lerobot_policy_node")

        self.model_path = str(self.declare_parameter("model_path", "").value)
        self.image_topic = str(
            self.declare_parameter("image_topic", "/camera/color/image_raw").value
        )
        self.state_topic = str(
            self.declare_parameter("state_topic", "/arm2/_lowState/joint").value
        )
        self.target_topic = str(
            self.declare_parameter("target_topic", "/joint_target_state").value
        )
        self.publish_rate_hz = float(self.declare_parameter("publish_rate_hz", 20.0).value)
        self.state_features = tuple(
            self.declare_parameter("state_features", ["q", "dq"]).value
        )
        self.log_predictions = bool(self.declare_parameter("log_predictions", False).value)
        self.device_override = str(self.declare_parameter("device", "auto").value)
        self.camera_key = str(
            self.declare_parameter("camera_key", "observation.images.front").value
        )
        self.lerobot_site_packages = tuple(
            split_paths(str(self.declare_parameter("lerobot_site_packages", "").value))
        )
        self.lerobot_library_paths = tuple(
            split_paths(str(self.declare_parameter("lerobot_library_paths", "").value))
        )

        if not self.model_path:
            raise RuntimeError("model_path parameter must not be empty")

        remove_user_site_packages()
        ensure_python_paths(self.lerobot_site_packages)
        ensure_library_paths(self.lerobot_library_paths)

        from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        model_dir = Path(self.model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"LeRobot model directory not found: {model_dir}")

        self.policy_config = PreTrainedConfig.from_pretrained(model_dir)
        if self.device_override != "auto":
            self.policy_config.device = self.device_override

        policy_cls = get_policy_class(self.policy_config.type)
        self.policy = policy_cls.from_pretrained(model_dir, config=self.policy_config)

        device_override = {"device_processor": {"device": str(self.policy_config.device)}}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy_config,
            pretrained_path=str(model_dir),
            preprocessor_overrides=device_override,
            postprocessor_overrides=device_override,
        )

        if self.camera_key not in self.policy_config.input_features:
            raise RuntimeError(
                f"camera_key={self.camera_key} not found in input_features={list(self.policy_config.input_features)}"
            )
        if "observation.state" not in self.policy_config.input_features:
            raise RuntimeError("Policy is missing observation.state input")
        if "action" not in self.policy_config.output_features:
            raise RuntimeError("Policy is missing action output")

        expected_image_shape = tuple(self.policy_config.input_features[self.camera_key].shape)
        if len(expected_image_shape) != 3:
            raise RuntimeError(f"Expected 3D image shape, got {expected_image_shape}")
        self.expected_image_shape = expected_image_shape
        self.expected_action_dim = int(self.policy_config.output_features["action"].shape[0])

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        target_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, image_qos
        )
        self.state_sub = self.create_subscription(
            RobotState, self.state_topic, self.state_callback, state_qos
        )
        self.target_pub = self.create_publisher(RobotState, self.target_topic, target_qos)

        self.latest_image: np.ndarray | None = None
        self.latest_state: RobotState | None = None
        self.timer = self.create_timer(
            1.0 / max(self.publish_rate_hz, 1.0), self.publish_prediction
        )

        self.get_logger().info(
            f"LeRobot policy node started. model={self.model_path} "
            f"policy_type={self.policy_config.type} device={self.policy_config.device} "
            f"image_topic={self.image_topic} state_topic={self.state_topic} "
            f"target_topic={self.target_topic}"
        )

    def image_callback(self, msg: Image) -> None:
        try:
            self.latest_image = image_msg_to_bgr(msg)
        except Exception as error:  # noqa: BLE001
            self.get_logger().warn(f"Failed to decode image: {error}")

    def state_callback(self, msg: RobotState) -> None:
        if len(msg.motor_state) < DOF:
            self.get_logger().warn("Received short RobotState message, ignoring.")
            return
        self.latest_state = msg

    def preprocess_image(self, image_bgr: np.ndarray) -> torch.Tensor:
        torch = import_torch()
        expected_c, expected_h, expected_w = self.expected_image_shape
        if expected_c != 3:
            raise RuntimeError(f"Only 3-channel RGB inputs are supported, got C={expected_c}")

        if image_bgr.shape[0] != expected_h or image_bgr.shape[1] != expected_w:
            image_bgr = cv2.resize(image_bgr, (expected_w, expected_h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return torch.from_numpy(chw).unsqueeze(0)

    def publish_prediction(self) -> None:
        if self.latest_image is None or self.latest_state is None:
            return

        try:
            image_tensor = self.preprocess_image(self.latest_image)
            state_vector = build_state_vector(self.latest_state, self.state_features)
            torch = import_torch()
            state_tensor = torch.from_numpy(state_vector).unsqueeze(0)
            processed_batch = self.preprocessor(
                {
                    self.camera_key: image_tensor,
                    "observation.state": state_tensor,
                }
            )
        except Exception as error:  # noqa: BLE001
            self.get_logger().warn(f"Failed to preprocess LeRobot inputs: {error}")
            return

        try:
            torch = import_torch()
            with torch.no_grad():
                policy_action = self.policy.select_action(processed_batch)
                action_tensor = self.postprocessor(policy_action)
        except Exception as error:  # noqa: BLE001
            self.get_logger().warn(f"Failed to run LeRobot policy: {error}")
            return

        torch = import_torch()
        if not isinstance(action_tensor, torch.Tensor):
            self.get_logger().warn(f"Unexpected postprocessor output type: {type(action_tensor)}")
            return

        action = action_tensor.squeeze(0).detach().cpu().numpy()
        if action.ndim != 1:
            self.get_logger().warn(f"Unexpected action shape: {action.shape}")
            return

        publish_dim = min(DOF, action.shape[0], self.expected_action_dim)
        target_msg = RobotState()
        target_msg.motor_state = []
        for joint_index in range(publish_dim):
            motor = MotorState()
            motor.q = float(action[joint_index])
            motor.dq = 0.0
            motor.ddq = 0.0
            motor.tau_est = 0.0
            motor.cur = 0.0
            motor.valid = True
            target_msg.motor_state.append(motor)

        self.target_pub.publish(target_msg)

        if self.log_predictions:
            self.get_logger().info(
                "Published LeRobot target q=%s"
                % np.array2string(action[:publish_dim], precision=4)
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    parse_args()
    rclpy.init()
    node = None
    try:
        node = LeRobotPolicyNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
