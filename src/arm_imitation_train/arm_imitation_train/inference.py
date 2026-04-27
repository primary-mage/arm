from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from robot_msgs.msg import MotorState, RobotState

from .model import build_model
from .utils import resolve_device


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


class ArmImitationPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("arm_imitation_policy_node")

        self.checkpoint_path = self.declare_parameter("checkpoint_path", "").value
        self.image_topic = self.declare_parameter("image_topic", "/camera/color/image_raw").value
        self.state_topic = self.declare_parameter("state_topic", "/arm2/_lowState/joint").value
        self.target_topic = self.declare_parameter("target_topic", "/joint_target_state").value
        self.publish_rate_hz = float(self.declare_parameter("publish_rate_hz", 20.0).value)
        self.image_size = tuple(
            int(v) for v in self.declare_parameter("image_size", [224, 224]).value
        )
        self.state_features = tuple(
            self.declare_parameter("state_features", ["q", "dq"]).value
        )
        self.frame_id = self.declare_parameter("frame_id", "").value
        self.log_predictions = bool(self.declare_parameter("log_predictions", False).value)

        if not self.checkpoint_path:
            raise RuntimeError("checkpoint_path parameter must not be empty")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.train_config = checkpoint["config"]
        self.device = resolve_device(self.train_config.get("device", "auto"))
        self.model = build_model(self.train_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

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
            f"Imitation policy node started. checkpoint={self.checkpoint_path} "
            f"image_topic={self.image_topic} state_topic={self.state_topic} "
            f"target_topic={self.target_topic} device={self.device}"
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
        resized = cv2.resize(image_bgr, self.image_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return torch.from_numpy(chw).unsqueeze(0)

    def publish_prediction(self) -> None:
        if self.latest_image is None or self.latest_state is None:
            return

        try:
            image_tensor = self.preprocess_image(self.latest_image).to(self.device)
            state_vector = build_state_vector(self.latest_state, self.state_features)
            state_tensor = torch.from_numpy(state_vector).unsqueeze(0).to(self.device)
        except Exception as error:  # noqa: BLE001
            self.get_logger().warn(f"Failed to preprocess inputs: {error}")
            return

        with torch.no_grad():
            prediction = self.model(image_tensor, state_tensor).squeeze(0).cpu().numpy()

        target_msg = RobotState()
        target_msg.motor_state = []
        for joint_index in range(DOF):
            motor = MotorState()
            motor.q = float(prediction[joint_index])
            motor.dq = 0.0
            motor.ddq = 0.0
            motor.tau_est = 0.0
            motor.cur = 0.0
            motor.valid = True
            target_msg.motor_state.append(motor)

        self.target_pub.publish(target_msg)

        if self.log_predictions:
            self.get_logger().info(
                "Published imitation target q=%s" % np.array2string(prediction, precision=4)
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
        node = ArmImitationPolicyNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
