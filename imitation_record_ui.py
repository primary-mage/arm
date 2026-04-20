#!/usr/bin/env python3
import os
import queue
import signal
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


WORKSPACE_DIR = Path(__file__).resolve().parent
STACK_SCRIPT = WORKSPACE_DIR / "run_imitation_record_real.sh"
ROS_SETUP = "/opt/ros/humble/setup.bash"
WS_SETUP = WORKSPACE_DIR / "install/setup.bash"


class CameraPreviewSubscriber(Node):
    def __init__(self, frame_queue: queue.Queue) -> None:
        super().__init__("imitation_record_ui_preview")
        self.frame_queue = frame_queue
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.subscription = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            image_qos,
        )

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self._image_msg_to_bgr(msg)
        except Exception as exc:
            self.frame_queue.put(("status", f"图像转换失败: {exc}"))
            return

        if frame is None or frame.size == 0:
            return

        preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_queue.put(("frame", preview))

    @staticmethod
    def _image_msg_to_bgr(msg: Image):
        if msg.height <= 0 or msg.width <= 0:
            raise ValueError("invalid image size")

        encoding = msg.encoding.lower()
        if encoding not in {"bgr8", "rgb8", "bgra8", "rgba8"}:
            raise ValueError(f"unsupported encoding: {msg.encoding}")

        if encoding in {"bgr8", "rgb8"}:
            channels = 3
        else:
            channels = 4

        expected_row_bytes = msg.width * channels
        if msg.step < expected_row_bytes:
            raise ValueError(
                f"invalid step {msg.step}, expected at least {expected_row_bytes}"
            )

        buffer = np.frombuffer(msg.data, dtype=np.uint8)
        required_size = msg.step * msg.height
        if buffer.size < required_size:
            raise ValueError(
                f"buffer too small: {buffer.size}, expected at least {required_size}"
            )

        frame = buffer[:required_size].reshape((msg.height, msg.step))
        frame = frame[:, :expected_row_bytes].reshape((msg.height, msg.width, channels))

        if encoding == "rgb8":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame.copy()


class ImitationRecordUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Imitation Record UI")
        self.root.geometry("1360x820")

        self.stack_process: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.preview_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.preview_photo: tk.PhotoImage | None = None
        self.preview_thread: threading.Thread | None = None
        self.preview_stop_event = threading.Event()
        self.last_preview_frame = None

        self.task_name_var = tk.StringVar(value="unnamed_task")
        self.dataset_root_var = tk.StringVar(value="dataset")
        self.status_var = tk.StringVar(value="Idle")
        self.camera_status_var = tk.StringVar(value="等待 /camera/color/image_raw")

        self._build_ui()
        self._poll_log_queue()
        self._poll_preview_queue()
        self._start_preview_subscription()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        controls = tk.Frame(self.root, padx=12, pady=12)
        controls.pack(fill=tk.X)

        tk.Label(controls, text="Task").grid(row=0, column=0, sticky="w")
        tk.Entry(controls, textvariable=self.task_name_var, width=28).grid(
            row=0, column=1, sticky="ew", padx=(8, 16)
        )

        tk.Label(controls, text="Dataset Root").grid(row=0, column=2, sticky="w")
        tk.Entry(controls, textvariable=self.dataset_root_var, width=28).grid(
            row=0, column=3, sticky="ew", padx=(8, 0)
        )

        controls.grid_columnconfigure(1, weight=1)
        controls.grid_columnconfigure(3, weight=1)

        buttons = tk.Frame(self.root, padx=12, pady=4)
        buttons.pack(fill=tk.X)

        self.start_stack_button = tk.Button(
            buttons, text="启动驱动与采集栈", width=16, command=self.start_stack
        )
        self.start_stack_button.pack(side=tk.LEFT)

        self.stop_stack_button = tk.Button(
            buttons, text="停止驱动与采集栈", width=16, command=self.stop_stack
        )
        self.stop_stack_button.pack(side=tk.LEFT, padx=(8, 0))

        self.start_episode_button = tk.Button(
            buttons, text="开始当前轨迹录制", width=16, command=self.start_episode
        )
        self.start_episode_button.pack(side=tk.LEFT, padx=(24, 0))

        self.success_button = tk.Button(
            buttons, text="成功", width=10, command=lambda: self.send_event("success")
        )
        self.success_button.pack(side=tk.LEFT, padx=(8, 0))

        self.fail_button = tk.Button(
            buttons, text="失败", width=10, command=lambda: self.send_event("fail")
        )
        self.fail_button.pack(side=tk.LEFT, padx=(8, 0))

        self.stop_episode_button = tk.Button(
            buttons, text="停止当前轨迹录制", width=14, command=lambda: self.send_event("stop")
        )
        self.stop_episode_button.pack(side=tk.LEFT, padx=(8, 0))

        status_frame = tk.Frame(self.root, padx=12, pady=4)
        status_frame.pack(fill=tk.X)
        tk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        tk.Label(status_frame, textvariable=self.status_var, anchor="w").pack(
            side=tk.LEFT, padx=(8, 0)
        )
        tk.Label(status_frame, text="  Camera:").pack(side=tk.LEFT, padx=(24, 0))
        tk.Label(status_frame, textvariable=self.camera_status_var, anchor="w").pack(
            side=tk.LEFT, padx=(8, 0)
        )

        content = tk.Frame(self.root, padx=12, pady=8)
        content.pack(fill=tk.BOTH, expand=True)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=1)

        preview_panel = tk.Frame(content)
        preview_panel.grid(row=0, column=0, sticky="nsew")
        preview_panel.grid_rowconfigure(1, weight=1)
        preview_panel.grid_columnconfigure(0, weight=1)

        tk.Label(preview_panel, text="相机预览").pack(anchor="w")
        self.preview_label = tk.Label(
            preview_panel,
            text="等待相机画面",
            bg="#101820",
            fg="#d8e2dc",
            anchor="center",
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.preview_label.bind("<Configure>", self._on_preview_resize)

        self.log_text = scrolledtext.ScrolledText(
            content, wrap=tk.WORD, font=("Courier", 10)
        )
        self.log_text.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        self.log_text.configure(state=tk.DISABLED)

    def append_log(self, line: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, line)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _poll_log_queue(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.append_log(line)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _poll_preview_queue(self) -> None:
        try:
            while True:
                kind, payload = self.preview_queue.get_nowait()
                if kind == "status":
                    self.camera_status_var.set(str(payload))
                elif kind == "frame":
                    self.last_preview_frame = payload
                    self._update_preview_image(payload)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_preview_queue)

    def _enqueue_line(self, line: str) -> None:
        self.log_queue.put(line)

    def _start_preview_subscription(self) -> None:
        self.preview_thread = threading.Thread(target=self._preview_thread_main, daemon=True)
        self.preview_thread.start()

    def _preview_thread_main(self) -> None:
        rclpy.init(args=None)
        node = CameraPreviewSubscriber(self.preview_queue)
        self.preview_queue.put(("status", "相机订阅已启动，等待图像"))
        try:
            while not self.preview_stop_event.is_set():
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    def _update_preview_image(self, frame_rgb) -> None:
        self.camera_status_var.set("相机图像正常")
        widget_width = max(self.preview_label.winfo_width(), 320)
        widget_height = max(self.preview_label.winfo_height(), 240)
        max_width = max(widget_width - 8, 320)
        max_height = max(widget_height - 8, 240)
        height, width = frame_rgb.shape[:2]
        scale = min(max_width / width, max_height / height)
        resized = cv2.resize(
            frame_rgb,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA,
        )

        ppm_header = f"P6 {resized.shape[1]} {resized.shape[0]} 255 ".encode("ascii")
        ppm_data = ppm_header + resized.tobytes()
        self.preview_photo = tk.PhotoImage(data=ppm_data, format="PPM")
        self.preview_label.configure(image=self.preview_photo, text="")

    def _on_preview_resize(self, _event) -> None:
        if self.last_preview_frame is not None:
            self._update_preview_image(self.last_preview_frame)

    def _reader_thread(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self._enqueue_line(line)
        process.wait()
        self._enqueue_line(f"\n[stack exited with code {process.returncode}]\n")
        self.root.after(0, lambda: self.status_var.set("Stack stopped"))
        self.root.after(0, self._sync_buttons)

    def _sync_buttons(self) -> None:
        running = self.stack_process is not None and self.stack_process.poll() is None
        self.start_stack_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_stack_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        event_state = tk.NORMAL if running else tk.DISABLED
        self.start_episode_button.configure(state=event_state)
        self.success_button.configure(state=event_state)
        self.fail_button.configure(state=event_state)
        self.stop_episode_button.configure(state=event_state)

    def start_stack(self) -> None:
        if self.stack_process is not None and self.stack_process.poll() is None:
            messagebox.showinfo("Stack Running", "The recording stack is already running.")
            return

        if not STACK_SCRIPT.exists():
            messagebox.showerror("Missing Script", f"Cannot find {STACK_SCRIPT}")
            return

        task_name = self.task_name_var.get().strip() or "unnamed_task"
        dataset_root = self.dataset_root_var.get().strip() or "dataset"

        cmd = [
            "bash",
            "-lc",
            (
                f"cd {WORKSPACE_DIR} && "
                f"source {ROS_SETUP} && "
                f"source {WS_SETUP} && "
                f"./run_imitation_record_real.sh --task {self._shell_quote(task_name)} "
                f"--dataset-root {self._shell_quote(dataset_root)} --no-auto-start"
            ),
        ]

        self.append_log(f"\n[starting stack for task={task_name} dataset_root={dataset_root}]\n")
        self.stack_process = subprocess.Popen(
            cmd,
            cwd=WORKSPACE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        threading.Thread(
            target=self._reader_thread, args=(self.stack_process,), daemon=True
        ).start()
        self.status_var.set("Stack running")
        self._sync_buttons()

    def stop_stack(self) -> None:
        if self.stack_process is None or self.stack_process.poll() is not None:
            return

        self.append_log("\n[stopping stack]\n")
        self.status_var.set("Stopping stack")
        os.killpg(os.getpgid(self.stack_process.pid), signal.SIGTERM)

    def start_episode(self) -> None:
        task_name = self.task_name_var.get().strip() or "unnamed_task"
        self.send_event(f"start:{task_name}")

    def send_event(self, event_text: str) -> None:
        if self.stack_process is None or self.stack_process.poll() is not None:
            messagebox.showwarning("Stack Not Running", "Start the stack first.")
            return

        self.append_log(f"[event] {event_text}\n")
        self.status_var.set(f"Sent event: {event_text}")
        threading.Thread(
            target=self._publish_event, args=(event_text,), daemon=True
        ).start()

    def _publish_event(self, event_text: str) -> None:
        cmd = [
            "bash",
            "-lc",
            (
                f"cd {WORKSPACE_DIR} && "
                f"source {ROS_SETUP} && "
                f"source {WS_SETUP} && "
                "ros2 topic pub --once /episode_event std_msgs/msg/String "
                f"\"{{data: '{event_text}'}}\""
            ),
        ]
        completed = subprocess.run(
            cmd,
            cwd=WORKSPACE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.stdout:
            self._enqueue_line(completed.stdout)
        if completed.returncode != 0:
            self._enqueue_line(f"[event publish failed with code {completed.returncode}]\n")

    def _on_close(self) -> None:
        if self.stack_process is not None and self.stack_process.poll() is None:
            if not messagebox.askyesno("Exit", "The stack is still running. Stop it and exit?"):
                return
            self.stop_stack()
        self.preview_stop_event.set()
        self.root.after(300, self.root.destroy)

    @staticmethod
    def _shell_quote(text: str) -> str:
        return "'" + text.replace("'", "'\"'\"'") + "'"


def main() -> None:
    root = tk.Tk()
    ui = ImitationRecordUI(root)
    ui._sync_buttons()
    root.mainloop()


if __name__ == "__main__":
    main()
