# arm_imitation_train

Minimal PyTorch imitation learning package for the ARM dataset layout.

## Supported Dataset Layout

Each episode directory is expected to contain:

- `meta.json`
- `rgb.mp4`
- `rgb_timestamps.csv`
- `states.csv`
- `actions.csv`

The current baseline learns:

`(image_t, q_t, dq_t) -> q_target_t`

The default config enables an episode quality filter that drops very short, low-motion, or interrupted recordings before training.

## Python Dependencies

Install these on the training machine:

```bash
pip install torch torchvision opencv-python numpy pyyaml
```

To convert this dataset into LeRobot format, install LeRobot separately on the machine that runs the converter:

```bash
pip install lerobot
```

## Train

```bash
python3 -m arm_imitation_train.train \
  --config /home/primarymage/WorkFile/arm/src/arm_imitation_train/config/default_config.yaml
```

The default config is tuned for a first local CPU-only smoke test:

- `batch_size: 8`
- `epochs: 3`
- output to `runs/arm_bc_local`

## Convert To LeRobot

The recorder output can be ported into a LeRobotDataset by decoding each episode video, aligning every frame to the nearest `states.csv` and `actions.csv` row, then writing:

- `observation.images.front` from `rgb.mp4`
- `observation.state` from the selected state fields, default `[q, dq]`
- `action` from `actions.csv`, default `[q]`

Example:

```bash
python3 -m arm_imitation_train.convert_to_lerobot \
  --input-root /home/primarymage/WorkFile/arm/dataset \
  --output-root /home/primarymage/WorkFile/arm/lerobot_dataset \
  --repo-id primarymage/arm-food-task \
  --fps 30 \
  --allowed-close-reason success \
  --allowed-close-reason fail
```

Useful options:

- `--dry-run` prints the inferred schema without writing files.
- `--state-features q dq ddq tau` controls how `observation.state` is built.
- `--action-mode q|dq|q_dq` controls how the LeRobot `action` vector is built.
- `--action-time-offset-sec <sec>` keeps the same action alignment offset used by the local loader.
- `--overwrite` replaces an existing output directory.

## Check Dataset

Before the first training run, you can verify that the dataset loads and batches correctly:

```bash
python3 -m arm_imitation_train.check_dataset \
  --config /home/primarymage/WorkFile/arm/src/arm_imitation_train/config/default_config.yaml
```

## Evaluate

```bash
python3 -m arm_imitation_train.eval \
  --config /home/primarymage/WorkFile/arm/src/arm_imitation_train/config/default_config.yaml \
  --checkpoint runs/arm_bc/best.pt
```

## ROS Inference

Update [`inference_config.yaml`](config/inference_config.yaml) with the trained checkpoint path, then run:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch arm_imitation_train imitation_policy_launch.py
```

The policy node subscribes to:

- `/camera/color/image_raw`
- `/arm2/_lowState/joint`

and publishes:

- `/joint_target_state`

## Notes

- The dataset loader aligns each video frame timestamp to the nearest state sample and nearest action sample.
- `states.csv` uses the robot feedback topic.
- `actions.csv` uses `/joint_target_state`.
- The first baseline only predicts the 5D target joint position vector.
