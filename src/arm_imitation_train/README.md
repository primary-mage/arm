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

## Train

```bash
python3 -m arm_imitation_train.train \
  --config /home/primarymage/WorkFile/arm/src/arm_imitation_train/config/default_config.yaml
```

The default config is tuned for a first local CPU-only smoke test:

- `batch_size: 8`
- `epochs: 3`
- output to `runs/arm_bc_local`

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
