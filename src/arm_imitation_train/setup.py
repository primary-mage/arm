from setuptools import find_packages, setup


package_name = "arm_imitation_train"


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/config",
            ["config/default_config.yaml", "config/inference_config.yaml"],
        ),
        (f"share/{package_name}/launch", ["launch/imitation_policy_launch.py"]),
        (f"share/{package_name}", ["README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="primarymage",
    maintainer_email="3188195224@qq.com",
    description="Minimal PyTorch imitation learning package for the ARM dataset format.",
    license="TODO: License declaration",
    entry_points={
        "console_scripts": [
            "arm_imitation_train = arm_imitation_train.train:main",
            "arm_imitation_eval = arm_imitation_train.eval:main",
            "arm_imitation_policy = arm_imitation_train.inference:main",
        ],
    },
)
