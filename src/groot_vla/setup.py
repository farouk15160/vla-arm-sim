from glob import glob

from setuptools import find_packages, setup

package_name = "groot_vla"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/config", glob("config/*.json")),
        # The policy servers are launched by absolute path with a DIFFERENT
        # interpreter (the torch venv), so they are installed as plain data
        # files rather than console_scripts.
        (
            "share/" + package_name + "/servers",
            [
                "groot_vla/mock_policy_server.py",
                "groot_vla/smolvla_server.py",
                "groot_vla/openvla_server.py",
                "groot_vla/groot_client.py",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="farouk",
    maintainer_email="farouk15160@gmail.com",
    description="NVIDIA Isaac GR00T N1.7 VLA bridge for a ROS 2 / MoveIt arm.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "policy_node = groot_vla.policy_node:main",
            "mock_policy_server = groot_vla.mock_policy_server:main",
            "probe_server = groot_vla.probe_server:main",
            "pick_place_demo = groot_vla.pick_place_demo:main",
            "episode_recorder = groot_vla.episode_recorder:main",
            "export_lerobot = groot_vla.export_lerobot:main",
            "scene_reset = groot_vla.scene_reset:main",
            "collect_demos = groot_vla.collect_demos:main",
            "domain_randomizer = groot_vla.domain_randomizer:main",
            "world_publisher = groot_vla.world_publisher:main",
            "goal_marker = groot_vla.goal_marker:main",
            "control_gui = groot_vla.control_gui:main",
        ],
    },
)
