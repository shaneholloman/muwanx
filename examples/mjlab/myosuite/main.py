"""MJLab Integration Example with mjlab_myosuite

This example demonstrates how to build mjswan projects using mjlab_myosuite setups.
"""

import os

os.environ["MUJOCO_GL"] = "disable"

import mjlab_myochallenge.tasks  # noqa: F401, E402
from mjlab.tasks.registry import list_tasks  # noqa: E402

import mjswan  # noqa: E402


def main():
    builder = mjswan.Builder()
    project = builder.add_project(name="mjlab Examples")

    for task_id in list_tasks():
        if not task_id.startswith("Myosuite"):
            continue
        project.add_scene_mjlab(task_id)

    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()
