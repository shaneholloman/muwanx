"""Newton's Cradle

Loads MuJoCo's official Newton's cradle model directly from its source repo:
https://github.com/google-deepmind/mujoco/blob/main/model/replicate/newton_cradle.xml
"""

import urllib.request

import mujoco

import mjswan

MODEL_URL = "https://raw.githubusercontent.com/google-deepmind/mujoco/main/model/replicate/newton_cradle.xml"


def main():
    builder = mjswan.Builder()
    project = builder.add_project(name="Physics Experience")
    with urllib.request.urlopen(MODEL_URL) as response:
        xml = response.read().decode()
    spec = mujoco.MjSpec.from_string(xml)
    project.add_scene(spec=spec, name="Newton's Cradle")

    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()
