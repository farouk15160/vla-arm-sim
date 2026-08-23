"""Randomise the scene between episodes, so the dataset teaches the task
rather than the decor.

A policy trained on one fixed scene learns that scene: "the graspable thing is
the small red square at pixel (x, y) on a beige table". It then fails the
moment anything changes. Varying the irrelevant parts forces the model to key
on what actually matters - object shape and position relative to the gripper.

What varies per episode:

    lighting     key light colour, direction and intensity, plus ambient level
    table        surface colour, and which texture (if any) it wears
    walls        colour of the room
    objects      shape (box / cylinder / sphere), size, colour and position
    distractors  extra non-target objects that must be ignored

What deliberately does NOT vary: the robot, the camera poses, the tray
position, and the physics. Randomising the observer as well as the observed
makes it much harder to tell whether a failure is the policy or the setup, and
camera extrinsics are something a real deployment would calibrate, not guess.

Everything is applied through Gazebo's runtime services (`create`, `remove`,
`light_config`), so no relaunch is needed between episodes.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

WORLD = "tabletop"
# Gripper aperture is 0.024 m closed to 0.104 m open, so a graspable object has
# to be comfortably inside that. Anything wider simply cannot be picked up and
# would only teach the policy to fail.
MIN_GRASP_SIZE = 0.035
MAX_GRASP_SIZE = 0.055

SHAPES = ("box", "cylinder", "sphere")

COLOUR_NAMES = {
    "red": (0.85, 0.10, 0.10),
    "green": (0.10, 0.75, 0.15),
    "blue": (0.12, 0.25, 0.90),
    "yellow": (0.90, 0.80, 0.10),
    "purple": (0.60, 0.15, 0.80),
    "orange": (0.95, 0.45, 0.05),
    "cyan": (0.10, 0.75, 0.80),
}

TABLE_COLOURS = (
    (0.78, 0.77, 0.74),   # light melamine
    (0.55, 0.42, 0.28),   # wood
    (0.35, 0.37, 0.40),   # dark steel
    (0.85, 0.85, 0.88),   # white lab bench
    (0.30, 0.45, 0.35),   # green cutting mat
    (0.20, 0.22, 0.26),   # near black
)

WALL_COLOURS = (
    (0.60, 0.62, 0.66),
    (0.72, 0.70, 0.65),
    (0.45, 0.50, 0.58),
    (0.80, 0.78, 0.74),
    (0.35, 0.38, 0.42),
)


def _gz(service: str, reqtype: str, request: str, timeout_ms: int = 5000) -> bool:
    """Call a Gazebo service. Returns False rather than raising.

    Always uses the /blocking variant. The asynchronous services return before
    the entity actually exists, so removing the table and immediately spawning
    objects onto it drops them through the floor - they were created in the
    window where the new table had no collision geometry yet.
    """
    result = subprocess.run(
        ["gz", "service", "-s", f"{service}/blocking", "--reqtype", reqtype,
         "--reptype", "gz.msgs.Boolean", "--timeout", str(timeout_ms), "--req", request],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and "true" in result.stdout


def remove_model(name: str, world: str = WORLD) -> bool:
    return _gz(f"/world/{world}/remove", "gz.msgs.Entity",
               f'name: "{name}", type: MODEL')


def model_exists(name: str, world: str = WORLD) -> bool:
    """True if a model of this name is still in the world."""
    result = subprocess.run(["gz", "model", "--list"], capture_output=True,
                            text=True, timeout=10)
    return f"- {name}" in result.stdout


def remove_and_wait(name: str, world: str = WORLD, attempts: int = 5) -> bool:
    """Remove a model and confirm it is gone.

    Even the blocking remove service occasionally returns before the entity has
    left the world, and spawning does not allow renaming, so the next create
    fails on the name. Polling the model list is the only reliable confirmation.
    """
    import time

    for attempt in range(attempts):
        remove_model(name, world)
        if not model_exists(name, world):
            return True
        time.sleep(0.3 * (attempt + 1))
    return not model_exists(name, world)


def spawn_sdf(sdf: str, world: str = WORLD, allow_renaming: bool = False) -> bool:
    # The SDF is embedded in a protobuf text field, so its double quotes have to
    # be escaped or the request fails to parse.
    escaped = sdf.replace('"', '\\"').replace("\n", " ")
    return _gz(f"/world/{world}/create", "gz.msgs.EntityFactory",
               f'sdf: "{escaped}", allow_renaming: {str(allow_renaming).lower()}')


# --------------------------------------------------------------------------- #
# SDF builders
# --------------------------------------------------------------------------- #
def _material(colour: tuple[float, float, float], roughness: float = 0.6,
              metalness: float = 0.0) -> str:
    r, g, b = colour
    return f"""<material>
        <ambient>{r * 0.5:.3f} {g * 0.5:.3f} {b * 0.5:.3f} 1</ambient>
        <diffuse>{r:.3f} {g:.3f} {b:.3f} 1</diffuse>
        <specular>0.2 0.2 0.2 1</specular>
        <pbr><metal>
          <roughness>{roughness:.2f}</roughness>
          <metalness>{metalness:.2f}</metalness>
        </metal></pbr>
      </material>"""


def _geometry(shape: str, size: float) -> str:
    if shape == "box":
        return f"<box><size>{size:.4f} {size:.4f} {size:.4f}</size></box>"
    if shape == "cylinder":
        # Radius from the graspable width, height a little taller so it stands.
        return (f"<cylinder><radius>{size / 2:.4f}</radius>"
                f"<length>{size * 1.1:.4f}</length></cylinder>")
    if shape == "sphere":
        return f"<sphere><radius>{size / 2:.4f}</radius></sphere>"
    raise ValueError(f"unknown shape {shape!r}")


def object_sdf(name: str, shape: str, size: float, colour: tuple[float, float, float],
               position: tuple[float, float, float], mass: float = 0.05) -> str:
    """A graspable object. High friction and soft contact, as in the base world."""
    inertia = mass * size * size / 6.0
    return f"""<sdf version="1.9">
  <model name="{name}">
    <pose>{position[0]:.4f} {position[1]:.4f} {position[2]:.4f} 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>{mass:.4f}</mass>
        <inertia><ixx>{inertia:.6f}</ixx><iyy>{inertia:.6f}</iyy><izz>{inertia:.6f}</izz>
                 <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <collision name="collision">
        <geometry>{_geometry(shape, size)}</geometry>
        <surface>
          <friction><ode><mu>2.0</mu><mu2>2.0</mu2></ode></friction>
          <contact><ode><kp>1e6</kp><kd>1e3</kd><min_depth>0.001</min_depth></ode></contact>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>{_geometry(shape, size)}</geometry>
        {_material(colour, roughness=0.35)}
      </visual>
    </link>
  </model>
</sdf>"""


def table_cloth_sdf(colour: tuple[float, float, float]) -> str:
    """A thin, VISUAL-ONLY slab laid over the real table to recolour it.

    The world's table is never removed. A static model re-created through the
    EntityFactory service comes back without collision geometry in gz-sim, so
    replacing the table drops every object straight through it to the floor.
    (Verified: spawning onto the original table rests at z = 0.620; onto a
    re-spawned one, objects end at z = 0.02.)

    So the collision surface stays exactly as the world file defines it, and
    only the appearance changes. The slab carries no <collision> element at
    all, which is also why it cannot interfere with grasping.
    """
    return f"""<sdf version="1.9">
  <model name="table_cloth">
    <static>true</static>
    <pose>0.50 0 0.6003 0 0 0</pose>
    <link name="link">
      <visual name="visual">
        <geometry><box><size>1.402 1.802 0.004</size></box></geometry>
        {_material(colour, roughness=0.7)}
      </visual>
    </link>
  </model>
</sdf>"""


def wall_sdf(name: str, position: tuple[float, float, float],
             size: tuple[float, float, float],
             colour: tuple[float, float, float]) -> str:
    return f"""<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <pose>{position[0]} {position[1]} {position[2]} 0 0 0</pose>
    <link name="link">
      <visual name="visual">
        <geometry><box><size>{size[0]} {size[1]} {size[2]}</size></box></geometry>
        {_material(colour, roughness=0.95)}
      </visual>
      <collision name="collision">
        <geometry><box><size>{size[0]} {size[1]} {size[2]}</size></box></geometry>
      </collision>
    </link>
  </model>
</sdf>"""


# --------------------------------------------------------------------------- #
@dataclass
class SceneObject:
    """A spawned object and the words that describe it."""

    name: str
    shape: str
    colour_name: str
    size: float
    position: tuple[float, float, float]

    @property
    def description(self) -> str:
        return f"{self.colour_name} {self.shape}"


class DomainRandomizer:
    """Rebuilds the scene's appearance between episodes."""

    WALLS = (
        ("wall_far", (2.10, 0.0, 1.40), (0.06, 6.0, 2.8)),
        ("wall_right", (0.0, -2.00, 1.40), (6.0, 0.06, 2.8)),
        ("wall_back", (-1.60, 0.0, 1.40), (0.06, 6.0, 2.8)),
    )
    # Where objects may be placed: in front of the arm, within comfortable
    # reach, and clear of the tray at (0.50, -0.35).
    X_RANGE = (0.36, 0.60)
    Y_RANGE = (-0.22, 0.26)
    TABLE_TOP_Z = 0.60

    def __init__(self, world: str = WORLD, seed: int | None = None,
                 shapes: tuple[str, ...] = SHAPES) -> None:
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.shapes = tuple(shapes)
        self._spawned: list[str] = []

    # -- lighting ---------------------------------------------------------- #
    def randomize_lighting(self) -> dict:
        """Vary the key light's colour, direction and intensity.

        Lighting is the single most valuable thing to randomise for a
        vision policy: it changes every pixel without changing the task.
        """
        warmth = self.rng.uniform(-0.18, 0.12)
        intensity = self.rng.uniform(0.55, 1.25)
        r = float(np.clip((1.0 + warmth) * intensity, 0.1, 1.5))
        g = float(np.clip((0.96 + warmth * 0.4) * intensity, 0.1, 1.5))
        b = float(np.clip((0.88 - warmth) * intensity, 0.1, 1.5))

        # Keep the elevation steep enough that the table is lit at all.
        azimuth = self.rng.uniform(-np.pi, np.pi)
        elevation = self.rng.uniform(0.55, 1.25)
        direction = (
            float(np.cos(azimuth) * np.cos(elevation)),
            float(np.sin(azimuth) * np.cos(elevation)),
            float(-np.sin(elevation)),
        )

        request = (
            f'name: "key", type: DIRECTIONAL, '
            f'diffuse: {{r: {r:.3f}, g: {g:.3f}, b: {b:.3f}, a: 1.0}}, '
            f'specular: {{r: 0.3, g: 0.3, b: 0.28, a: 1.0}}, '
            f'direction: {{x: {direction[0]:.3f}, y: {direction[1]:.3f}, '
            f'z: {direction[2]:.3f}}}, '
            f'cast_shadows: true'
        )
        _gz(f"/world/{self.world}/light_config", "gz.msgs.Light", request)
        return {"diffuse": (r, g, b), "direction": direction}

    # -- surfaces ---------------------------------------------------------- #
    def randomize_table(self) -> tuple[float, float, float]:
        """Recolour the work surface via an overlay, never by replacing it."""
        colour = TABLE_COLOURS[self.rng.integers(len(TABLE_COLOURS))]
        # Jitter the chosen colour so the model does not just memorise six.
        colour = tuple(
            float(np.clip(c + self.rng.uniform(-0.06, 0.06), 0.05, 0.98)) for c in colour
        )
        remove_model("table_cloth", self.world)
        spawn_sdf(table_cloth_sdf(colour), self.world)
        return colour

    def randomize_walls(self) -> tuple[float, float, float]:
        colour = WALL_COLOURS[self.rng.integers(len(WALL_COLOURS))]
        colour = tuple(
            float(np.clip(c + self.rng.uniform(-0.05, 0.05), 0.05, 0.98)) for c in colour
        )
        for name, position, size in self.WALLS:
            remove_model(name, self.world)
            spawn_sdf(wall_sdf(name, position, size, colour), self.world)
        return colour

    # -- objects ----------------------------------------------------------- #
    def _sample_position(self, taken: list[tuple[float, float]], size: float) -> tuple[float, float, float]:
        """Pick a spot that does not overlap an already-placed object."""
        for _ in range(40):
            x = float(self.rng.uniform(*self.X_RANGE))
            y = float(self.rng.uniform(*self.Y_RANGE))
            if all(np.hypot(x - px, y - py) > size + 0.05 for px, py in taken):
                return (x, y, self.TABLE_TOP_Z + size / 2.0 + 0.004)
        return (x, y, self.TABLE_TOP_Z + size / 2.0 + 0.004)

    #: Upper bound on obj_N names swept when clearing. Removal must not depend
    #: on this process having created them.
    MAX_OBJECTS = 12

    def clear_objects(self) -> None:
        """Remove every randomiser-spawned object.

        Sweeps the whole obj_0..obj_N namespace rather than only what this
        instance created. Each CLI invocation constructs a fresh randomiser, so
        an instance-local list is empty and the previous episode's objects
        survive - and because spawning does not allow renaming, the new ones
        then fail silently on the name collision. The visible symptom is a
        scene whose objects never move however much you randomise.
        """
        for index in range(self.MAX_OBJECTS):
            remove_and_wait(f"obj_{index}", self.world)
        self._spawned.clear()
        # The world file's original cubes, if they are still around.
        for name in ("red_cube", "green_cube", "blue_cube"):
            remove_model(name, self.world)

    def randomize_objects(self, distractors: int = 2) -> tuple[SceneObject, list[SceneObject]]:
        """Spawn one target plus some distractors. Returns (target, others)."""
        self.clear_objects()

        count = 1 + max(distractors, 0)
        colours = list(COLOUR_NAMES)
        self.rng.shuffle(colours)
        taken: list[tuple[float, float]] = []
        objects: list[SceneObject] = []

        for index in range(count):
            shape = self.shapes[self.rng.integers(len(self.shapes))]
            size = float(self.rng.uniform(MIN_GRASP_SIZE, MAX_GRASP_SIZE))
            colour_name = colours[index % len(colours)]
            position = self._sample_position(taken, size)
            taken.append((position[0], position[1]))

            name = f"obj_{index}"
            sdf = object_sdf(name, shape, size, COLOUR_NAMES[colour_name], position)
            # Retry: a stale entity that has not finished being torn down will
            # block the create, and one hiccup should not end a collection run.
            for attempt in range(3):
                if spawn_sdf(sdf, self.world):
                    break
                remove_and_wait(name, self.world)
            else:
                raise RuntimeError(
                    f"failed to spawn {name} after 3 attempts; a leftover entity "
                    "of that name is blocking it (spawning does not allow renaming)"
                )
            self._spawned.append(name)
            objects.append(SceneObject(name, shape, colour_name, size, position))

        return objects[0], objects[1:]

    # -- everything -------------------------------------------------------- #
    def randomize_all(self, distractors: int = 2, settle: float = 1.0) -> dict:
        import time

        lighting = self.randomize_lighting()
        table = self.randomize_table()
        walls = self.randomize_walls()
        # Give the rebuilt table a moment to register its collision before
        # anything is dropped onto it.
        time.sleep(settle)
        target, others = self.randomize_objects(distractors)
        time.sleep(settle)
        return {
            "lighting": lighting,
            "table_colour": table,
            "wall_colour": walls,
            "target": target,
            "distractors": others,
        }


def main(argv: list[str] | None = None) -> None:
    """Randomise the scene once, from the command line.

        ros2 run groot_vla domain_randomizer --distractors 3
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default=WORLD)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--shapes", default=",".join(SHAPES),
                        help="comma-separated subset of box,cylinder,sphere")
    parser.add_argument("--seed", type=int, default=None)
    args, _unknown = parser.parse_known_args(argv)

    randomizer = DomainRandomizer(
        args.world, args.seed, tuple(args.shapes.split(",")))
    scene = randomizer.randomize_all(args.distractors)
    target = scene["target"]
    print(f"target      : {target.description} at "
          f"({target.position[0]:.3f}, {target.position[1]:.3f}) size {target.size:.3f}")
    print(f"distractors : {[o.description for o in scene['distractors']]}")
    print(f"table       : {tuple(round(c, 2) for c in scene['table_colour'])}")
    print(f"light       : {tuple(round(c, 2) for c in scene['lighting']['diffuse'])}")


if __name__ == "__main__":
    main()
