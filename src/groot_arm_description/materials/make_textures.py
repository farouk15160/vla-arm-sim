"""Generate the procedural textures used by worlds/tabletop.sdf.

Textures are generated rather than downloaded so the repository stays
self-contained and carries no third-party asset licensing. Run this only if you
want to change the look; the PNGs it produces are committed.

    python3 materials/make_textures.py

All maps are 512x512.
"""

from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 512
OUT = Path(__file__).resolve().parent / "textures"
rng = np.random.default_rng(7)


def smooth_noise(shape: tuple[int, int], octaves: int = 4) -> np.ndarray:
    """Value noise: low-frequency layers summed at halving amplitude."""
    total = np.zeros(shape, dtype=np.float64)
    amplitude = 1.0
    for octave in range(octaves):
        cells = 2 ** (octave + 2)
        coarse = rng.random((cells, cells))
        layer = np.asarray(
            Image.fromarray((coarse * 255).astype(np.uint8)).resize(shape, Image.BICUBIC),
            dtype=np.float64,
        ) / 255.0
        total += layer * amplitude
        amplitude *= 0.5
    total -= total.min()
    return total / max(total.max(), 1e-9)


def to_image(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def bench() -> Image.Image:
    """Light melamine lab worktop: fine speckle, subtle warm grey.

    Spatial frequency matters more than prettiness here. The map is stretched
    once across a 1.4 x 1.8 m table, so a feature that is 50 px wide in this
    image ends up ~14 cm wide in the world. Detail therefore has to live at
    high frequency or it reads as blotchy mud rather than a surface. A light
    matte top also keeps the coloured cubes high-contrast for the cameras,
    which is what the policy keys on.
    """
    fine = rng.random((SIZE, SIZE))
    # Slight clumping of the speckle, still high frequency.
    clump = np.asarray(
        Image.fromarray((rng.random((SIZE // 4, SIZE // 4)) * 255).astype(np.uint8))
        .resize((SIZE, SIZE), Image.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    value = 0.62 * fine + 0.38 * clump
    # Very broad shading variation so it is not perfectly uniform.
    value = 0.88 * value + 0.12 * smooth_noise((SIZE, SIZE), 3)

    base = np.stack([
        196 + 26 * value,
        192 + 26 * value,
        183 + 26 * value,
    ], axis=-1)
    return to_image(base)


def wood() -> Image.Image:
    """Oak worktop, kept as an alternative to `bench`.

    ~40 grain cycles across the map so that, stretched over the table, lines
    land roughly 3-4 cm apart. The earlier version used ~5 cycles and looked
    like spilled paint.
    """
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    warp = smooth_noise((SIZE, SIZE), 4) * 1.6
    rings = np.sin(x * 0.49 + warp)
    grain = 0.5 + 0.5 * rings
    value = 0.62 * grain + 0.38 * rng.random((SIZE, SIZE))

    base = np.stack([
        176 + 34 * value,
        146 + 30 * value,
        109 + 24 * value,
    ], axis=-1)
    for _ in range(2):
        cx, cy = rng.integers(0, SIZE, 2)
        r = rng.integers(10, 18)
        d = np.hypot(x - cx, y - cy)
        base *= (1.0 - 0.22 * np.exp(-(d / r) ** 2))[..., None]
    return to_image(base)


def concrete() -> Image.Image:
    """Polished concrete floor: mottled grey, slightly cool."""
    # Mostly high-frequency: this map covers a large floor area.
    value = 0.35 * smooth_noise((SIZE, SIZE), 6) + 0.65 * rng.random((SIZE, SIZE))
    base = np.stack([108 + 42 * value, 110 + 42 * value, 116 + 42 * value], axis=-1)
    return to_image(base)


def brushed_metal() -> Image.Image:
    """Brushed steel for the tray: fine directional streaks."""
    streaks = rng.normal(0.5, 0.16, (SIZE, 1)).repeat(SIZE, axis=1).T
    value = 0.7 * streaks + 0.3 * smooth_noise((SIZE, SIZE), 3)
    base = np.stack([120 + 90 * value, 124 + 90 * value, 130 + 90 * value], axis=-1)
    return to_image(base)


def wall() -> Image.Image:
    """Painted wall: near-uniform, just enough noise to avoid banding."""
    value = smooth_noise((SIZE, SIZE), 5)
    base = np.stack([168 + 14 * value, 172 + 14 * value, 178 + 14 * value], axis=-1)
    return to_image(base)


def height_to_normal(height: np.ndarray, strength: float = 2.5) -> Image.Image:
    """Derive a tangent-space normal map from a height field.

    Ogre2 has no global illumination in this Gazebo build, so the cheapest
    route to surfaces that look like materials rather than coloured cardboard
    is per-pixel relief. A normal map perturbs the shading normal so a flat
    primitive catches light unevenly - which is most of what the eye reads as
    "real" on close-up surfaces like a worktop.

    Encoding is the standard one: RGB = (x, y, z) remapped from [-1, 1] to
    [0, 255], with +z out of the surface (so a flat area is ~(128, 128, 255)).
    """
    height = height.astype(np.float64)
    # np.roll wraps, which keeps the map tileable.
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * strength
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * strength

    normal = np.stack([-dx, -dy, np.ones_like(height)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return to_image((normal * 0.5 + 0.5) * 255.0)


def to_grey(values: np.ndarray, low: float, high: float) -> Image.Image:
    """Single-channel map scaled into [low, high], written as RGB.

    Gazebo reads roughness/AO maps as greyscale but the loader is happiest
    with a normal 3-channel PNG.
    """
    scaled = np.clip(low + values * (high - low), 0.0, 1.0) * 255.0
    return to_image(np.repeat(scaled[..., None], 3, axis=-1))


def luminance(image: Image.Image) -> np.ndarray:
    """Perceptual luminance in [0, 1], used as the height field."""
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # (name, builder, normal strength, roughness range, AO strength)
    # Rougher surfaces get a wider roughness spread; smooth ones stay tight.
    recipes = (
        ("bench", bench, 1.6, (0.55, 0.80), 0.25),
        ("wood", wood, 2.6, (0.40, 0.70), 0.30),
        ("concrete", concrete, 3.2, (0.70, 0.95), 0.40),
        ("metal", brushed_metal, 1.2, (0.15, 0.40), 0.15),
        ("wall", wall, 0.8, (0.85, 0.98), 0.20),
    )
    for name, builder, strength, (r_low, r_high), ao in recipes:
        albedo = builder()
        albedo.save(OUT / f"{name}.png", optimize=True)

        height = luminance(albedo)
        height_to_normal(height, strength).save(OUT / f"{name}_normal.png", optimize=True)
        # Rougher where the surface is darker: grime and pits scatter light.
        to_grey(1.0 - height, r_low, r_high).save(OUT / f"{name}_rough.png", optimize=True)
        # Crude cavity AO: recessed (dark) areas receive less ambient light.
        to_grey(height, 1.0 - ao, 1.0).save(OUT / f"{name}_ao.png", optimize=True)

        total = sum((OUT / f"{name}{suffix}.png").stat().st_size
                    for suffix in ("", "_normal", "_rough", "_ao"))
        print(f"  {name}: albedo + normal + roughness + ao = {total // 1024} KB")


if __name__ == "__main__":
    main()
