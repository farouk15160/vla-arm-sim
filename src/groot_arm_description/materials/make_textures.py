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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder in (("bench", bench), ("wood", wood), ("concrete", concrete),
                          ("metal", brushed_metal), ("wall", wall)):
        path = OUT / f"{name}.png"
        builder().save(path, optimize=True)
        print(f"  {path.name}: {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
