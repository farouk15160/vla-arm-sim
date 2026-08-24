#!/usr/bin/env bash
# Fetch the CC0 PBR texture sets used by worlds/tabletop.sdf from ambientcg.com.
#
# These are NOT committed: they are ~25 MB of third-party JPEGs, and the
# repository stays self-contained without them (materials/make_textures.py
# generates procedural stand-ins). Run this to get the photographic versions.
#
# Licence: everything on ambientcg.com is CC0 (public domain).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/ambientcg"
mkdir -p "$OUT"

ASSETS=(Wood062 Concrete034 Metal032 Plaster001)

for asset in "${ASSETS[@]}"; do
  if [ -d "$OUT/$asset" ]; then
    echo "  $asset: already present, skipping"
    continue
  fi
  echo "  $asset: downloading ..."
  curl -sSL --max-time 300 -o "$OUT/$asset.zip" \
    "https://ambientcg.com/get?file=${asset}_1K-JPG.zip"
  unzip -o -q "$OUT/$asset.zip" -d "$OUT/$asset"
  rm -f "$OUT/$asset.zip"
  # Keep only the maps Gazebo reads; the rest is Blender/MaterialX scaffolding.
  find "$OUT/$asset" -type f \
    \( -name "*.blend" -o -name "*.mtlx" -o -name "*Displacement*" -o -name "*NormalDX*" \) -delete
done

echo "done -> $OUT"
