#!/usr/bin/env python3
"""Generate a photomosaic of the Fastly tachometer logo from Fastly GitHub org member avatars.

Only members whose org membership is public are included. Each avatar is used
exactly once, with no modifications. The logo emerges subtly from strategic
placement of brighter vs darker avatars.
"""

import json
import math
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

# --- Config ---
TILE_SIZE = 40
GRID_COLS = 30
GRID_ROWS = 12
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
LOGO_PATH = SCRIPT_DIR / "fastly-wordmark.png"
AVATAR_DIR = Path("/tmp/montage-avatars")
OUTPUT_PATH = SCRIPT_DIR / ".." / "images" / "montage.png"
BG_COLOR = (255, 255, 255)


def fetch_members():
    """Fetch public org members via gh CLI."""
    print("Fetching public members...")
    result = subprocess.run(
        ["gh", "api", "/orgs/fastly/public_members", "--paginate"],
        capture_output=True, text=True, check=True,
    )
    raw = result.stdout.strip()
    try:
        members = json.loads(raw)
    except json.JSONDecodeError:
        members = []
        arrays = re.findall(r'\[.*?\]', raw, re.DOTALL)
        for arr in arrays:
            members.extend(json.loads(arr))
    print(f"  Found {len(members)} public members")
    return [(m["login"], m["avatar_url"]) for m in members]


def download_avatar(login, avatar_url):
    """Download a single avatar."""
    AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    path = AVATAR_DIR / f"{login}.png"
    if path.exists():
        return path
    url = f"{avatar_url}&s=64" if "?" in avatar_url else f"{avatar_url}?s=64"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def download_all(members):
    """Download all avatars with thread pool."""
    print(f"Downloading avatars (up to {len(members)})...")
    done = 0
    total = len(members)
    paths = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(download_avatar, login, url): login for login, url in members}
        for fut in as_completed(futures):
            login = futures[fut]
            done += 1
            try:
                paths[login] = fut.result()
            except Exception as e:
                print(f"  Failed {login}: {e}")
            if done % 50 == 0 or done == total:
                print(f"  Downloaded {done}/{total}")
    return paths


def _color_entropy(img):
    """Compute Shannon entropy of the color distribution of an RGB image."""
    pixels = list(img.getdata())
    total = len(pixels)
    freq = {}
    for p in pixels:
        freq[p] = freq.get(p, 0) + 1
    entropy = 0.0
    for count in freq.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy


def _dominant_color_ratio(img, top_n=3):
    """Quantize to 16 colors and return the fraction of pixels covered by the top_n colors."""
    quantized = img.quantize(colors=16, method=Image.Quantize.MEDIANCUT).convert("RGB")
    pixels = list(quantized.getdata())
    total = len(pixels)
    freq = {}
    for p in pixels:
        freq[p] = freq.get(p, 0) + 1
    top_counts = sorted(freq.values(), reverse=True)[:top_n]
    return sum(top_counts) / total


def filter_identicons(paths):
    """Filter out default/identicon avatars using multiple heuristics."""
    usable = {}
    removed = 0
    reasons = {"low_colors": 0, "low_entropy": 0, "dominant_colors": 0}
    for login, path in paths.items():
        try:
            img = Image.open(path).convert("RGB").resize((32, 32))
            unique_colors = len(set(img.getdata()))
            if unique_colors < 200:
                removed += 1
                reasons["low_colors"] += 1
                continue
            entropy = _color_entropy(img)
            if entropy < 4.0:
                removed += 1
                reasons["low_entropy"] += 1
                continue
            dom_ratio = _dominant_color_ratio(img)
            if dom_ratio >= 0.90:
                removed += 1
                reasons["dominant_colors"] += 1
                continue
            usable[login] = path
        except Exception:
            removed += 1
    print(f"Filtered: {len(usable)} usable, {removed} identicons removed")
    print(f"  Removal breakdown: {reasons}")
    return usable


def compute_brightness(img):
    """Compute average perceived brightness of an RGB image."""
    pixels = list(img.getdata())
    n = len(pixels)
    if n == 0:
        return 128.0
    total = sum(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2] for p in pixels)
    return total / n


def compute_grid_size(n_avatars):
    """Find the best grid (rows x cols) that fits all avatars and is roughly square."""
    best = None
    best_waste = float("inf")
    for cols in range(1, n_avatars + 1):
        rows = math.ceil(n_avatars / cols)
        total = rows * cols
        waste = total - n_avatars
        aspect = max(rows, cols) / max(min(rows, cols), 1)
        # Prefer roughly square grids with minimal waste
        if aspect <= 1.3 and waste < best_waste:
            best = (rows, cols)
            best_waste = waste
        if aspect <= 1.3 and waste == best_waste and best is not None:
            # Prefer the one closer to square
            old_aspect = max(best[0], best[1]) / max(min(best[0], best[1]), 1)
            if aspect < old_aspect:
                best = (rows, cols)
    if best is None:
        side = math.ceil(math.sqrt(n_avatars))
        best = (side, side)
    return best


def generate_mosaic(avatar_paths):
    """Generate photomosaic: avatars placed unmodified, positioned to hint at the logo."""
    print("Generating photomosaic...")

    n_avatars = len(avatar_paths)
    rows, cols = GRID_ROWS, GRID_COLS
    total_cells = rows * cols
    empty_cells = total_cells - n_avatars

    print(f"  Avatars: {n_avatars}, Grid: {rows}x{cols} = {total_cells} cells, {empty_cells} empty")

    # Load logo: composite RGBA onto white, then convert to grayscale
    print("  Loading logo...")
    logo_rgba = Image.open(str(LOGO_PATH)).convert("RGBA")
    logo_bg = Image.new("RGBA", logo_rgba.size, (255, 255, 255, 255))
    logo_bg.paste(logo_rgba, mask=logo_rgba)
    logo = logo_bg.convert("L")  # grayscale
    logo_grid = logo.resize((cols, rows), Image.LANCZOS)

    # Get target brightness for each cell
    logo_pixels = list(logo_grid.getdata())
    cell_brightnesses = []
    for idx in range(total_cells):
        r = idx // cols
        c = idx % cols
        target_b = logo_pixels[r * cols + c]
        cell_brightnesses.append((idx, target_b))

    # Load avatars and compute their brightness
    print("  Loading and measuring avatars...")
    avatar_data = []  # list of (path, brightness, tile_image)
    for login, path in avatar_paths.items():
        try:
            img = Image.open(path).convert("RGB").resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
            b = compute_brightness(img)
            avatar_data.append((login, b, img))
        except Exception as e:
            print(f"  Skipping {login}: {e}")

    n_usable = len(avatar_data)
    print(f"  {n_usable} avatar tiles loaded")

    if n_usable == 0:
        print("ERROR: No usable avatar images!")
        sys.exit(1)

    # Recompute empty cells if we lost some avatars during loading
    if n_usable != n_avatars:
        empty_cells = total_cells - n_usable
        print(f"  Adjusted: {n_usable} usable, {empty_cells} empty cells")

    # Sort cells by target brightness
    cell_brightnesses.sort(key=lambda x: x[1])

    # Sort avatars by actual brightness
    avatar_data.sort(key=lambda x: x[1])

    # Assign: match sorted avatars to sorted cells (darkest avatar → darkest cell)
    # Handle empty cells: place them at corners/edges (cells closest to corners)
    # First, identify which cells will be empty
    if empty_cells > 0:
        # Find the cells furthest from center (corners) to leave empty
        center_r = (rows - 1) / 2.0
        center_c = (cols - 1) / 2.0
        all_cells_with_dist = []
        for idx in range(total_cells):
            r = idx // cols
            c = idx % cols
            dist = math.sqrt((r - center_r) ** 2 + (c - center_c) ** 2)
            all_cells_with_dist.append((idx, dist))
        # Sort by distance from center, descending — furthest cells become empty
        all_cells_with_dist.sort(key=lambda x: -x[1])
        empty_set = set(x[0] for x in all_cells_with_dist[:empty_cells])

        # Remove empty cells from the brightness assignment list
        cell_brightnesses = [(idx, b) for idx, b in cell_brightnesses if idx not in empty_set]
        cell_brightnesses.sort(key=lambda x: x[1])

    # Now assign: sorted avatars → sorted cells
    assignments = {}  # cell_idx -> avatar tile image
    for i, (cell_idx, _target_b) in enumerate(cell_brightnesses):
        _login, _ab, tile_img = avatar_data[i]
        assignments[cell_idx] = tile_img

    # Create output image
    out_w = cols * TILE_SIZE
    out_h = rows * TILE_SIZE
    out = Image.new("RGB", (out_w, out_h), BG_COLOR)

    print("  Compositing tiles...")
    for idx in range(total_cells):
        r = idx // cols
        c = idx % cols
        x = c * TILE_SIZE
        y = r * TILE_SIZE
        if idx in assignments:
            out.paste(assignments[idx], (x, y))
        # else: leave as white background

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(OUTPUT_PATH), "PNG")
    print(f"  Output size: {out_w}x{out_h}")
    print(f"Saved to {OUTPUT_PATH}")


def main():
    random.seed(42)
    members = fetch_members()
    paths = download_all(members)
    usable = filter_identicons(paths)
    generate_mosaic(usable)
    print("Done!")


if __name__ == "__main__":
    main()
