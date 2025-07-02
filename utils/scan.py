import numpy as np
import os
from .io import save_pgm

PGM_SAVE_PATH = 'pgm_outputs'
os.makedirs(PGM_SAVE_PATH, exist_ok=True)

def simulate_lidar(grid, x, y, n_beams=36, max_range=10):
    h, w = grid.shape
    scan = []
    for i in range(n_beams):
        angle = 2 * np.pi * i / n_beams
        for r in np.linspace(0, max_range, int(max_range*5)):
            nx = int(round(x + r * np.cos(angle)))
            ny = int(round(y + r * np.sin(angle)))
            if 0<=nx<w and 0<=ny<h:
                if grid[ny, nx] == 1:
                    scan.append((nx, ny))
                    break
            else:
                break
    return scan

def scan_to_pgm(grid, scan_points, idx):
    img = np.ones_like(grid, dtype=np.uint8) * 255
    img[grid==1] = 0
    for x, y in scan_points:
        scan = simulate_lidar(grid, x, y)
        for sx, sy in scan:
            img[sy, sx] = 128
    save_pgm(os.path.join(PGM_SAVE_PATH, f'scan_{idx}.pgm'), img)
    return img

def fuse_scans(scans):
    fused = np.min(np.stack(scans, axis=0), axis=0)
    return fused 