import json
import matplotlib.pyplot as plt
import numpy as np
import os
from .io import save_pgm

def load_line_segments_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments = data.get('segments', data.get('line_segments', []))
    start_point = data.get('start_point', [0, 0])
    return segments, start_point

def save_pgm(filename, img, maxval=255):
    h, w = img.shape
    with open(filename, 'w') as f:
        f.write(f'P2\n{w} {h} {maxval}\n')
        for row in img:
            f.write(' '.join(str(int(val)) for val in row) + '\n')

def save_png(filename, img):
    plt.imsave(filename, img, cmap='gray', vmin=0, vmax=255)

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

def find_maze_exits(grid):
    h, w = grid.shape
    exits = []
    for x in range(w):
        if grid[0, x] == 0:
            exits.append((x, 0))
        if grid[h-1, x] == 0:
            exits.append((x, h-1))
    for y in range(h):
        if grid[y, 0] == 0:
            exits.append((0, y))
        if grid[y, w-1] == 0:
            exits.append((w-1, y))
    return list(set(exits))

def is_exit(point, grid):
    x, y = point
    h, w = grid.shape
    if (x == 0 or x == w-1 or y == 0 or y == h-1) and grid[y, x] == 0:
        return True
    return False

def grid_to_obstacle_list(grid):
    ob = []
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] == 1:
                ob.append([x, y])
    return ob

def all_reachable_visited(visited, grid):
    return np.all((grid==1) | visited)

def bfs_generate_targets(grid, start, stride=2):
    from collections import deque
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    q = deque()
    q.append(start)
    visited[start[1], start[0]] = True
    targets = []
    while q:
        x, y = q.popleft()
        if (x % stride == 0 and y % stride == 0) and grid[y, x] == 0:
            targets.append((x, y))
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0<=nx<w and 0<=ny<h and not visited[ny, nx] and grid[ny, nx]==0:
                visited[ny, nx] = True
                q.append((nx, ny))
    return targets 