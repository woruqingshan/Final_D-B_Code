import json
import numpy as np
import os
from collections import deque
from planning.a_star import AStarPlanner
from utils.gui import MazeVisualizer
from simulation.slam_simulator import SLAMSimulator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time


# PGM保存工具
PGM_SAVE_PATH = 'pgm_outputs'
os.makedirs(PGM_SAVE_PATH, exist_ok=True)

def save_pgm(filename, img, maxval=255):
    h, w = img.shape
    with open(filename, 'w') as f:
        f.write(f'P2\n{w} {h} {maxval}\n')
        for row in img:
            f.write(' '.join(str(int(val)) for val in row) + '\n')

def save_png(filename, img):
    plt.imsave(filename, img, cmap='gray', vmin=0, vmax=255)

def load_maze_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments = data.get('segments', data.get('line_segments', []))
    start_point = tuple(data.get('start_point', [0, 0]))
    # 计算迷宫范围（严格0~max_x, 0~max_y）
    max_x = max(max(seg['start'][0], seg['end'][0]) for seg in segments)
    max_y = max(max(seg['start'][1], seg['end'][1]) for seg in segments)
    grid = np.zeros((max_y+1, max_x+1), dtype=np.uint8)  # 0: free, 1: wall
    for seg in segments:
        x0, y0 = seg['start']
        x1, y1 = seg['end']
        if x0 == x1:
            for y in range(min(y0, y1), max(y0, y1)+1):
                grid[y, x0] = 1
        elif y0 == y1:
            for x in range(min(x0, x1), max(x0, x1)+1):
                grid[y0, x] = 1
        else:
            continue
    return grid, start_point, segments

def bfs_maze(grid, start):
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    q = deque()
    q.append(start)
    visited[start[1], start[0]] = True
    scan_points = []
    trajectory = []
    while q:
        x, y = q.popleft()
        trajectory.append((x, y))
        if (x % 2 == 0 and y % 2 == 0):
            scan_points.append((x, y))
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0<=nx<w and 0<=ny<h and not visited[ny, nx] and grid[ny, nx]==0:
                visited[ny, nx] = True
                q.append((nx, ny))
    return scan_points, trajectory, visited

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

def all_reachable_visited(visited, grid):
    # 只要所有可行走区域都被访问过就算完成
    return np.all((grid==1) | visited)

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
    # 边界且可通行
    if (x == 0 or x == w-1 or y == 0 or y == h-1) and grid[y, x] == 0:
        return True
    return False

def main():
    print('main函数开始执行...')
    grid, start, segments = load_maze_from_json('data/line_segments.json')
    print('Maze loaded. Start:', start)
    vis = MazeVisualizer(map_size_pixels=grid.shape[1], map_size_meters=grid.shape[0])
    vis.load_line_segments(segments, start_point=start)
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    q = deque()
    q.append(start)
    visited[start[1], start[0]] = True
    traj = []
    scan_pts = []
    moves = [(-1,0),(1,0),(0,-1),(0,1)]  # 只允许上下左右
    scan_imgs = []
    potential_end = None
    print('开始BFS遍历整个迷宫...')
    try:
        step = 0
        while q:
            x, y = q.popleft()
            print(f'BFS节点: ({x},{y}), step={step}')
            traj.append((x, y))
            if (x % 2 == 0 and y % 2 == 0):
                scan_pts.append((x, y))
                img = scan_to_pgm(grid, [(x, y)], len(scan_imgs))
                scan_imgs.append(img)
            vis.set_trajectory(traj)
            vis.set_scan_points(scan_pts)
            vis.show()
            # 判断是否为出口
            if is_exit((x, y), grid):
                potential_end = (x, y)
            for dx, dy in moves:
                nx, ny = x+dx, y+dy
                if 0<=nx<w and 0<=ny<h and not visited[ny, nx] and grid[ny, nx]==0:
                    visited[ny, nx] = True
                    q.append((nx, ny))
            step += 1
    except Exception as e:
        print('BFS循环异常:', e)
    print(f'Sampled {len(scan_imgs)} scan points.')
    print('BFS结束，potential_end:', potential_end)
    print('traj长度:', len(traj))
    print('visited总数:', np.sum(visited))
    # 遍历完后，A*移动到终点
    if potential_end is None:
        print('未找到出口，终点设为最后遍历点')
        end_point = traj[-1]
    else:
        print(f'遍历完后终点设为出口: {potential_end}')
        end_point = potential_end
    print('End point:', end_point)
    # 合并所有扫描PGM为灰度图
    fused_img = fuse_scans(scan_imgs)
    fused_path = os.path.join(PGM_SAVE_PATH, 'fused_maze.pgm')
    save_pgm(fused_path, fused_img)
    fused_png_path = os.path.join(PGM_SAVE_PATH, 'fused_maze.png')
    plt.imsave(fused_png_path, fused_img, cmap='gray', vmin=0, vmax=255)
    print(f'Fused maze map saved: {fused_path} and {fused_png_path}')
    # 初始化SLAM
    slam = SLAMSimulator(map_size_pixels=fused_img.shape[1], map_size_meters=fused_img.shape[0])
    slam.set_occupancy_grid((fused_img<128).astype(np.uint8))
    # 动画：当前位置到终点
    if traj[-1] != end_point:
        print('遍历完后当前位置不是终点，A*移动到终点...')
        planner = AStarPlanner(grid)
        path_to_end = planner.planning(traj[-1][::-1], end_point[::-1])
        if path_to_end:
            for p in path_to_end[1:]:
                traj.append(p[::-1])
                if (p[1] % 2 == 0 and p[0] % 2 == 0):
                    img = scan_to_pgm(grid, [(p[1], p[0])], len(scan_imgs))
                    scan_imgs.append(img)
                vis.set_trajectory(traj)
                vis.set_scan_points(scan_pts)
                # SLAM右侧显示
                pose = [p[1] * slam.MAP_SIZE_METERS / fused_img.shape[1], p[0] * slam.MAP_SIZE_METERS / fused_img.shape[0], 0]
                scan = slam.simulate_laser_scan(pose)
                slam.update(scan, (0, 0, 0.1))
                slam_map = np.array(slam.get_map(), dtype=np.uint8).reshape(slam.MAP_SIZE_PIXELS, slam.MAP_SIZE_PIXELS)
                vis.set_pgm_img(slam_map)
                vis.show()
                time.sleep(0.2)
    # 使用SLAM技术对合并图做雷达扫描，保存SLAM地图
    print('使用SLAM技术对合并灰度图做雷达扫描...')
    # 以所有采样点为轨迹，模拟SLAM
    for idx, (x, y) in enumerate(scan_pts):
        pose = [x * slam.MAP_SIZE_METERS / fused_img.shape[1], y * slam.MAP_SIZE_METERS / fused_img.shape[0], 0]
        scan = slam.simulate_laser_scan(pose)
        slam.update(scan, (0, 0, 0.1))
    slam_map = np.array(slam.get_map(), dtype=np.uint8).reshape(slam.MAP_SIZE_PIXELS, slam.MAP_SIZE_PIXELS)
    slam_map_path = os.path.join(PGM_SAVE_PATH, 'slam_result.png')
    plt.imsave(slam_map_path, slam_map, cmap='gray', vmin=0, vmax=255)
    print(f'SLAM结果图已保存为{slam_map_path}')
    # 最后A*从终点回到起点，动态GUI显示红色路径和光标动画


    #存在问题


    print('A*从终点回到起点，GUI动态显示...')
    planner = AStarPlanner(grid)
    path = planner.planning(end_point[::-1], start[::-1])
    if path:
        vis.set_path(path)  # 蓝色最优路径
        vis._end_point = end_point
        return_traj = []
        for i, p in enumerate(path):
            return_traj.append(p[::-1])
            vis.set_trajectory([])  # 不显示橙色轨迹
            vis.set_path(path)      # 蓝色最优路径
            vis.set_return_trajectory(return_traj)  # 红色回程轨迹
            # SLAM右侧显示
            pose = [p[1] * slam.MAP_SIZE_METERS / fused_img.shape[1], p[0] * slam.MAP_SIZE_METERS / fused_img.shape[0], 0]
            scan = slam.simulate_laser_scan(pose)
            slam.update(scan, (0, 0, 0.1))
            slam_map = np.array(slam.get_map(), dtype=np.uint8).reshape(slam.MAP_SIZE_PIXELS, slam.MAP_SIZE_PIXELS)
            vis.set_pgm_img(slam_map)
            vis.show()
            time.sleep(0.2)
    else:
        print('No path found!')
    print('全部流程结束，所有结果已保存到pgm_outputs。')

if __name__ == '__main__':
    main()
    print('即将调用plt.show...')
    plt.show(block=True)
    print('plt.show已返回')