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
from scipy.ndimage import zoom
import math
#from Github_Code.PythonRobotics.PathPlanning.DynamicWindowApproach.dynamic_window_approach import dwa_control, Config, motion
import sys
import os
from queue import PriorityQueue

# ========== Config, Maze, FloodFillExplorer ========== #
class Config:
    def __init__(self):
        self.heuristic_weight = 1.0
        self.exploration_threshold = 1.0
        self.goal_distance_threshold = 2.0

class Maze:
    def __init__(self, segments, start_point, config):
        self.segments = segments
        self.start_point = start_point
        self.config = config
        self.max_x = max([max(seg['start'][0], seg['end'][0]) for seg in segments])
        self.max_y = max([max(seg['start'][1], seg['end'][1]) for seg in segments])
        self.cols = self.max_x
        self.rows = self.max_y
        self.grid_size = [self.cols, self.rows]
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self._build_maze_grid()

    def _build_maze_grid(self):
        for seg in self.segments:
            start = np.array(seg['start'], dtype=int)
            end = np.array(seg['end'], dtype=int)
            if start[0] == end[0]:
                min_y, max_y = sorted([start[1], end[1]])
                for y in range(min_y, max_y + 1):
                    x = start[0]
                    if 0 <= x < self.cols and 0 <= y < self.rows:
                        self.grid[y, x] = 1
            else:
                min_x, max_x = sorted([start[0], end[0]])
                for x in range(min_x, max_x + 1):
                    y = start[1]
                    if 0 <= x < self.cols and 0 <= y < self.rows:
                        self.grid[y, x] = 1
        x, y = self.start_point
        if 0 <= x < self.cols and 0 <= y < self.rows:
            self.grid[y, x] = 3

    def is_valid(self, x, y):
        return 0 <= x < self.cols and 0 <= y < self.rows and self.grid[y, x] != 1

    def mark_explored(self, x, y):
        if 0 <= x < self.cols and 0 <= y < self.rows:
            self.grid[y, x] = 2

    def get_exploration_rate(self):
        total_cells = self.rows * self.cols
        explored_cells = np.sum(self.grid == 2)
        obstacle_cells = np.sum(self.grid == 1)
        return explored_cells / (total_cells - obstacle_cells) if (total_cells - obstacle_cells) > 0 else 0

class FloodFillExplorer:
    def __init__(self, maze, config):
        self.maze = maze
        self.config = config

    def generate_exploration_goals(self):
        goals = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for y in range(self.maze.rows):
            for x in range(self.maze.cols):
                if self.maze.grid[y, x] == 2:
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.maze.cols and 0 <= ny < self.maze.rows:
                            if self.maze.grid[ny, nx] == 0 and (nx, ny) not in goals:
                                goals.append((nx, ny))
        return goals

    def explore(self, start_pos):
        self.maze.mark_explored(int(start_pos[0]), int(start_pos[1]))
        path_history = [start_pos]
        current_pos = start_pos
        start_pos_tuple = (self.maze.start_point[0], self.maze.start_point[1])
        detected_ends = set()
        while self.maze.get_exploration_rate() < 1.0:
            goals = self.generate_exploration_goals()
            if not goals:
                break
            found_path = False
            for goal in sorted(goals, key=lambda g: ((g[0] - current_pos[0])**2 + (g[1] - current_pos[1])**2)):
                if goal == current_pos:
                    continue
                # flood fill直接一步走到目标
                x, y = goal
                self.maze.mark_explored(int(x), int(y))
                current_pos = (x, y)
                path_history.append(current_pos)
                if (int(x) == 0 or int(x) == self.maze.cols-1 or int(y) == 0 or int(y) == self.maze.rows-1) and self.maze.grid[int(y), int(x)] == 0:
                    if (int(x), int(y)) != start_pos_tuple and (int(x), int(y)) not in detected_ends:
                        detected_ends.add((int(x), int(y)))
                found_path = True
                break
            if not found_path:
                break
        all_boundary_ends = set()
        for x in range(self.maze.cols):
            for y in [0, self.maze.rows-1]:
                if self.maze.grid[y, x] == 0 and (x, y) != start_pos_tuple:
                    all_boundary_ends.add((x, y))
        for y in range(self.maze.rows):
            for x in [0, self.maze.cols-1]:
                if self.maze.grid[y, x] == 0 and (x, y) != start_pos_tuple:
                    all_boundary_ends.add((x, y))
        all_boundary_ends.update(detected_ends)
        if all_boundary_ends:
            nearest_end = min(all_boundary_ends, key=lambda p: ((p[0] - current_pos[0])**2 + (p[1] - current_pos[1])**2))
            if current_pos != nearest_end:
                x, y = nearest_end
                self.maze.mark_explored(int(x), int(y))
                current_pos = (x, y)
                path_history.append(current_pos)
        return path_history, all_boundary_ends


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

def bfs_generate_targets(grid, start, stride=2):
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

def grid_to_obstacle_list(grid):
    ob = []
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] == 1:
                ob.append([x, y])
    return np.array(ob)

def is_goal_reached(x, goal, threshold=0.5):
    return np.hypot(x[0] - goal[0], x[1] - goal[1]) <= threshold

def load_line_segments_from_json(json_file_path):
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            segments = data.get('segments', data.get('line_segments', []))
            start_point = data.get('start_point', [0, 0])
            return segments, start_point
        except Exception as e:
            print(f"❌ Failed to load JSON: {e}")
            return [], [0, 0] 

def main():
    # 加载迷宫
    segments, start_point = load_line_segments_from_json('data/line_segments.json')
    if not segments:
        print("未加载到迷宫线段数据，程序退出")
        return
    print('Maze loaded. Start:', start_point)
    # 构建迷宫对象
    config = Config()
    maze = Maze(segments, start_point, config)
    explorer = FloodFillExplorer(maze, config)
    # 洪水填充探索
    path, exits = explorer.explore(start_point)
    print(f"\n探索完成！")
    print(f"出口位置: {exits}")
    print(f"最终探索率: {maze.get_exploration_rate()*100:.1f}%")

    # 可视化与采集同步
    vis = MazeVisualizer(map_size_pixels=maze.grid.shape[1], map_size_meters=maze.grid.shape[0])
    vis.load_line_segments(segments, start_point=start_point)
    traj = []
    scan_pts = []
    scan_imgs = []
    maze_exits = set()

    for idx, pt in enumerate(path):
        traj.append(pt)
        pt_int = (int(round(pt[0])), int(round(pt[1])))
        if pt_int not in scan_pts:
            scan_pts.append(pt_int)
            img = scan_to_pgm(maze.grid, [pt_int], len(scan_imgs))
            scan_imgs.append(img)
            # 判断是否为出口
            x, y = pt_int
            if (x == 0 or x == maze.grid.shape[1] - 1 or y == 0 or y == maze.grid.shape[0] - 1) and maze.grid[y, x] == 0:
                maze_exits.add(pt_int)
        vis.set_trajectory(traj)
        vis.set_scan_points(scan_pts)
        vis.show()

    print(f'采集到的出口点: {maze_exits}')
    # 选maze_exits中距离终点最近的点作为最终出口
    if maze_exits:
        last_pt = traj[-1]
        end_pt = min(maze_exits, key=lambda pt: math.hypot(pt[0]-last_pt[0], pt[1]-last_pt[1]))
        end_point = end_pt
    else:
        end_point = (int(round(traj[-1][0])), int(round(traj[-1][1])))
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
    # SLAM轨迹模拟
    for idx, (x, y) in enumerate(scan_pts):
        pose = [x * slam.MAP_SIZE_METERS / fused_img.shape[1], y * slam.MAP_SIZE_METERS / fused_img.shape[0], 0]
        scan = slam.simulate_laser_scan(pose)
        slam.update(scan, (0, 0, 0.1))
    slam_map = np.array(slam.get_map(), dtype=np.uint8).reshape(slam.MAP_SIZE_PIXELS, slam.MAP_SIZE_PIXELS)
    slam_map_path = os.path.join(PGM_SAVE_PATH, 'slam_result.png')
    plt.imsave(slam_map_path, slam_map, cmap='gray', vmin=0, vmax=255)
    print(f'SLAM结果图已保存为{slam_map_path}')
    
    # 最后A*从终点回到起点，动态GUI显示红色路径和光标动画
    print('A*从终点回到起点，GUI动态显示...')
    step = 0.5
    planner = AStarPlanner(maze.grid, step=step)
    path = planner.planning(tuple(map(float, end_point)), tuple(map(float, start_point)))
    if path:
        vis.set_path([(x/maze.grid.shape[1], y/maze.grid.shape[0]) for (x, y) in path])  # 蓝色最优路径
        vis._end_point = end_point
        return_traj = []
        for i, p in enumerate(path):
            return_traj.append((p[0]/maze.grid.shape[1], p[1]/maze.grid.shape[0]))
            vis.set_trajectory([])  # 不显示橙色轨迹
            vis.set_path([(x/maze.grid.shape[1], y/maze.grid.shape[0]) for (x, y) in path])      # 蓝色最优路径
            vis.set_return_trajectory(return_traj)  # 红色回程轨迹
            # SLAM右侧显示
            pose = [p[0] * slam.MAP_SIZE_METERS / fused_img.shape[1], p[1] * slam.MAP_SIZE_METERS / fused_img.shape[0], 0]
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
    plt.show(block=True)