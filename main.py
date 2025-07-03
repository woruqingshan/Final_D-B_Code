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
        
        # 新增：动画速度控制
        self.gui_refresh_rate = 0.001  # GUI刷新间隔（秒）
        self.animation_speed = 0.05    # A*回程动画速度（秒）

class Maze:
    def __init__(self, segments, start_point, config):
        self.segments = segments
        self.start_point = start_point
        self.config = config
        self.max_x = max([max(seg['start'][0], seg['end'][0]) for seg in segments])
        self.max_y = max([max(seg['start'][1], seg['end'][1]) for seg in segments])
        self.cols = self.max_x + 1  # 修正：索引从0开始，所以大小要+1
        self.rows = self.max_y + 1  # 修正：索引从0开始，所以大小要+1
        self.grid_size = [self.cols, self.rows]
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self._build_maze_grid()
        
        # 新增：扩展迷宫边界，添加障碍
        self._expand_maze_boundaries()

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
        
        # 如果json有明确的边界通道，可在此处单独开放

    def _expand_maze_boundaries(self):
        print(f"迷宫大小: {self.cols} x {self.rows}")
        print(f"起点坐标: {self.start_point}")
        print(f"迷宫范围: x=[0,{self.cols-1}], y=[0,{self.rows-1}]")

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
        # 新增：存储检测到的终点，用于可视化
        self.detected_ends = set()

    def is_near_boundary(self, x, y):
        """判断点(x,y)是否为临近边界点：x=1或x=rows-2或y=1或y=cols-2"""
        return (x == 1 or x == self.maze.cols-2 or y == 1 or y == self.maze.rows-2)

    def is_corner(self, x, y):
        """判断点(x,y)是否在角落"""
        return ((x == 1 or x == self.maze.cols-2) and (y == 1 or y == self.maze.rows-2))

    def check_boundary_exit(self, x, y):
        """检查临近边界点(x,y)是否有边界出口，返回边界出口点"""
        exits = set()
        
        # 1. 判断点是否为临近边界点
        if not self.is_near_boundary(x, y):
            return exits
        
        # 2. 判断是否在角落
        is_corner = self.is_corner(x, y)
        
        
        # 3. 如果在角落，特殊处理
        if is_corner:
            # 左上角落 (x=1, y=1)
            if x == 1 and y == 1:
                # 检查左边界 (0,1) 和上边界 (1,0)
                if self.maze.grid[1, 0] == 0:  # 左边界为空地
                    exits.add((0, 1))
                if self.maze.grid[0, 1] == 0:  # 上边界为空地
                    exits.add((1, 0))
            
            # 右上角落 (x=cols-2, y=1)
            elif x == self.maze.cols - 2 and y == 1:
                # 检查右边界 (cols-1,1) 和上边界 (cols-2,0)
                if self.maze.grid[1, self.maze.cols-1] == 0:  # 右边界为空地
                    exits.add((self.maze.cols-1, 1))
                if self.maze.grid[0, self.maze.cols-2] == 0:  # 上边界为空地
                    exits.add((self.maze.cols-2, 0))
            
            # 左下角落 (x=1, y=rows-2)
            elif x == 1 and y == self.maze.rows - 2:
                # 检查左边界 (0,rows-2) 和下边界 (1,rows-1)
                if self.maze.grid[self.maze.rows-2, 0] == 0:  # 左边界为空地
                    exits.add((0, self.maze.rows-2))
                if self.maze.grid[self.maze.rows-1, 1] == 0:  # 下边界为空地
                    exits.add((1, self.maze.rows-1))
            
            # 右下角落 (x=cols-2, y=rows-2)
            elif x == self.maze.cols - 2 and y == self.maze.rows - 2:
                # 检查右边界 (cols-1,rows-2) 和下边界 (cols-2,rows-1)
                if self.maze.grid[self.maze.rows-2, self.maze.cols-1] == 0:  # 右边界为空地
                    exits.add((self.maze.cols-1, self.maze.rows-2))
                if self.maze.grid[self.maze.rows-1, self.maze.cols-2] == 0:  # 下边界为空地
                    exits.add((self.maze.cols-2, self.maze.rows-1))
        
        # 4. 如果是非角落
        else:
            # x = 1时，判断左边界 (0,y) 是否为空地
            if x == 1:
                if self.maze.grid[y, 0] == 0:  # 左边界为空地
                    exits.add((0, y))
            
            # x = cols-2时，判断右边界 (cols-1,y) 是否为空地
            elif x == self.maze.cols - 2:
                if self.maze.grid[y, self.maze.cols-1] == 0:  # 右边界为空地
                    exits.add((self.maze.cols-1, y))
            
            # y = 1时，判断上边界 (x,0) 是否为空地
            if y == 1:
                if self.maze.grid[0, x] == 0:  # 上边界为空地
                    exits.add((x, 0))
            
            # y = rows-2时，判断下边界 (x,rows-1) 是否为空地
            elif y == self.maze.rows - 2:
                if self.maze.grid[self.maze.rows-1, x] == 0:  # 下边界为空地
                    exits.add((x, self.maze.rows-1))
        
        return exits

    def generate_exploration_goals(self, current_pos):
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
        exits = set()
        current_pos = start_pos
        start_pos_tuple = (self.maze.start_point[0], self.maze.start_point[1])

        while self.maze.get_exploration_rate() < self.config.exploration_threshold:
            goals = self.generate_exploration_goals(current_pos)
            if not goals:
                print("没有可探索的目标点，结束探索")
                break

            found_path = False
            for goal in sorted(goals, key=lambda g: np.hypot(g[0] - current_pos[0], g[1] - current_pos[1])):
                if goal == current_pos:
                    continue
                # 使用A*算法规划路径，实现逐步移动而不是瞬移
                path = self._plan_path(current_pos, goal)
                if path:
                    found_path = True
                    for pos in path[1:]:  # 跳过起点，从第二个点开始
                        x, y = pos
                        
                        # 实时检测：检查当前点是否为临近边界点
                        boundary_exits = self.check_boundary_exit(x, y)
                        if boundary_exits:
                            for exit_point in boundary_exits:
                                if exit_point != start_pos_tuple and exit_point not in exits:
                                    exits.add(exit_point)
                                    print(f"发现出口: {exit_point}")
                                    print(f"检测到终点: {exit_point}")
                                    # 实时添加到检测到的终点集合
                                    self.detected_ends.add(exit_point)
                        
                        # 标记为已探索
                        self.maze.mark_explored(int(x), int(y))
                        current_pos = (x, y)
                        path_history.append(current_pos)
                    break
            if not found_path:
                print("所有边界目标都无法从当前位置到达，探索结束")
                break

        # 探索结束后，如果有检测到的终点，选择最近的作为最终终点
        if self.detected_ends:
            nearest_end = min(self.detected_ends, key=lambda p: np.hypot(p[0] - current_pos[0], p[1] - current_pos[1]))
            if current_pos != nearest_end:
                print(f"探索结束后，当前位置{current_pos}不是终点{nearest_end}，自动移动到终点")
                path = self._plan_path(current_pos, nearest_end)
                if path:
                    for pos in path[1:]:
                        x, y = pos
                        self.maze.mark_explored(int(x), int(y))
                        current_pos = (x, y)
                        path_history.append(current_pos)
                        if (x, y) == nearest_end:
                            print(f"到达终点: {nearest_end}")
                else:
                    print(f"无法从{current_pos}到达终点{nearest_end}")

        return path_history, self.detected_ends

    def get_detected_ends(self):
        """获取检测到的所有终点，用于可视化"""
        return self.detected_ends

    def _plan_path(self, start, goal):
        """使用A*算法规划路径"""
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while not open_set.empty():
            _, current = open_set.get()

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # 反转路径

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                if not self.maze.is_valid(neighbor[0], neighbor[1]):
                    continue

                tentative_g = g_score[current] + np.hypot(dx, dy)  # 使用欧几里得距离

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.config.heuristic_weight * self._heuristic(neighbor, goal)
                    open_set.put((f_score[neighbor], neighbor))

        return []  # 没有找到路径

    def _heuristic(self, a, b):
        """启发式函数：欧几里得距离"""
        return np.hypot(a[0] - b[0], a[1] - b[1])


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
    segments, start_point = load_line_segments_from_json('data/2.json')
    if not segments:
        print("未加载到迷宫线段数据，程序退出")
        return
    print('Maze loaded. Start:', start_point)
    
    # 构建迷宫对象
    config = Config()
    
    # 快速模式：大幅提升动画速度
    #fast_mode = True  # 设置为True启用快速模式
    #if fast_mode:
    #    config.gui_refresh_rate = 0.0001  # 极快刷新
    #    config.animation_speed = 0.01     # 极快动画
    #    print("🚀 快速模式已启用")
    
    maze = Maze(segments, start_point, config)
    explorer = FloodFillExplorer(maze, config)
    # 洪水填充探索
    path, exits = explorer.explore(start_point)
    print(f"\n探索完成！")
    print(f"出口位置: {exits}")
    print(f"最终探索率: {maze.get_exploration_rate()*100:.1f}%")

    # 获取检测到的终点
    detected_ends = explorer.get_detected_ends()
    print(f"检测到的终点: {detected_ends}")

    # 可视化与采集同步
    vis = MazeVisualizer(map_size_pixels=maze.grid.shape[1], map_size_meters=maze.grid.shape[0], 
                        refresh_rate=config.gui_refresh_rate)
    vis.load_line_segments(segments, start_point=start_point)
    # 新增：设置检测到的终点到GUI
    vis.set_detected_ends(detected_ends)
    traj = []
    scan_pts = []
    scan_imgs = []
    maze_exits = set()

    # 初始化SLAM
    slam = SLAMSimulator(map_size_pixels=maze.grid.shape[1], map_size_meters=maze.grid.shape[0])
    slam.set_occupancy_grid((maze.grid==1).astype(np.uint8))
    vis.set_slam_simulator(slam)

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
        # 新增：SLAM右侧激光探索视图实时更新
        pose = [pt[0] * slam.MAP_SIZE_METERS / maze.grid.shape[1], pt[1] * slam.MAP_SIZE_METERS / maze.grid.shape[0], 0]
        vis.update_slam_pose(pose)
        vis.show_full_slam()

    print(f'采集到的出口点: {maze_exits}')
    print(f'检测到的终点: {detected_ends}')
    
    # 优先使用检测到的终点，如果没有则使用采集到的出口点
    if detected_ends:
        # 选择距离最后位置最近的检测到的终点
        last_pt = traj[-1]
        end_pt = min(detected_ends, key=lambda pt: math.hypot(pt[0]-last_pt[0], pt[1]-last_pt[1]))
        end_point = end_pt
        print(f'使用检测到的终点: {end_point}')
    elif maze_exits:
        # 如果没有检测到的终点，使用采集到的出口点
        last_pt = traj[-1]
        end_pt = min(maze_exits, key=lambda pt: math.hypot(pt[0]-last_pt[0], pt[1]-last_pt[1]))
        end_point = end_pt
        print(f'使用采集到的出口点: {end_point}')
    else:
        # 如果都没有，使用路径的最后一个点
        end_point = (int(round(traj[-1][0])), int(round(traj[-1][1])))
        print(f'使用路径终点: {end_point}')
    
    print('End point:', end_point)

    # 到达终点后，A*规划最优路径（终点->起点）
    planner = AStarPlanner(maze.grid, step=0.5)
    a_star_path = planner.planning(tuple(map(float, end_point)), tuple(map(float, start_point)))
    if a_star_path:
        pixel_path = [(int(round(x)), int(round(y))) for (x, y) in a_star_path]
        # 1. 只显示静态蓝色最优路径
        vis.set_trajectory([])
        vis.set_scan_points([])
        vis.set_path(pixel_path)
        vis._end_point = end_point
        vis.show_left()
        time.sleep(0.5)
        # 2. 红色回程轨迹动画（终点->起点）
        return_traj = []
        for p in pixel_path:
            return_traj.append(p)
            vis.set_trajectory([])
            vis.set_scan_points([])
            vis.set_path(pixel_path)
            vis.set_return_trajectory(return_traj)
            vis.show_left()
            time.sleep(config.animation_speed)
    else:
        print('No path found!')

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
        vis.update_slam_pose(pose)
        vis.show_full_slam()
    slam_map = np.array(slam.get_map(), dtype=np.uint8).reshape(slam.MAP_SIZE_PIXELS, slam.MAP_SIZE_PIXELS)
    slam_map_path = os.path.join(PGM_SAVE_PATH, 'slam_result.png')
    plt.imsave(slam_map_path, slam_map, cmap='gray', vmin=0, vmax=255)
    print(f'SLAM结果图已保存为{slam_map_path}')


    print('全部流程结束，所有结果已保存到pgm_outputs。')

if __name__ == '__main__':
    main()
    plt.show(block=True)