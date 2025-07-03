import matplotlib.pyplot as plt
import numpy as np
import json
import math

class MazeVisualizer:
    def __init__(self, map_size_pixels=21, map_size_meters=21.0, title="Maze Visualizer", refresh_rate=0.001):
        self.map_size_pixels = map_size_pixels
        self.map_size_meters = float(map_size_meters)
        self.refresh_rate = refresh_rate  # 新增：刷新速率配置
        self.fig, (self.ax_left, self.ax_right) = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.suptitle(title, fontsize=16)
        self._maze_segments = []
        self._start_point = None
        self._end_point = None
        self._trajectory = None
        self._scan_points = None
        self._path = None
        self._pgm_img = None
        self._return_traj = None
        # 新增：SLAM激光探索掩码
        self._slam_simulator = None
        self._slam_scanned_mask = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)
        self._slam_pose = None
        # 新增：存储检测到的终点，用于可视化
        self._detected_ends = set()
        # 新增：SLAM探索掩码（可行区域）和障碍掩码
        self._slam_explored_mask = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)
        self._slam_obstacle_mask = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)
        # 新增：SLAM地图显示缓存
        self._slam_mapbytes = None
        # 新增：记录被扫描到的障碍线段索引
        self._scanned_obstacle_segments = set()
        # 新增：初始化理论障碍掩码
        self._theoretical_obstacle_mask = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)

    def load_line_segments(self, segments, start_point=None, end_point=None):
        self._maze_segments = segments
        self._start_point = start_point
        self._end_point = end_point
        # 初始化理论障碍掩码
        for seg in self._maze_segments:
            x0, y0 = seg['start']
            x1, y1 = seg['end']
            ix0, iy0 = self.world_to_pixel(x0, y0)
            ix1, iy1 = self.world_to_pixel(x1, y1)
            rr, cc = self.bresenham_line(iy0, ix0, iy1, ix1)
            for y, x in zip(rr, cc):
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        yy, xx = y+dy, x+dx
                        if 0 <= yy < self.map_size_pixels and 0 <= xx < self.map_size_pixels:
                            self._theoretical_obstacle_mask[yy, xx] = True
        # 初始化显示用掩码
        self._slam_obstacle_mask = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)
        self._slam_explored_mask = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)

    def set_trajectory(self, points):
        """设置探索轨迹，points为(x, y)列表，空列表则清空"""
        self.trajectory = points if points else []
        self.show()

    def set_scan_points(self, points):
        """设置激光点，points为(x, y)列表，空列表则清空"""
        self.scan_points = points if points else []
        self.show()

    def set_path(self, points):
        """设置最优路径，points为像素坐标(x, y)列表"""
        self.path = points if points else []
        self.show()

    def set_pgm_img(self, pgm_img):
        self._pgm_img = pgm_img

    def set_return_trajectory(self, points):
        """设置回程轨迹，points为(x, y)列表，红色显示"""
        self.return_trajectory = points if points else []
        self.show()

    def set_slam_simulator(self, slam_simulator):
        self._slam_simulator = slam_simulator
        self._slam_scanned_mask = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)
        self._slam_explored_mask = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)
        self._slam_obstacle_mask = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)
        self._scanned_obstacle_segments = set()

    def update_slam_pose(self, pose):
        self._slam_pose = pose
        if self._slam_simulator is not None and pose is not None:
            scan = self._slam_simulator.simulate_laser_scan(pose)
            # 强制高密度激光
            n_angles = 360
            angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
            x0, y0, theta = pose
            scan_angles = np.linspace(0, 2*np.pi, len(scan), endpoint=False)
            for a in angles:
                # 用线性插值获取距离
                r = np.interp(a, scan_angles, scan)
                r_m = r / 1000.0
                x_end = x0 + r_m * np.cos(a + theta)
                y_end = y0 + r_m * np.sin(a + theta)
                self._mark_laser_path(x0, y0, x_end, y_end)

    # 新增：设置检测到的终点
    def set_detected_ends(self, detected_ends):
        """设置检测到的终点集合，用于可视化"""
        self._detected_ends = detected_ends

    def bresenham_line(self, y0, x0, y1, x1):
        """返回Bresenham算法生成的所有(y, x)像素点列表，自动裁剪到合法像素范围"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        if dx > dy:
            err = dx // 2
            while x != x1:
                points.append((y, x))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy // 2
            while y != y1:
                points.append((y, x))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((y1, x1))
        # 边界裁剪
        max_idx = self.map_size_pixels - 1
        points = [(yy, xx) for (yy, xx) in points if 0 <= yy <= max_idx and 0 <= xx <= max_idx]
        if points:
            rr, cc = zip(*points)
            return np.array(rr), np.array(cc)
        else:
            return np.array([], dtype=int), np.array([], dtype=int)

    def world_to_pixel(self, x, y):
        ix = int(x / self.map_size_meters * self.map_size_pixels)
        iy = int(y / self.map_size_meters * self.map_size_pixels)
        ix = min(max(ix, 0), self.map_size_pixels - 1)
        iy = min(max(iy, 0), self.map_size_pixels - 1)
        return ix, iy

    def _mark_laser_path(self, x0, y0, x1, y1):
        max_range = math.hypot(self.map_size_meters, self.map_size_meters)
        angle = math.atan2(y1 - y0, x1 - x0)
        hit_point, hit_seg_idx = self.find_laser_obstacle_intersection(x0, y0, angle, self._maze_segments, max_range)
        if hit_point is not None:
            x_end, y_end = hit_point
        else:
            x_end, y_end = x1, y1
        ix0, iy0 = self.world_to_pixel(x0, y0)
        ix1, iy1 = self.world_to_pixel(x_end, y_end)
        rr, cc = self.bresenham_line(iy0, ix0, iy1, ix1)
        for y, x in zip(rr, cc):
            if self._theoretical_obstacle_mask[y, x]:
                self._slam_obstacle_mask[y, x] = True
                break
            self._slam_explored_mask[y, x] = True


    def set_multi_paths(self, paths, colors):
        self._multi_paths = (paths, colors)

    def show_left(self):
        """只刷新左侧地图和轨迹，不刷新右侧SLAM画面"""
        self.ax_left.clear()
        for seg in self._maze_segments:
            x0, y0 = seg['start']
            x1, y1 = seg['end']
            self.ax_left.plot([x0, x1], [y0, y1], color='black', linewidth=4)
        if self._start_point is not None:
            self.ax_left.plot(self._start_point[0], self._start_point[1], 'o', color='green', markersize=12, label='Start')
        if self._end_point is not None:
            self.ax_left.plot(self._end_point[0], self._end_point[1], 'o', color='red', markersize=12, label='End')
        # 新增：同步显示当前移动点（不显示轨迹和观测点）
        if hasattr(self, 'trajectory') and self.trajectory and len(self.trajectory) > 0:
            x, y = self.trajectory[-1]
            self.ax_left.plot(x, y, 'o', color='magenta', markersize=18, markeredgecolor='black', label='Current')
        if hasattr(self, 'path') and self.path and len(self.path) > 0:
            xs, ys = zip(*self.path)
            self.ax_left.plot(xs, ys, color='blue', linewidth=2, label='A* Path')
        if hasattr(self, 'return_trajectory') and self.return_trajectory and len(self.return_trajectory) > 1:
            xs, ys = zip(*self.return_trajectory)
            self.ax_left.plot(xs, ys, color='red', linewidth=2, label='Return Trajectory')
            self.ax_left.scatter([xs[-1]], [ys[-1]], 
                    s=300, c='magenta', edgecolors='black', linewidths=2, zorder=10, label='Left Cursor')

        if hasattr(self, '_multi_paths') :
            paths, colors = self._multi_paths
            for idx,p in enumerate(paths):
                if len(p) > 1:
                    xs,ys = zip(*p)
                    self.ax_left.plot(xs, ys, color=colors[idx%len(colors)], linewidth=2, label=f'Path{idx+1}')
        # 新增：绘制检测到的终点（五角星）
        if self._detected_ends:
            end_x = [end[0] for end in self._detected_ends]
            end_y = [end[1] for end in self._detected_ends]
            self.ax_left.scatter(end_x, end_y, color='magenta', s=200, marker='*', 
                               edgecolors='black', linewidths=2, zorder=15, label='Detected Ends')
        self.ax_left.set_title("Maze & Path")
        self.ax_left.set_xlabel('X')
        self.ax_left.set_ylabel('Y')
        self.ax_left.set_aspect('equal')
        self.ax_left.legend(loc='upper right')
        self.ax_left.grid(True)
        plt.draw()
        plt.pause(0.0001)  # 强制极快刷新
        
    

    def show_all(self):
        """只展示左侧地图和右侧SLAM激光探索视图"""
        self.show_left()
        # 右侧SLAM激光探索视图
        self.ax_right.clear()
        base_img = np.full((self.map_size_pixels, self.map_size_pixels), 0.5)  # 全灰
        base_img[self._slam_explored_mask] = 1.0  # 白色
        base_img[self._slam_obstacle_mask] = 0.0  # 黑色
        self.ax_right.imshow(base_img, cmap='gray', origin='lower', vmin=0, vmax=1,
                             extent=[0, self.map_size_meters, 0, self.map_size_meters])
        if self._slam_simulator is not None:
            traj = np.array(self._slam_simulator.get_trajectory())
            if self._slam_pose is not None:
                self.ax_right.scatter([self._slam_pose[0]], [self._slam_pose[1]], 
                     s=250, c='magenta', edgecolors='black', linewidths=2, zorder=10)
                scan = self._slam_simulator.simulate_laser_scan(self._slam_pose)
                n_angles = self._slam_simulator.laser.scan_size
                angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
                for r, a in zip(scan, angles):
                    r_m = r / 1000.0
                    x0, y0, theta = self._slam_pose
                    x_end = x0 + r_m * np.cos(a + theta)
                    y_end = y0 + r_m * np.sin(a + theta)
                    self.ax_right.plot([x0, x_end], [y0, y_end], 'r-', alpha=0.2)
        self.ax_right.set_xlim(0, self.map_size_meters)
        self.ax_right.set_ylim(0, self.map_size_meters)
        self.ax_right.set_aspect('equal')
        self.ax_right.set_title('SLAM Laser Exploration (No Obstacles)')
        self.ax_right.grid(False)
        plt.draw()
        plt.pause(self.refresh_rate)

    def show(self):
        self.show_left()

    
    def show_lidar_scan(self, grid, x, y, n_beams=36, max_range=10):
        # 在左图上画出以(x, y)为中心的雷达扫描线
        h, w = grid.shape
        for i in range(n_beams):
            angle = 2 * np.pi * i / n_beams
            for r in np.linspace(0, max_range, int(max_range*5)):
                nx = int(round(x + r * np.cos(angle)))
                ny = int(round(y + r * np.sin(angle)))
                if 0<=nx<w and 0<=ny<h:
                    if grid[ny, nx] == 1:
                        self.ax_left.plot([x, nx], [y, ny], color='cyan', alpha=0.2, linewidth=1)
                        break
                else:
                    break
        plt.draw()
        plt.pause(0.05)  # 使用配置的刷新速率

    def save_current_fig_as_png(self, filename):
        plt.savefig(filename, dpi=150, bbox_inches='tight') 

    def set_slam_mapbytes(self, mapbytes):
        self._slam_mapbytes = mapbytes

    # 激光射线与线段相交
    def compute_ray_segment_intersection(self, origin, angle, seg_a, seg_b):
        x0, y0 = origin
        dx = math.cos(angle)
        dy = math.sin(angle)
        x1, y1 = seg_a
        x2, y2 = seg_b
        denom = (x2 - x1) * dy - (y2 - y1) * dx
        if abs(denom) < 1e-8:
            return None
        t = ((x1 - x0) * dy - (y1 - y0) * dx) / denom
        u = ((x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)) / denom
        if 0 <= t <= 1 and u >= 0:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return (ix, iy)
        return None

    # 找到最近障碍交点
    def find_laser_obstacle_intersection(self, x0, y0, angle, segments, max_range):
        min_dist = max_range
        hit_point = None
        hit_seg_idx = None
        for idx, seg in enumerate(segments):
            x1, y1 = seg['start']
            x2, y2 = seg['end']
            intersect = self.compute_ray_segment_intersection((x0, y0), angle, (x1, y1), (x2, y2))
            if intersect:
                dist = math.hypot(intersect[0] - x0, intersect[1] - y0)
                if dist < min_dist:
                    min_dist = dist
                    hit_point = intersect
                    hit_seg_idx = idx
        return hit_point, hit_seg_idx


# 辅助函数：点到线段距离
def point_to_segment_dist(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)

def refresh_left_maze_with_path_and_point(ax, segments, start_point, end_point, path, current_idx):
    
    ax.clear()
    # 1. 迷宫线段
    for seg in segments:
        x0, y0 = seg['start']
        x1, y1 = seg['end']
        ax.plot([x0, x1], [y0, y1], color='black', linewidth=4)
    # 2. 起点
    if start_point is not None:
        ax.plot(start_point[0], start_point[1], 'o', color='blue', markersize=12, label='Start')
    # 3. 终点
    if end_point is not None:
        ax.plot(end_point[0], end_point[1], 'o', color='red', markersize=12, label='End')
    # 4. A*最优路径
    if path and len(path) > 1:
        xs, ys = zip(*path)
        ax.plot(xs, ys, color='blue', linewidth=2, label='A* Path')
    # 5. 已走过的红色轨迹
    if path and current_idx > 0:
        xs, ys = zip(*path[:current_idx+1])
        ax.plot(xs, ys, color='red', linewidth=2, label='Return Trajectory')
    # 6. 当前点
    if path and 0 <= current_idx < len(path):
        x, y = path[current_idx]
        ax.plot(x, y, 'o', color='red', markersize=16, markeredgecolor='black', label='Current')
    ax.set_title("Maze & Path")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.draw()
    plt.pause(0.01)
