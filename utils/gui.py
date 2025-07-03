import matplotlib.pyplot as plt
import numpy as np
import json

class MazeVisualizer:
    def __init__(self, map_size_pixels=21, map_size_meters=21, title="Maze Visualizer", refresh_rate=0.001):
        self.map_size_pixels = map_size_pixels
        self.map_size_meters = map_size_meters
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

    def load_line_segments(self, segments, start_point=None, end_point=None):
        self._maze_segments = segments
        self._start_point = start_point
        self._end_point = end_point

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
        # 重置掩码
        self._slam_scanned_mask = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)
        self._slam_explored_mask = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)
        self._slam_obstacle_mask = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)

    def update_slam_pose(self, pose):
        self._slam_pose = pose
        if self._slam_simulator is not None and pose is not None:
            scan = self._slam_simulator.simulate_laser_scan(pose)
            n_angles = self._slam_simulator.laser.scan_size
            angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
            for r, a in zip(scan, angles):
                r_m = r / 1000.0
                x0, y0, theta = pose
                x_end = x0 + r_m * np.cos(a + theta)
                y_end = y0 + r_m * np.sin(a + theta)
                self._mark_laser_path(x0, y0, x_end, y_end)

    # 新增：设置检测到的终点
    def set_detected_ends(self, detected_ends):
        """设置检测到的终点集合，用于可视化"""
        self._detected_ends = detected_ends

    def _mark_laser_path(self, x0, y0, x1, y1):
        # 将激光束路径上的像素点在scanned_mask中置True
        num = int(max(abs(x1-x0), abs(y1-y0)) * (self.map_size_pixels-1) / (self.map_size_meters-1) * 2)
        if num < 2:
            num = 2
        xs = np.linspace(x0, x1, num)
        ys = np.linspace(y0, y1, num)
        for idx, (x, y) in enumerate(zip(xs, ys)):
            # 修正：最大坐标映射到最后一格
            ix = int(round(x / (self.map_size_meters-1) * (self.map_size_pixels-1)))
            iy = int(round(y / (self.map_size_meters-1) * (self.map_size_pixels-1)))
            ix = min(max(ix, 0), self.map_size_pixels - 1)
            iy = min(max(iy, 0), self.map_size_pixels - 1)
            if 0 <= ix < self.map_size_pixels and 0 <= iy < self.map_size_pixels:
                self._slam_scanned_mask[iy, ix] = True
                # 新增：终点判断障碍/可行区域
                if idx == len(xs) - 1:
                    if self._slam_simulator and self._slam_simulator.occupancy_grid is not None:
                        grid = self._slam_simulator.occupancy_grid
                        grid_h, grid_w = grid.shape
                        gx = min(max(int(round(ix / (self.map_size_pixels-1) * (grid_w-1))), 0), grid_w - 1)
                        gy = min(max(int(round(iy / (self.map_size_pixels-1) * (grid_h-1))), 0), grid_h - 1)
                        if 0 <= gx < grid_w and 0 <= gy < grid_h:
                            if grid[gy, gx] == 1:
                                self._slam_obstacle_mask[iy, ix] = True
                            else:
                                self._slam_explored_mask[iy, ix] = True
                else:
                    self._slam_explored_mask[iy, ix] = True

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

    def show_both(self):
        """刷新左侧地图和右侧SLAM画面"""
        self.show_left()
        # 右侧SLAM激光探索视图
        self.ax_right.clear()
        # 新逻辑：全灰，障碍黑色，可行区域白色
        base_img = np.full((self.map_size_pixels, self.map_size_pixels), 0.5)  # 全灰
        base_img[self._slam_explored_mask] = 1.0  # 白色
        base_img[self._slam_obstacle_mask] = 0.0  # 黑色
        self.ax_right.imshow(base_img, cmap='gray', origin='lower', vmin=0, vmax=1,
                             extent=[0, self.map_size_meters, 0, self.map_size_meters])
        if self._slam_simulator is not None:
            traj = np.array(self._slam_simulator.get_trajectory())
            if len(traj) > 1:
                self.ax_right.plot(traj[:,0], traj[:,1], 'b-', linewidth=2)
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

    # 兼容主探索阶段调用
    def show_full_slam(self):
        self.show_both()

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
