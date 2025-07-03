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

    def load_line_segments(self, segments, start_point=None, end_point=None):
        self._maze_segments = segments
        self._start_point = start_point
        self._end_point = end_point

    def set_trajectory(self, trajectory):
        self._trajectory = trajectory

    def set_scan_points(self, scan_points):
        self._scan_points = scan_points

    def set_path(self, path):
        self._path = path

    def set_pgm_img(self, pgm_img):
        self._pgm_img = pgm_img

    def set_return_trajectory(self, return_traj):
        self._return_traj = return_traj

    def set_slam_simulator(self, slam_simulator):
        self._slam_simulator = slam_simulator
        # 重置掩码
        self._slam_scanned_mask = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)

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

    def _mark_laser_path(self, x0, y0, x1, y1):
        # 将激光束路径上的像素点在scanned_mask中置True
        num = int(max(abs(x1-x0), abs(y1-y0)) * self.map_size_pixels / self.map_size_meters * 2)
        if num < 2:
            num = 2
        xs = np.linspace(x0, x1, num)
        ys = np.linspace(y0, y1, num)
        for x, y in zip(xs, ys):
            ix = int(x / self.map_size_meters * self.map_size_pixels)
            iy = int(y / self.map_size_meters * self.map_size_pixels)
            if 0 <= ix < self.map_size_pixels and 0 <= iy < self.map_size_pixels:
                self._slam_scanned_mask[iy, ix] = True

    def show(self):
        # 左侧：迷宫线段、起点、终点、轨迹、扫描点、路径
        self.ax_left.clear()
        for seg in self._maze_segments:
            x0, y0 = seg['start']
            x1, y1 = seg['end']
            self.ax_left.plot([x0, x1], [y0, y1], color='black', linewidth=4)
        if self._start_point is not None:
            self.ax_left.plot(self._start_point[0], self._start_point[1], 'o', color='green', markersize=12, label='Start')
        if self._end_point is not None:
            self.ax_left.plot(self._end_point[0], self._end_point[1], 'o', color='red', markersize=12, label='End')
        if self._trajectory is not None and len(self._trajectory) > 0:
            xs, ys = zip(*self._trajectory)
            self.ax_left.plot(xs, ys, color='orange', linewidth=2, alpha=0.7, label='Trajectory')
            self.ax_left.scatter([xs[-1]], [ys[-1]], 
                        s=300, c='magenta', edgecolors='black', linewidths=2, zorder=10, label='Flood Cursor')
        if self._scan_points is not None and len(self._scan_points) > 0:
            xs, ys = zip(*self._scan_points)
            self.ax_left.scatter(xs, ys, color='cyan', s=30, label='Scan Points', alpha=0.7)
        if self._path is not None and len(self._path) > 0:
            xs, ys = zip(*[(x, y) for y, x in self._path])
            self.ax_left.plot(xs, ys, color='blue', linewidth=2, label='A* Path')
        if hasattr(self, '_return_traj') and self._return_traj and len(self._return_traj) > 1:
            xs, ys = zip(*self._return_traj)
            self.ax_left.plot(xs, ys, color='red', linewidth=2, label='Return Trajectory')
            # 左侧光标：填充色黑色，边缘黑色，更大尺寸更醒目
            self.ax_left.scatter([xs[-1]], [ys[-1]], 
                    s=300, c='magenta', edgecolors='black', linewidths=2, zorder=10, label='Left Cursor')
        self.ax_left.set_title("Maze & Path")
        self.ax_left.set_xlabel('X')
        self.ax_left.set_ylabel('Y')
        self.ax_left.set_aspect('equal')
        self.ax_left.legend(loc='upper right')
        self.ax_left.grid(True)
        # 右侧：SLAM激光探索视图
        self.ax_right.clear()
        self.ax_right.set_facecolor('0.5')
        # 右侧底图：未探索（黑），障碍（灰，永久），已探索（白）
        base_img = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=float)  # 黑色
        if self._slam_simulator is not None and hasattr(self._slam_simulator, 'occupancy_grid') and self._slam_simulator.occupancy_grid is not None:
            occ = self._slam_simulator.occupancy_grid
            if occ.shape != base_img.shape:
                from scipy.ndimage import zoom
                occ = zoom(occ, (self.map_size_pixels / occ.shape[0], self.map_size_pixels / occ.shape[1]), order=0)
            base_img[occ == 1] = 0.5  # 障碍灰色，永久
        # 激光照射到的非障碍区域置为白色
        mask_explored = self._slam_scanned_mask & (base_img != 0.5)
        base_img[mask_explored] = 1.0
        self.ax_right.imshow(base_img, cmap='gray', origin='lower', vmin=0, vmax=1,
                             extent=[0, self.map_size_meters, 0, self.map_size_meters])
        # 轨迹
        if self._slam_simulator is not None:
            traj = np.array(self._slam_simulator.get_trajectory())
            if len(traj) > 1:
                self.ax_right.plot(traj[:,0], traj[:,1], 'b-', linewidth=2)
            if self._slam_pose is not None:
                # 右侧光标：填充色黑色，边缘黑色，调整尺寸和样式更醒目
                self.ax_right.scatter([self._slam_pose[0]], [self._slam_pose[1]], 
                     s=250, c='magenta', edgecolors='black', linewidths=2, zorder=10)
                # 激光束
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
        plt.pause(0.05)  # 使用配置的刷新速率

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
