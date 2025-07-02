import matplotlib.pyplot as plt
import numpy as np
import json

class MazeVisualizer:
    def __init__(self, map_size_pixels=21, map_size_meters=21, title="Maze Visualizer"):
        self.map_size_pixels = map_size_pixels
        self.map_size_meters = map_size_meters
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
        if self._scan_points is not None and len(self._scan_points) > 0:
            xs, ys = zip(*self._scan_points)
            self.ax_left.scatter(xs, ys, color='cyan', s=30, label='Scan Points', alpha=0.7)
        if self._path is not None and len(self._path) > 0:
            xs, ys = zip(*[(x, y) for y, x in self._path])
            self.ax_left.plot(xs, ys, color='blue', linewidth=2, label='A* Path')
        if hasattr(self, '_return_traj') and self._return_traj and len(self._return_traj) > 1:
            xs, ys = zip(*self._return_traj)
            self.ax_left.plot(xs, ys, color='red', linewidth=2, label='Return Trajectory')
            self.ax_left.plot(xs[-1], ys[-1], 'o', color='red', markersize=10, label='Cursor')
        self.ax_left.set_title("Maze & Path")
        self.ax_left.set_xlabel('X')
        self.ax_left.set_ylabel('Y')
        self.ax_left.set_aspect('equal')
        self.ax_left.legend(loc='upper right')
        self.ax_left.grid(True)
        # 右侧：PGM地图
        self.ax_right.clear()
        if self._pgm_img is not None:
            if self._pgm_img.dtype != np.uint8:
                img = self._pgm_img.astype(np.uint8)
            else:
                img = self._pgm_img
            self.ax_right.imshow(img, cmap='gray', origin='upper', vmin=0, vmax=255)
            self.ax_right.set_title("PGM Map")
            self.ax_right.set_xlabel('X')
            self.ax_right.set_ylabel('Y')
            self.ax_right.set_aspect('equal')
        self.ax_right.grid(False)
        plt.draw()
        plt.pause(0.01)

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
        plt.pause(0.01)

    def save_current_fig_as_png(self, filename):
        plt.savefig(filename, dpi=150, bbox_inches='tight') 