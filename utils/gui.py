import matplotlib.pyplot as plt
import numpy as np
import json
import math
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from typing import Optional
matplotlib.use('TkAgg', force=True)  # 强制使用TkAgg后端
matplotlib.interactive(False)  # 禁用交互模式
plt.ioff()  # 关闭matplotlib的交互模式

# 蓝色主题
PRIMARY_COLOR = '#0D47A1'  # 深蓝色
HOVER_COLOR = '#1565C0'    # 悬停深蓝
BG_COLOR = '#eaf1fb'       # 主窗口淡蓝
STATUS_BG = '#1976D2'      # 状态栏深蓝
STATUS_FG = 'white'        # 状态栏白字
BTN_FONT = ('Bradley Hand', 16, 'bold')
LABEL_FONT = ('Bradley Hand', 12)
TITLE_FONT = {'fontsize': 20, 'fontweight': 'bold', 'fontname': 'Bradley Hand', 'color': PRIMARY_COLOR}
LEGEND_FONT = {'fontsize': 12, 'fontname': 'Bradley Hand'}

class MazeVisualizer:
    canvas: Optional[FigureCanvasTkAgg] = None  # 类型注解，兼容None和FigureCanvasTkAgg
    def __init__(self, map_size_pixels=21, map_size_meters=21.0, title="Maze Visualizer", refresh_rate=0.001):
        # 确保matplotlib不会弹出新窗口
        plt.ioff()
        matplotlib.interactive(False) 

        self.map_size_pixels = map_size_pixels
        self.map_size_meters = float(map_size_meters)
        self.refresh_rate = refresh_rate  # 新增：刷新速率配置
        self.fig, (self.ax_left, self.ax_right) = plt.subplots(1, 2, figsize=(14, 7))
        # 确保matplotlib不会创建独立窗口
        plt.ioff()
        matplotlib.interactive(False)
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.18)
        self.fig.set_facecolor(BG_COLOR)
        self.fig.suptitle(title, **TITLE_FONT)
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
        # 右上角蓝色横线装饰
        self.fig.subplots_adjust(top=0.88)
        self.fig.text(0.5, 0.96, '', ha='center', va='center', color=PRIMARY_COLOR, fontsize=1, bbox=dict(facecolor=PRIMARY_COLOR, edgecolor='none', boxstyle='square,pad=0.1'))
        self.canvas = None  # 用于TkAgg画布

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
        if self.canvas is not None:
            self.canvas.draw()
            try:
                self.canvas.get_tk_widget().update()
            except Exception:
                pass

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
        plt.pause(self.refresh_rate)  # 使用配置的刷新速率

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

    def mark_explored_by_laser(self, pose):
        # 累积激光束路径到_explored_mask
        if not hasattr(self, '_slam_explored_mask'):
            self._slam_explored_mask = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)
        if self._slam_simulator is None:
            return
        scan = self._slam_simulator.simulate_laser_scan(pose)
        n_angles = self._slam_simulator.laser.scan_size
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        x0, y0, theta = pose
        for r, a in zip(scan, angles):
            r_m = r / 1000.0
            x_end = x0 + r_m * np.cos(a + theta)
            y_end = y0 + r_m * np.sin(a + theta)
            num = int(max(abs(x_end-x0), abs(y_end-y0)) * self.map_size_pixels / self.map_size_meters * 2)
            if num < 2:
                num = 2
            xs = np.linspace(x0, x_end, num)
            ys = np.linspace(y0, y_end, num)
            for x, y in zip(xs, ys):
                ix = int(x / self.map_size_meters * self.map_size_pixels)
                iy = int(y / self.map_size_meters * self.map_size_pixels)
                if 0 <= ix < self.map_size_pixels and 0 <= iy < self.map_size_pixels:
                    self._slam_explored_mask[iy, ix] = True


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

class RoundedButton:
    def __init__(self, parent, text, command, bg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR, 
                 width=120, height=40, corner_radius=10):
        self.parent = parent
        self.text = text
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.width = width
        self.height = height
        self.corner_radius = corner_radius
        self.is_hovered = False
        
        # 创建Canvas
        self.canvas = tk.Canvas(parent, width=width, height=height, 
                               bg=BG_COLOR, highlightthickness=0, relief=tk.FLAT)
        
        # 绘制圆角矩形
        self.draw_button()
        
        # 绑定事件
        self.canvas.bind('<Enter>', self.on_enter)
        self.canvas.bind('<Leave>', self.on_leave)
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        
    def draw_button(self):
        self.canvas.delete("all")
        color = self.hover_color if self.is_hovered else self.bg_color
        
        # 绘制圆角矩形
        x1, y1 = self.corner_radius, self.corner_radius
        x2, y2 = self.width - self.corner_radius, self.height - self.corner_radius
        
        # 主体矩形
        self.canvas.create_rectangle(x1, 0, x2, self.height, fill=color, outline=color)
        self.canvas.create_rectangle(0, y1, self.width, y2, fill=color, outline=color)
        
        # 四个圆角
        self.canvas.create_arc(0, 0, 2*self.corner_radius, 2*self.corner_radius, 
                              start=90, extent=90, fill=color, outline=color)
        self.canvas.create_arc(self.width-2*self.corner_radius, 0, self.width, 2*self.corner_radius, 
                              start=0, extent=90, fill=color, outline=color)
        self.canvas.create_arc(0, self.height-2*self.corner_radius, 2*self.corner_radius, self.height, 
                              start=180, extent=90, fill=color, outline=color)
        self.canvas.create_arc(self.width-2*self.corner_radius, self.height-2*self.corner_radius, 
                              self.width, self.height, start=270, extent=90, fill=color, outline=color)
        
        # 添加文本
        self.canvas.create_text(self.width//2, self.height//2, text=self.text, 
                               fill='white', font=BTN_FONT)
    
    def on_enter(self, event):
        self.is_hovered = True
        self.draw_button()
        
    def on_leave(self, event):
        self.is_hovered = False
        self.draw_button()
        
    def on_click(self, event):
        self.command()
        
    def on_release(self, event):
        pass
        
    def pack(self, **kwargs):
        self.canvas.pack(**kwargs)
        
    def grid(self, **kwargs):
        self.canvas.grid(**kwargs)

class MazeApp(tk.Tk):
    def __init__(self, visualizer: MazeVisualizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("智能迷宫探索与SLAM建图系统")
        self.geometry("1200x800")
        self.configure(bg=BG_COLOR)
        self.visualizer = visualizer
        # 渐变背景Canvas
        self.bg_canvas = tk.Canvas(self, width=1200, height=800, highlightthickness=0, bd=0)
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self._draw_gradient(self.bg_canvas, 1200, 800, color1="#eaf1fb", color2="#1565C0")
        self.bind("<Configure>", self._on_resize_bg)
        # 顶部按钮区，使用Frame，背景色与渐变顶部色一致
        self.btn_frame = tk.Frame(self, highlightthickness=0, relief=tk.FLAT, bg='#eaf1fb')
        self.btn_frame.pack(side=tk.TOP, fill=tk.X, pady=24)
        self.btns = []
        self.btn_explore = self._create_rounded_button(self.btn_frame, "开始探索", self.on_explore)
        self.btn_return = self._create_rounded_button(self.btn_frame, "回到起点(A*)", self.on_return)
        self.btn_slam = self._create_rounded_button(self.btn_frame, "SLAM建图", self.on_slam)
        self.btn_save = self._create_rounded_button(self.btn_frame, "保存视图", self.on_save)
        self.btn_reset = self._create_rounded_button(self.btn_frame, "重置", self.on_reset)
        self.btn_exit = self._create_rounded_button(self.btn_frame, "退出", self.quit)
        btns = [self.btn_explore, self.btn_return, self.btn_slam, self.btn_save, self.btn_reset, self.btn_exit]
        for i, btn in enumerate(btns):
            btn.grid(row=0, column=i, padx=18, pady=6, sticky='nsew')
        self.btn_frame.grid_columnconfigure(tuple(range(len(btns))), weight=1)
        # matplotlib画布自适应Tk窗口
        plt.ioff()
        matplotlib.interactive(False)
        self.canvas = FigureCanvasTkAgg(self.visualizer.fig, master=self)
        self.visualizer.canvas = self.canvas  # 关键：让visualizer能访问canvas
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=16, pady=10)
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("欢迎使用智能迷宫探索系统！")
        self.status_bar = tk.Label(self, textvariable=self.status_var, bd=0, anchor=tk.W,
                                   bg=STATUS_BG, fg=STATUS_FG, font=LABEL_FONT, height=2)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 2))
        # 逻辑回调占位
        self.explore_callback = None
        self.return_callback = None
        self.slam_callback = None
        self.save_callback = None
        self.reset_callback = None
        plt.ioff()

    def _draw_gradient(self, canvas, width, height, color1, color2):
        # 竖直渐变色
        r1, g1, b1 = self.winfo_rgb(color1)
        r2, g2, b2 = self.winfo_rgb(color2)
        r_ratio = (r2 - r1) / height
        g_ratio = (g2 - g1) / height
        b_ratio = (b2 - b1) / height
        for i in range(height):
            nr = int(r1 + (r_ratio * i)) // 256
            ng = int(g1 + (g_ratio * i)) // 256
            nb = int(b1 + (b_ratio * i)) // 256
            color = f'#{nr:02x}{ng:02x}{nb:02x}'
            canvas.create_line(0, i, width, i, fill=color)

    def _on_resize_bg(self, event):
        # 窗口大小变化时重绘渐变背景
        self.bg_canvas.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        self.bg_canvas.config(width=w, height=h)
        self._draw_gradient(self.bg_canvas, w, h, color1="#eaf1fb", color2="#1565C0")

    def _create_rounded_button(self, parent, text, command):
        btn = RoundedButton(parent, text, command, 
                           bg_color=PRIMARY_COLOR, hover_color=HOVER_COLOR,
                           width=140, height=45, corner_radius=12)
        self.btns.append(btn)
        return btn

    def set_callbacks(self, explore_cb, return_cb, slam_cb, save_cb=None, reset_cb=None):
        self.explore_callback = explore_cb
        self.return_callback = return_cb
        self.slam_callback = slam_cb
        self.save_callback = save_cb
        self.reset_callback = reset_cb

    def on_explore(self):
        if self.explore_callback:
            self.status_var.set("正在探索迷宫...")
            self.explore_callback()
            self.status_var.set("探索完成！")

    def on_return(self):
        if self.return_callback:
            self.status_var.set("正在A*回到起点...")
            self.return_callback()
            self.status_var.set("回到起点完成！")

    def on_slam(self):
        if self.slam_callback:
            self.status_var.set("正在进行SLAM建图...")
            self.slam_callback()
            self.status_var.set("SLAM建图完成！")

    def on_save(self):
        if self.save_callback:
            self.save_callback()
            self.status_var.set("视图已保存！")
        else:
            self.visualizer.save_current_fig_as_png("maze_gui_snapshot.png")
            self.status_var.set("视图已保存为maze_gui_snapshot.png！")

    def on_reset(self):
        if self.reset_callback:
            self.reset_callback()
            self.status_var.set("已重置！")
        else:
            self.status_var.set("重置功能未实现！") 

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
