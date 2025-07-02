import numpy as np
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser

class SLAMSimulator:
    def __init__(self, map_size_pixels=800, map_size_meters=15.0):
        # 自动修正像素分辨率，保证每米不少于20像素
        min_pixels_per_meter = 20
        if map_size_pixels < map_size_meters * min_pixels_per_meter:
            map_size_pixels = int(map_size_meters * min_pixels_per_meter)
        self.MAP_SIZE_PIXELS = map_size_pixels
        self.MAP_SIZE_METERS = map_size_meters
        self.laser = Laser(360, 10, 360, 10000, 0, 0)
        self.slam = RMHC_SLAM(self.laser, self.MAP_SIZE_PIXELS, self.MAP_SIZE_METERS)
        self.occupancy_grid = None  # Binary maze grid (1=obstacle, 0=free)
        self.mapbytes = bytearray(self.MAP_SIZE_PIXELS * self.MAP_SIZE_PIXELS)
        self.visited_positions = []  # 新增：记录走过的点 (x, y, theta)
        self.loop_closure_threshold = 0.5  # 单位：米
        self.last_loop_detected = False

    def set_occupancy_grid(self, grid):
        # grid应为未膨胀的maze_grid
        self.occupancy_grid = grid

    def simulate_laser_scan(self, pose=None):
        # Simulate a laser scan based on the occupancy grid and robot pose
        if self.occupancy_grid is None or pose is None:
            return list(np.random.randint(0, self.laser.distance_no_detection_mm, self.laser.scan_size))
        scan = []
        n_angles = self.laser.scan_size
        max_range = self.laser.distance_no_detection_mm / 1000.0  # meters
        angle_min = 0
        angle_max = 2 * np.pi
        angles = np.linspace(angle_min, angle_max, n_angles, endpoint=False)
        # Robot pose: [x, y, theta] in meters/radians
        x0 = float(np.clip(pose[0], 0, self.MAP_SIZE_METERS-1e-3))
        y0 = float(np.clip(pose[1], 0, self.MAP_SIZE_METERS-1e-3))
        theta0 = pose[2] if len(pose) > 2 else 0
        grid_h, grid_w = self.occupancy_grid.shape
        scale_x = self.MAP_SIZE_METERS / grid_w
        scale_y = self.MAP_SIZE_METERS / grid_h
        for a in angles:
            found = False
            for r in np.linspace(0, max_range, int(max_range/0.02)):
                x = x0 + r * np.cos(a + theta0)
                y = y0 + r * np.sin(a + theta0)
                j = int(round(x / scale_x))
                i = int(round(y / scale_y))
                if 0 <= i < grid_h and 0 <= j < grid_w:
                    if self.occupancy_grid[i, j] == 1:
                        scan.append(int(r * 1000))
                        found = True
                        break
                else:
                    scan.append(int(r * 1000))
                    found = True
                    break
            if not found:
                scan.append(int(max_range * 1000))
        # scan合法性检查
        scan = np.clip(scan, 0, self.laser.distance_no_detection_mm)
        if len(scan) != self.laser.scan_size:
            scan = list(scan)[:self.laser.scan_size] + [self.laser.distance_no_detection_mm]*(self.laser.scan_size-len(scan))
        scan = list(scan)  # 强制转为list
        return scan

    def update(self, sensor_data=None, pose_change=None):
        if sensor_data is None:
            sensor_data = self.simulate_laser_scan()
        if pose_change is None:
            pose_change = (0, 0, 0.1)
        sensor_data = np.clip(sensor_data, 0, self.laser.distance_no_detection_mm)
        if len(sensor_data) != self.laser.scan_size:
            sensor_data = list(sensor_data)[:self.laser.scan_size] + [self.laser.distance_no_detection_mm]*(self.laser.scan_size-len(sensor_data))
        sensor_data = list(sensor_data)  # 强制转为list
        self.slam.update(sensor_data, pose_change)
        # 新增：记录当前位置
        x_mm, y_mm, theta_degrees = self.slam.getpos()
        x_m = x_mm / 1000.0
        y_m = y_mm / 1000.0
        self.visited_positions.append(( x_m,y_m, theta_degrees))  # [x, y, theta]
        # 新增：回环检测
        self.last_loop_detected = False
        for prev_x, prev_y, _ in self.visited_positions[:-1]:
            distance = np.hypot(x_m - prev_x, y_m - prev_y)
            if distance < self.loop_closure_threshold:
                self.last_loop_detected = True
                break
    

    def get_map(self):
        self.slam.getmap(self.mapbytes)
        return self.mapbytes
    def get_position(self):
        x_mm, y_mm, theta_degrees = self.slam.getpos()
        x_m = x_mm / 1000.0
        y_m = y_mm / 1000.0
        return np.array([x_m, y_m, theta_degrees])
    def get_trajectory(self):
        return self.visited_positions
    def get_last_loop_detected(self):
        return self.last_loop_detected
    
    