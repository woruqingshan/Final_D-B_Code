def slam_mapping(self):
        if hasattr(self, '_slam_running') and self._slam_running:
            return  # 防止重复点击
        if not self.scan_imgs:
            print('请先进行探索')
            return
        self._slam_running = True
        fused_img = fuse_scans(self.scan_imgs)
        fused_path = os.path.join(self.PGM_SAVE_PATH, 'fused_maze.pgm')
        save_pgm(fused_path, fused_img)
        fused_png_path = os.path.join(self.PGM_SAVE_PATH, 'fused_maze.png')
        plt.imsave(fused_png_path, fused_img, cmap='gray', vmin=0, vmax=255)
        self.fused_img = fused_img
        # 初始化SLAM
        map_size_pixels = int(self.map_size_meters * self.upscale)
        self._slam_map_size_pixels = map_size_pixels
        self._slam_mapbytes = bytearray(map_size_pixels * map_size_pixels)
        self._slam_pose_idx = 0
        self._slam_traj = list(self.traj)
        self._slam_slam = SLAMSimulator(map_size_pixels=map_size_pixels, map_size_meters=self.map_size_meters)
        self._slam_slam.set_occupancy_grid((fused_img<128).astype(np.uint8))
        self.vis.set_slam_simulator(self._slam_slam)
        self._slam_pose = [self.start_point[0] / (self.maze.grid.shape[1] - 1) * self.map_size_meters, self.start_point[1] / (self.maze.grid.shape[0] - 1) * self.map_size_meters, 0]
        self._slam_mapping_step()

    def _slam_mapping_step(self):
        if self._slam_pose_idx >= len(self._slam_traj):
            # 保存最终SLAM地图
            slam_map = np.array(self._slam_mapbytes, dtype=np.uint8).reshape(self._slam_map_size_pixels, self._slam_map_size_pixels)
            slam_map_path = os.path.join(self.PGM_SAVE_PATH, 'slam_result.png')
            plt.imsave(slam_map_path, slam_map, cmap='gray', vmin=0, vmax=255)
            print(f'SLAM结果图已保存为{slam_map_path}')
            self._slam_running = False
            return
        x, y = self._slam_traj[self._slam_pose_idx]
        new_pose = [x / (self.maze.grid.shape[1] - 1) * self.map_size_meters, y / (self.maze.grid.shape[0] - 1) * self.map_size_meters, 0]
        dx = new_pose[0] - self._slam_pose[0]
        dy = new_pose[1] - self._slam_pose[1]
        dtheta = new_pose[2] - self._slam_pose[2]
        pose_change = (dx, dy, dtheta)
        self._slam_pose = new_pose
        scan = self._slam_slam.simulate_laser_scan(self._slam_pose)
        self._slam_slam.update(scan, pose_change)
        self._slam_slam.slam.getmap(self._slam_mapbytes)
        self.vis.set_slam_mapbytes(self._slam_mapbytes)
        self.vis.trajectory = self._slam_traj[:self._slam_pose_idx+1]
        self.vis.update_slam_pose(self._slam_pose)
        self.vis.show_all()
        self._slam_pose_idx += 1
        canvas = getattr(self.vis, 'canvas', None)
        if canvas is not None and hasattr(canvas, 'get_tk_widget'):
            canvas.get_tk_widget().after(10, self._slam_mapping_step)
        else:
            import time; time.sleep(self.config.animation_speed)
            self._slam_mapping_step()
