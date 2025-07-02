# -*- coding: utf-8 -*-

import math
from heapq import heappush, heappop

class AStarPlanner:
    def __init__(self, grid, step=1.0):
        self.grid = grid
        self.step = step
        self.motion = self.get_motion_model(step)

    def planning(self, start, goal):
        open_list, closed_list = [], {}
        heappush(open_list, (0, start))
        cost = {start: 0}
        parent = {start: None}
        while open_list:
            _, current = heappop(open_list)
            if self.is_goal(current, goal):
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                return path
            closed_list[current] = True
            for move_x, move_y, move_cost in self.motion:
                next_node = (round(current[0] + move_x, 3), round(current[1] + move_y, 3))
                if next_node in closed_list:
                    continue
                if not self.verify_node(next_node):
                    continue
                new_cost = cost[current] + move_cost
                if next_node not in cost or new_cost < cost[next_node]:
                    cost[next_node] = new_cost
                    priority = new_cost + self.heuristic(next_node, goal)
                    heappush(open_list, (priority, next_node))
                    parent[next_node] = current
        return None

    def verify_node(self, node):
        y, x = int(round(node[0])), int(round(node[1]))
        if y < 0 or y >= self.grid.shape[0] or x < 0 or x >= self.grid.shape[1]:
            return False
        if self.grid[y, x] == 1:
            return False
        return True

    def heuristic(self, node, goal):
        return math.hypot(node[0] - goal[0], node[1] - goal[1])

    def get_motion_model(self, step=1.0):
        s = step
        return [[s, 0, s], [0, s, s], [-s, 0, s], [0, -s, s],
                [s, s, math.sqrt(2)*s], [-s, s, math.sqrt(2)*s],
                [s, -s, math.sqrt(2)*s], [-s, -s, math.sqrt(2)*s]]

    def is_goal(self, node, goal):
        return math.hypot(node[0] - goal[0], node[1] - goal[1]) < self.step/2