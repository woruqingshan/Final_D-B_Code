# 洪水填充（Flood Fill）+ A* 迷宫探索算法

## 算法原理

本算法结合了洪水填充（Flood Fill）和A*路径规划，实现了对未知迷宫的全局遍历与可视化。其核心思想是：
- 以起点为种子点，逐步扩展所有可达区域。
- 每次从当前机器人位置出发，A*到最近的未探索边界点，沿路径逐步移动并标记为已探索。
- 支持遇到死路后自动回溯，最终遍历所有可达区域。
- 边界上的空地（非起点）被视为出口。

## 算法流程
1. 初始化迷宫栅格，标记障碍、起点。
2. 标记起点为已探索。
3. 重复以下步骤直到探索率达到阈值：
   - 生成所有已探索点的邻居中未探索的点，作为边界目标。
   - 选择距离当前位置最近的目标，A*规划路径。
   - 沿A*路径逐步移动，每步都标记为已探索、可视化。
   - 若到达边界空地且非起点，标记为出口。
   - 若所有目标都不可达，则终止。
4. 探索结束后，若当前位置不是最近的出口，则A*移动到最近出口。

## 伪代码
```
current_pos = start
mark_explored(current_pos)
while exploration_rate < threshold:
    goals = generate_exploration_goals()
    for goal in sorted(goals, by distance):
        path = astar(current_pos, goal)
        if path:
            for pos in path[1:]:
                mark_explored(pos)
                current_pos = pos
                if is_exit(pos):
                    record_exit(pos)
            break
    else:
        break
# 结束后移动到最近出口
if current_pos != nearest_exit:
    path = astar(current_pos, nearest_exit)
    for pos in path[1:]:
        mark_explored(pos)
        current_pos = pos
```

## 主要接口说明
- `FloodFillExplorer(maze, config)`：探索器主类。
- `explore(start_pos)`：主循环，返回完整轨迹和出口集合。
- `generate_exploration_goals()`：生成所有边界目标点。
- `AStarPlanner.plan(start, goal)`：A*路径规划。
- `Maze.mark_explored(x, y)`：标记已探索。
- `Maze.get_exploration_rate()`：返回探索率。

## 适用场景
- 需要全局遍历未知迷宫、采集数据、同步可视化、自动检测出口等任务。

## 参考实现
详见本目录下`floodfill_explorer.py`与`test_flood.py`。 