import heapq
import numpy as np

class AStarPlanner:
    def __init__(self, grid_size=(50, 50)):
        self.grid_size = grid_size

    def heuristic(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def plan_path(self, occupancy_grid, start, goal):
        rows, cols = self.grid_size
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = { (r, c): float('inf') for r in range(rows) for c in range(cols) }
        g_score[start] = 0
        f_score = { (r, c): float('inf') for r in range(rows) for c in range(cols) }
        f_score[start] = self.heuristic(start, goal)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in neighbors:
                neighbor = (current[0] + dr, current[1] + dc)
                r, c = neighbor
                if 0 <= r < rows and 0 <= c < cols:
                    if occupancy_grid[r, c] == 1:
                        continue
                    weight = 1.414 if (dr != 0 and dc != 0) else 1.0
                    tentative_g = g_score[current] + weight
                    if tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []
