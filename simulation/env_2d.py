import pygame
import numpy as np
import cv2

class NavigationEnvironment:
    def __init__(self, width=800, height=800, grid_size=(50, 50)):
        pygame.init()
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AI-Based Autonomous Navigation System")
        self.clock = pygame.time.Clock()
        self.robot_pos = [5, 5]
        self.goal_pos = [42, 42]
        self.robot_heading = 0.0
        self.obstacles = [
            (15, 15, 10, 10),
            (25, 5, 8, 20),
            (10, 30, 20, 8),
            (35, 25, 10, 10)
        ]

    def render_world(self, path=None, detections=None):
        self.screen.fill((20, 20, 25))
        cell_h = self.height / self.grid_size[0]
        cell_w = self.width / self.grid_size[1]

        for r in range(self.grid_size[0]):
            pygame.draw.line(self.screen, (35, 35, 45), (0, r * cell_h), (self.width, r * cell_h))
        for c in range(self.grid_size[1]):
            pygame.draw.line(self.screen, (35, 35, 45), (c * cell_w, 0), (c * cell_w, self.height))

        for (r, c, h, w) in self.obstacles:
            rect = pygame.Rect(c * cell_w, r * cell_h, w * cell_w, h * cell_h)
            pygame.draw.rect(self.screen, (220, 50, 50), rect)

        if path and len(path) > 1:
            points = [(p[1] * cell_w + cell_w / 2, p[0] * cell_h + cell_h / 2) for p in path]
            pygame.draw.lines(self.screen, (0, 255, 150), False, points, 3)

        goal_px = (self.goal_pos[1] * cell_w + cell_w / 2, self.goal_pos[0] * cell_h + cell_h / 2)
        pygame.draw.circle(self.screen, (50, 255, 50), goal_px, 10)

        rx = self.robot_pos[1] * cell_w + cell_w / 2
        ry = self.robot_pos[0] * cell_h + cell_h / 2
        pygame.draw.circle(self.screen, (0, 180, 255), (rx, ry), 8)
        
        hx = rx + 15 * np.cos(self.robot_heading)
        hy = ry + 15 * np.sin(self.robot_heading)
        pygame.draw.line(self.screen, (255, 255, 255), (rx, ry), (hx, hy), 2)

        pygame.display.flip()
        self.clock.tick(30)

    def get_frame(self):
        data = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(data, (1, 0, 2))
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
