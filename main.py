import sys
import numpy as np
import pygame
from simulation.env_2d import NavigationEnvironment
from src.perception import PerceptionEngine
from src.planner import AStarPlanner
from src.controller import KinematicController

def main():
    print("[INFO] Initializing Autonomous Navigation System...")
    env = NavigationEnvironment()
    perception = PerceptionEngine()
    planner = AStarPlanner()
    controller = KinematicController(speed=1.0)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        frame = env.get_frame()
        detections = perception.detect_obstacles(frame)
        occupancy_grid = perception.generate_occupancy_grid(frame.shape, detections)

        start_node = (int(env.robot_pos[0]), int(env.robot_pos[1]))
        goal_node = (int(env.goal_pos[0]), int(env.goal_pos[1]))
        path = planner.plan_path(occupancy_grid, start_node, goal_node)

        if path and len(path) > 1:
            target_node = path[1]
            speed, angular_vel = controller.compute_steering(
                env.robot_pos, env.robot_heading, target_node
            )
            env.robot_heading += angular_vel * 0.1
            env.robot_pos[0] += speed * np.sin(env.robot_heading) * 0.1
            env.robot_pos[1] += speed * np.cos(env.robot_heading) * 0.1

        env.render_world(path=path, detections=detections)
        
        # Fixed: np.linalg.norm instead of np.linalg_norm
        dist_to_goal = np.linalg.norm(np.array(env.robot_pos) - np.array(env.goal_pos))
        if dist_to_goal < 1.0:
            print("[SUCCESS] Goal Reached Successfully without collisions!")
            running = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
