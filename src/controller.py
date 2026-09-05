import math

class KinematicController:
    def __init__(self, speed=3.0):
        self.speed = speed

    def compute_steering(self, current_pos, current_heading, target_pos):
        dx = target_pos[1] - current_pos[1]
        dy = target_pos[0] - current_pos[0]
        target_angle = math.atan2(dy, dx)
        angle_error = target_angle - current_heading
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        kp = 1.5
        angular_velocity = kp * angle_error
        return self.speed, angular_velocity
