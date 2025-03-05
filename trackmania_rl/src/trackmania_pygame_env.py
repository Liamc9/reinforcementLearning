import pygame
import math
import numpy as np

class TrackmaniaPygameEnv:
    """
    A simplified Trackmania environment for initial training.
    The car moves at a constant speed (so it only needs to learn to steer) and
    checkpoints are drawn as perpendicular lines across the full track width.
    The reward only considers progress toward the next checkpoint and a bonus for hitting it.
    """
    def __init__(self, config):
        pygame.init()
        self.screen_width = config.get("screen_width", 800)
        self.screen_height = config.get("screen_height", 600)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Trackmania RL - Simplified Training")
        self.clock = pygame.time.Clock()

        # Define a basic track as a list of waypoints (centerline)
        self.track_points = [
            (100, 400),
            (200, 250),
            (300, 300),
            (400, 350),
            (500, 300),
            (600, 250),
            (700, 300)
        ]
        # off_track_threshold defines half the track width (and is used for checkpoint detection)
        self.off_track_threshold = config.get("off_track_threshold", 50)

        # Reward parameters for the simplified training phase.
        self.step_penalty = config.get("step_penalty", -0.1)
        self.progress_factor = config.get("progress_factor", 0.05)
        self.waypoint_bonus = config.get("waypoint_bonus", 10)
        self.finish_bonus = config.get("finish_bonus", 50)

        # Constant speed mode: the car always moves at a fixed speed.
        self.constant_speed = config.get("constant_speed", True)
        self.constant_speed_value = config.get("constant_speed_value", 5)

        self.reset()

    def reset(self):
        # Start at the first waypoint with fixed speed if in constant speed mode.
        speed = self.constant_speed_value if self.constant_speed else 0
        self.state = np.array([self.track_points[0][0],
                               self.track_points[0][1],
                               speed,
                               0], dtype=np.float32)
        self.current_waypoint_index = 1
        self.done = False
        # Record initial distance to the first checkpoint.
        if self.current_waypoint_index < len(self.track_points):
            self.prev_distance = math.hypot(
                self.state[0] - self.track_points[self.current_waypoint_index][0],
                self.state[1] - self.track_points[self.current_waypoint_index][1]
            )
        else:
            self.prev_distance = 0
        return self.state

    def _segments_intersect(self, p1, p2, q1, q2):
        # Returns True if line segments p1-p2 and q1-q2 intersect.
        def orientation(a, b, c):
            val = (b[1]-a[1])*(c[0]-b[0]) - (b[0]-a[0])*(c[1]-b[1])
            if abs(val) < 1e-6:
                return 0
            return 1 if val > 0 else 2

        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)

        if o1 != o2 and o3 != o4:
            return True
        return False

    def _get_checkpoint_line(self, waypoint, idx):
        # Compute the local tangent direction using neighboring waypoints.
        if idx == 0:
            dx = self.track_points[1][0] - waypoint[0]
            dy = self.track_points[1][1] - waypoint[1]
        elif idx == len(self.track_points) - 1:
            dx = waypoint[0] - self.track_points[idx - 1][0]
            dy = waypoint[1] - self.track_points[idx - 1][1]
        else:
            dx1 = waypoint[0] - self.track_points[idx - 1][0]
            dy1 = waypoint[1] - self.track_points[idx - 1][1]
            dx2 = self.track_points[idx + 1][0] - waypoint[0]
            dy2 = self.track_points[idx + 1][1] - waypoint[1]
            dx = (dx1 + dx2) / 2
            dy = (dy1 + dy2) / 2

        mag = math.hypot(dx, dy)
        if mag == 0:
            norm_dx, norm_dy = 0, 0
        else:
            norm_dx, norm_dy = dx / mag, dy / mag

        # The checkpoint line is perpendicular to the tangent.
        perp_x = -norm_dy
        perp_y = norm_dx
        half_width = self.off_track_threshold
        start_point = (waypoint[0] + perp_x * half_width, waypoint[1] + perp_y * half_width)
        end_point = (waypoint[0] - perp_x * half_width, waypoint[1] - perp_y * half_width)
        return start_point, end_point

    def _min_distance_to_track(self, x, y):
        # Compute the minimum distance from (x, y) to the track centerline.
        min_dist = float('inf')
        for i in range(len(self.track_points) - 1):
            x1, y1 = self.track_points[i]
            x2, y2 = self.track_points[i+1]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                dist = math.hypot(x - x1, y - y1)
            else:
                t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx*dx + dy*dy)))
                nearest_x = x1 + t * dx
                nearest_y = y1 + t * dy
                dist = math.hypot(x - nearest_x, y - nearest_y)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def step(self, action):
        old_state = self.state.copy()
        old_position = (old_state[0], old_state[1])
        x, y, speed, angle = self.state

        # In constant speed mode, only allow steering.
        if self.constant_speed:
            if action == 2:      # Steer left
                angle -= 5
            elif action == 3:    # Steer right
                angle += 5
            speed = self.constant_speed_value
        else:
            if action == 0:      # Accelerate
                speed += 1
            elif action == 1:    # Brake
                speed = max(0, speed - 1)
            elif action == 2:
                angle -= 5
            elif action == 3:
                angle += 5

        # Update position.
        x += speed * math.cos(math.radians(angle))
        y += speed * math.sin(math.radians(angle))
        self.state = np.array([x, y, speed, angle], dtype=np.float32)
        new_position = (x, y)

        # Compute reward: step penalty + progress toward next checkpoint.
        reward = self.step_penalty
        if self.current_waypoint_index < len(self.track_points):
            waypoint = self.track_points[self.current_waypoint_index]
            current_distance = math.hypot(x - waypoint[0], y - waypoint[1])
            reward += self.progress_factor * (self.prev_distance - current_distance)
            self.prev_distance = current_distance

            # Check for checkpoint crossing via segment intersection.
            cp_start, cp_end = self._get_checkpoint_line(waypoint, self.current_waypoint_index)
            if self._segments_intersect(old_position, new_position, cp_start, cp_end):
                reward += self.waypoint_bonus
                self.current_waypoint_index += 1
                if self.current_waypoint_index < len(self.track_points):
                    next_wp = self.track_points[self.current_waypoint_index]
                    self.prev_distance = math.hypot(x - next_wp[0], y - next_wp[1])
        else:
            reward += self.finish_bonus
            self.done = True

        # End episode if off track or off-screen.
        if self._min_distance_to_track(x, y) > self.off_track_threshold:
            self.done = True
            reward = -10
        if x < 0 or x > self.screen_width or y < 0 or y > self.screen_height:
            self.done = True
            reward = -10

        return self.state, reward, self.done, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((144, 238, 144))
        # Draw the track as a thick line.
        track_width = int(self.off_track_threshold * 2)
        if len(self.track_points) >= 2:
            pygame.draw.lines(self.screen, (50, 50, 50), False, self.track_points, track_width)

        # Draw perpendicular checkpoint lines.
        for i, point in enumerate(self.track_points):
            cp_start, cp_end = self._get_checkpoint_line(point, i)
            pygame.draw.line(self.screen, (255, 165, 0),
                             (int(cp_start[0]), int(cp_start[1])),
                             (int(cp_end[0]), int(cp_end[1])), 3)
            pygame.draw.circle(self.screen, (255, 0, 0), (int(point[0]), int(point[1])), 5)

        # Highlight the current checkpoint.
        if self.current_waypoint_index < len(self.track_points):
            target = self.track_points[self.current_waypoint_index]
            pygame.draw.circle(self.screen, (0, 255, 0), (int(target[0]), int(target[1])), 8, 2)

        # Draw the car.
        x, y, speed, angle = self.state
        car_width, car_height = 40, 20
        car_surface = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
        car_surface.fill((0, 0, 255))
        rotated_car = pygame.transform.rotate(car_surface, -angle)
        car_rect = rotated_car.get_rect(center=(int(x), int(y)))
        self.screen.blit(rotated_car, car_rect.topleft)

        # Overlay status.
        font = pygame.font.SysFont("Arial", 18)
        stats_text = f"Speed: {speed:.1f}  Angle: {angle:.1f}"
        text_surface = font.render(stats_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
