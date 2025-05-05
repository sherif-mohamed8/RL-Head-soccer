import pygame
import numpy as np
from settings import *

class Player:
    def __init__(self, x, y, color, ai=False):
        self.pos = np.array([x, y], dtype=np.float32)
        self.color = color
        self.ai = ai
        self.prev_pos = self.pos.copy()

    def move(self, keys, bounds, up=None, down=None, left=None, right=None):
        """
        Move the player according to pressed keys.
        If ai=True, do nothing. Otherwise, use the provided key mappings
        or default to WASD.
        """
        self.prev_pos = self.pos.copy()
        if self.ai:
            return

        # Determine which keys to use (dynamic or default WASD)
        up_key    = up    or pygame.K_w
        down_key  = down  or pygame.K_s
        left_key  = left  or pygame.K_a
        right_key = right or pygame.K_d

        # Apply movement
        if keys[up_key]:
            self.pos[1] -= PLAYER_SPEED
        if keys[down_key]:
            self.pos[1] += PLAYER_SPEED
        if keys[left_key]:
            self.pos[0] -= PLAYER_SPEED
        if keys[right_key]:
            self.pos[0] += PLAYER_SPEED

        # Constrain within bounds
        self._clamp(bounds)

    def reset(self, x, y):
        self.pos = np.array([x, y], dtype=np.float32)
        self.prev_pos = self.pos.copy()

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (*self.pos.astype(int), PLAYER_SIZE, PLAYER_SIZE))

    def _clamp(self, rect):
        self.pos[0] = np.clip(self.pos[0], rect.left, rect.right - PLAYER_SIZE)
        self.pos[1] = np.clip(self.pos[1], rect.top, rect.bottom - PLAYER_SIZE)

class Ball:
    def __init__(self, x, y, bounds, is_large_field=False):
        self.bounds = bounds
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = self._random_velocity()
        self.is_large_field = is_large_field

    def _random_velocity(self):
        angle = np.random.uniform(0, 2*np.pi)
        return np.array([BALL_SPEED * np.cos(angle), BALL_SPEED * np.sin(angle)], dtype=np.float32)

    def update(self):
        self.pos += self.vel
        self.vel *= 0.98 if np.linalg.norm(self.vel) > 0.1 else 0

        if self.pos[1] <= self.bounds.top + BALL_RADIUS or self.pos[1] >= self.bounds.bottom - BALL_RADIUS:
            self.vel[1] *= -0.9
            self.pos[1] = np.clip(self.pos[1], self.bounds.top + BALL_RADIUS, self.bounds.bottom - BALL_RADIUS)
        if self.pos[0] <= self.bounds.left + BALL_RADIUS or self.pos[0] >= self.bounds.right - BALL_RADIUS:
            self.vel[0] *= -0.9
            self.pos[0] = np.clip(self.pos[0], self.bounds.left + BALL_RADIUS, self.bounds.right - BALL_RADIUS)

    def collide_with_player(self, player):
        offset = PLAYER_SIZE / 2
        dist = np.linalg.norm(self.pos - (player.pos + offset))
        if dist < BALL_RADIUS + offset:
            normal = (self.pos - (player.pos + offset)) / dist
            self.vel += normal * BALL_SPEED * 0.6

    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, self.pos.astype(int), BALL_RADIUS)

    def reset(self):
        self.pos = np.array([WIDTH//2, HEIGHT//2], dtype=np.float32)
        self.vel = self._random_velocity()

    def check_goal(self):
        y = self.pos[1]
        goal_y_offset = 110 if self.is_large_field else 0
        if (HEIGHT//2 - GOAL_WIDTH//2 + goal_y_offset) < y < (HEIGHT//2 + GOAL_WIDTH//2 + goal_y_offset):
            if self.pos[0] <= self.bounds.left + BALL_RADIUS:
                return 2
            elif self.pos[0] >= self.bounds.right - BALL_RADIUS:
                return 1
        return 0