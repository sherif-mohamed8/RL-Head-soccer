import pygame

# Initialize pygame to reference key constants
pygame.init()

# Field sizes
NORMAL_WIDTH = 800
NORMAL_HEIGHT = 500
LARGE_WIDTH = 1200
LARGE_HEIGHT = 750

# Current field size (can be toggled between normal and large)
WIDTH = NORMAL_WIDTH
HEIGHT = NORMAL_HEIGHT

# Game constants
PLAYER_SIZE = 40
BALL_RADIUS = 18
PLAYER_SPEED = 5
BALL_SPEED = 4
FPS = 60
WINNING_SCORE = 5
GOAL_WIDTH = 100

# Colors (R, G, B)
WHITE  = (255, 255, 255)
BLACK  = (30,  30,  30)
RED    = (200,  30,  30)
BLUE   = (30,  30, 200)
GREEN  = (0, 150,   0)
YELLOW = (255, 255,   0)

# Game states
STATE_TEAM_SELECT = 'team_select'
STATE_PLAYING     = 'playing'
STATE_GAME_OVER   = 'game_over'

# Player control mappings
# Player 1 (red) uses WASD keys
PLAYER1_CONTROLS = {
    'up':    pygame.K_w,
    'down':  pygame.K_s,
    'left':  pygame.K_a,
    'right': pygame.K_d,
}
# Player 3 (yellow) uses arrow keys
PLAYER3_CONTROLS = {
    'up':    pygame.K_UP,
    'down':  pygame.K_DOWN,
    'left':  pygame.K_LEFT,
    'right': pygame.K_RIGHT,
}
