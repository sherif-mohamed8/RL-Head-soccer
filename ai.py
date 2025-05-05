import numpy as np
from settings import *

def ai_move(player, ball):
    offset = PLAYER_SIZE / 2
    target = ball.pos - offset
    to_ball = target - player.pos
    dist = np.linalg.norm(to_ball)

    if dist > 10:
        direction = to_ball / dist
        player.pos += direction * PLAYER_SPEED * 0.9
    else:
        # Ball is too close, move slightly in random direction to free it
        player.pos += np.random.uniform(-1, 1, 2) * 2
