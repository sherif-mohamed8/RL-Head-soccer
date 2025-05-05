import pygame
from game import SoccerGame

def main():
    # Initialize game with Q-learning enabled and training mode on
    game = SoccerGame(use_q_learning=True, training_mode=True)
    
    running = True
    while running:
        running = game.handle_events()
        game.update()
        game.render()
    
    game.quit()

if __name__ == "__main__":
    main()