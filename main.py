import time
import pygame
from agent_brain import Agent
from game import Cell, RayCaster, Vehicle, Obstacle

# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Constants
NUM_OBSTACLE = 10
GRID_SIZE = 20  # size of each grid cell in pixels
NUM_CELLS_X = SCREEN_WIDTH // GRID_SIZE
NUM_CELLS_Y = SCREEN_HEIGHT // GRID_SIZE

GOAL = [NUM_CELLS_X-1, NUM_CELLS_Y-1]



if __name__ == "__main__":

    agent = Agent()
    vehicle = Vehicle()
    learn = True
    

    for episode in range(agent.episodes):
        
        # RESET
        agent.reset(episode)
        # if episode % 5000 == 1:
        #     agent.inference()
        print("Episode : " ,episode, "   area : ",agent.area, "  REWARD: ",agent.get_reward(agent.current_state), " epsilon: ",agent.epsilon)
        agent.run(episode)

    agent.inference()
    agent.reset(episode)
    
    #-------------------------------------------------------------
    #-------------------------------------------------------------
    # ------------------- START GAME -----------------------------
    #-------------------------------------------------------------
    #-------------------------------------------------------------
    
    # Set up the screen and clock
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("2D Vehicle Game with Observable Space")
    clock = pygame.time.Clock()
    all_sprites = pygame.sprite.Group()
    obstacles = pygame.sprite.Group()
    
    grid = [[Cell(x*GRID_SIZE, y*GRID_SIZE) for y in range(NUM_CELLS_Y)] for x in range(NUM_CELLS_X)]
    for pos in agent.stone_list:
        obstacle = Obstacle(pos[0]*GRID_SIZE, pos[1]*GRID_SIZE)
        all_sprites.add(obstacle)
        obstacles.add(obstacle)

    all_sprites.add(vehicle)
    
    a_index = agent.policy[agent.current_state[0]][agent.current_state[1]]
    next_state = agent.action_result(a_index, agent.current_state)
    reward = agent.get_reward(next_state)

    
    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Close ...")
                running = False

        # Choose Action
        a_index = agent.policy[vehicle.rect.x//GRID_SIZE][vehicle.rect.y//GRID_SIZE]
        # next_state = agent.action_result(a_index, agent.current_state)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            a_index=2
        if keys[pygame.K_RIGHT]:
            a_index=3
        if keys[pygame.K_UP]:
            a_index=1#0
        if keys[pygame.K_DOWN]:
            a_index=0#1
        # Buna gerek olmaya bilir biz zaten bu loop icinde her dongude bir step gitmis oluyoruz

        vehicle.update(a_index, grid) #env.step(action)
        new_observation, reward, done = vehicle.get_state(agent.stone_list, vehicle.rect, agent.area)
        # Render Game
        screen.fill(WHITE)
        vehicle.draw_path(screen)
        vehicle.draw_fov(screen, grid, obstacles)
        all_sprites.draw(screen)
      
        # Update 
        # observation = new_observation
        print("new_observation : ",new_observation)
        print("GOAL : ",tuple(GOAL))
        if new_observation == tuple(GOAL):
            agent.reset(1)
            vehicle.reset()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    # sys.exit()