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


#@TODO: 200 ep'den beri buluyor dolayi ile ara ara inference yapilabilir daha hizli görebilmek icin sonuclari


if __name__ == "__main__":

    agent = Agent()
    vehicle = Vehicle()
    learn = True
    
    for episode in range(agent.episodes):
        # RESET
        agent.reset(episode)
        agent.run(episode)

    for row in agent.raycast_list:
            print(' '.join([f'{value:<5.2f}' for value in row]))
    agent.inference()
    agent.reset(episode)
    

    # Set up the screen and clock
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("2D Vehicle Game with Observable Space")
    clock = pygame.time.Clock()
    all_sprites = pygame.sprite.Group()
    obstacles = pygame.sprite.Group()
    
    # Init Sprites
    grid = [[Cell(x*GRID_SIZE, y*GRID_SIZE) for y in range(NUM_CELLS_Y)] for x in range(NUM_CELLS_X)]
    for pos in agent.stone_list:
        obstacle = Obstacle(pos[0]*GRID_SIZE, pos[1]*GRID_SIZE)
        all_sprites.add(obstacle)
        obstacles.add(obstacle)
    all_sprites.add(vehicle)
    
    # First Step
    a_index = agent.policy[agent.current_state[0]][agent.current_state[1]]
    next_state = agent.action_result(a_index, agent.current_state)
    reward = agent.get_reward(next_state)
    
    running = True

    # for row in agent.raycast_list:
    #     for value in row:
    #         print(value, end=" ")
    #     print()
    # time.sleep(10)

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Close ...")
                running = False

        # Choose Action
        a_index = agent.policy[vehicle.rect.x//GRID_SIZE][vehicle.rect.y//GRID_SIZE]
        next_state = agent.action_result(a_index, agent.current_state)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            a_index=2
        elif keys[pygame.K_RIGHT]:
            a_index=3
        elif keys[pygame.K_UP]:
            a_index=1#0
        elif keys[pygame.K_DOWN]:
            a_index=0#1
        # else:
        #     a_index=-1

        if a_index>-1:
            vehicle.update(a_index, grid) #env.step(action)
            
        new_observation, reward, done = vehicle.get_state(agent.stone_list, vehicle.rect, agent.area)
        # Render Game
        screen.fill(WHITE)
        vehicle.draw_path(screen)
        vehicle.draw_fov(screen, grid, obstacles)
        all_sprites.draw(screen)
      
        # Update 
        observation = new_observation

        if new_observation == tuple(GOAL):
            agent.reset(1)
            vehicle.reset()

        # print("vehicle.rect    : ",vehicle.rect.x, " , ",vehicle.rect.y)
        # print("vehicle.rect    : ",vehicle.rect.x//GRID_SIZE, " , ",vehicle.rect.y//GRID_SIZE)
        # print("unvisible area  : ",vehicle.unvisible_area)
        # print("raycast_list    : ",agent.raycast_list[vehicle.rect.x//GRID_SIZE][vehicle.rect.y//GRID_SIZE])
        # print("raycast_list    : ",agent.raycast_list[agent.current_state[0]][agent.current_state[1]])
        # print()
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()
    # sys.exit()