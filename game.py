import time
from datetime import datetime

import numpy as np
import pygame
import sys
import math
import random
import pandas as pd



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
GRID_WIDTH = 10  # Number of grid cells in width
GRID_HEIGHT = 10  # Number of grid cells in height
NUM_CELLS_X = SCREEN_WIDTH // GRID_SIZE
NUM_CELLS_Y = SCREEN_HEIGHT // GRID_SIZE

# x: 26 - y:20
WINDOW_WIDTH = GRID_SIZE * GRID_WIDTH
WINDOW_HEIGHT = GRID_SIZE * GRID_HEIGHT

GOAL = [NUM_CELLS_X-1, NUM_CELLS_Y-1]


# Sprite groups
all_sprites = pygame.sprite.Group()
obstacles = pygame.sprite.Group()
obstacle_positions = [(1*SCREEN_WIDTH//4, GRID_SIZE*i) for i in range(2*NUM_CELLS_Y//3)]


class Cell:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
        self.color = (255, 255, 255)  # default to white
        

    def darken(self):
        self.color = (50, 50, 50)  # dark color
    
    def paint_cell(self):
        self.color = (150, 0 ,0)  # dark color

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)


class Vehicle(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([NUM_CELLS_X, NUM_CELLS_Y])
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.x = 3 #(SCREEN_WIDTH // 2)
        self.rect.y = 3 #SCREEN_HEIGHT //2 
        self.speed = 1
        self.unvisible_area = 0
        self.visited_grid = []
        self.total_steps = 0
        self.start_point = (3,3)

        print("Vehicle start x : %d   y: %d"%(self.rect.x, self.rect.y))

    def reset(self):
        self.rect = self.image.get_rect()
        self.rect.x = 3#(SCREEN_WIDTH // 2)
        self.rect.y = 3#SCREEN_HEIGHT //2
        self.total_steps = 0

    def distance_to(self, point1, point2):

        return ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)**0.5
    
    def get_areareward(self,area):
        start_point = self.start_point
        final_point = GOAL
        state = (self.rect.x, self.rect.y)

        # print(start_point)
        # print(final_point)
        # print(state)
        # print(area)
        # print()
        ES = (final_point[0] - start_point[0], final_point[1] - start_point[1])
        ES2 = ES[0] ** 2 + ES[1] ** 2

        EP = (final_point[0] - state[0], final_point[1] - state[1])
        EP2 = EP[0] ** 2 + EP[1] ** 2
        ES_EP = ES[0] * EP[0] + ES[1] * EP[1]
        ES_EP2 = ES_EP ** 2

        distance = (EP2 - ES_EP2 / ES2)**0.5
        if distance < area:
            area_reward = 0
        else:
            area_reward = -10

        return area_reward
    
    def get_state(self, obstacles, observation, area):
        """ return new_observation, reward, done """

        new_observation = (self.rect.x // GRID_SIZE, self.rect.y // GRID_SIZE)
        p1 = [self.rect.x, self.rect.y]
        # print("p1: ",p1)
        goal_distance = self.distance_to(p1, GOAL)
        # print(GOAL, "  DISTANCE: ", goal_distance)
        # print("AREA : ",area)
        areareward = self.get_areareward(area)

        if goal_distance<3:
            reward = areareward + 10
            done = True
            #observation = 'goal'
        else:
            reward = areareward -1 #- self.unvisible_area - goal_distance #- self.total_steps
            done = False

        if observation == new_observation:
            reward = areareward -3 

        # vehicle_hit_list = pygame.sprite.spritecollide(self, obstacles, False)
        # for hit in vehicle_hit_list:
        if new_observation in obstacles:
            # running = False
            reward = -10
            #new_observation = 'obstacle'
            done = True
            print("OBSTACLE !!!")
            
        return new_observation, reward, done
    
    def update(self, action = None, grid=None):
        # action = ["up", "down", "left", "right","left_up","left_down","right_up","right_down"]
        # action index = [up down left right]
        if action == None:
            return
        # UP - DOWN
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or action==2: 
            self.rect.x = max(0, self.rect.x - self.speed)
        if keys[pygame.K_RIGHT] or action==3:
            self.rect.x = min(SCREEN_WIDTH-1, self.rect.x + self.speed)
        if keys[pygame.K_UP] or action==1:
            self.rect.y = max(0, self.rect.y - self.speed)
        if keys[pygame.K_DOWN] or action==0:
            self.rect.y = min(SCREEN_HEIGHT-1, self.rect.y + self.speed)
        if action==5:
            self.rect.x = max(0, self.rect.x - self.speed) # LEFT
            self.rect.y = max(0, self.rect.y - self.speed) # UP
        if action==4:
            self.rect.x = max(0, self.rect.x - self.speed) # LEFT
            self.rect.y = min(SCREEN_HEIGHT-1, self.rect.y + self.speed) # DOWN
        if action==7:
            self.rect.x = min(SCREEN_WIDTH-1, self.rect.x + self.speed) # RIGHT
            self.rect.y = max(0, self.rect.y - self.speed) # UP
        if action==6:
            self.rect.x = min(SCREEN_WIDTH-1, self.rect.x + self.speed) # RIGHT
            self.rect.y = min(SCREEN_HEIGHT-1, self.rect.y + self.speed) # DOWN
        
        current_cell = grid[self.rect.x//GRID_SIZE][self.rect.y//GRID_SIZE]
        self.visited_grid.append(current_cell)
        if len(self.visited_grid)>30:
            self.visited_grid.pop(0)

        self.total_steps += 1

    def update_new(self, action = None, grid = None):
        
        # action index = [up down left right]
        if action==2: 
            self.rect.x = max(0, self.rect.x - self.speed)
        if action==3:
            self.rect.x = min(NUM_CELLS_X-1, self.rect.x + self.speed)
        if action==0:
            self.rect.y = max(0, self.rect.y - self.speed)
        if action==1:
            self.rect.y = min(NUM_CELLS_Y-1, self.rect.y + self.speed)
        
        current_cell = grid[self.rect.x//GRID_SIZE][self.rect.y//GRID_SIZE]
        self.visited_grid.append(current_cell)
        if len(self.visited_grid)>30:
            self.visited_grid.pop(0)

        self.total_steps += 1

    def draw_path(self, screen):
        
        for i,cell in enumerate(self.visited_grid):
            cell.color = (0, 0, 255)  # draw free window cells
            cell.draw(screen)

    def draw_fov(self, screen, grid, obstacles):
        caster = RayCaster(self.rect.center)
        w_w = WINDOW_WIDTH // GRID_SIZE
        w_h = WINDOW_HEIGHT // GRID_SIZE

        self.unvisible_area = 0
        goal_cell = grid[GOAL[0]][ GOAL[1]]
        goal_cell.color = (0,250,250)
        goal_cell.draw(screen)
        for x in range(-w_w // 2, w_w // 2 + 1):
            for y in range(-w_h // 2, w_h // 2 + 1):
            
                cell_x = x + self.rect.x // GRID_SIZE
                cell_y = y + self.rect.y // GRID_SIZE

                if cell_x >= len(grid) or cell_x <= 0 or cell_y >= len(grid[0]) or cell_y <= 0:
                    pass
                else: 
                    cell = grid[cell_x][cell_y]
                    hit_point = caster.cast_to_point([o.rect for o in obstacles], cell.rect.center)
                    if hit_point:
                        cell.darken()
                        cell.draw(screen)
                        self.unvisible_area += 1
                    else:
                        cell.color = (200, 200, 255)  # draw free window cells
                        cell.draw(screen)
                

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface([GRID_SIZE, GRID_SIZE])
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        


class RayCaster:
    def __init__(self, origin):
        self.origin = origin
    
    def cast_to_point(self, obstacles, cell_point):
        """
        cell_point  : center of casting cell
        self.origin : vehicle center point
        """
        
        ray = pygame.math.Vector2(cell_point) - self.origin
        ray_length = ray.length()
        if ray.length() < 1e-6:  # if the length is close to zero
            return None
        ray.normalize_ip()

        for i in range(0, int(ray_length), 1):
            current_point = self.origin + ray * i
            
            for obs in obstacles:
                if obs.collidepoint(current_point.x, current_point.y):
                    return current_point
                
        return None
    
# grid = [[Cell(x*GRID_SIZE, y*GRID_SIZE) for y in range(NUM_CELLS_Y)] for x in range(NUM_CELLS_X)]
# for pos in obstacle_positions:
#     obstacle = Obstacle(pos[0], pos[1])
#     all_sprites.add(obstacle)
#     obstacles.add(obstacle)
