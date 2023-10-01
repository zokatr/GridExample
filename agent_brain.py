import time
import random
import numpy as np
import matplotlib.pyplot as plt
from game import *
from datetime import datetime


class Agent:
    
    def __init__(self,):
        print("Init Brain")

        self.n = 30
        self.C = []
        self.Y = []
        self.area = 1
        self.alpha = 0.8
        self.gamma = 0.5
        self.epsilon0 = 0.3
        self.episodes = 5000
        self.success_episode = 0
        self.sum_reward_list = []
        self.start_position = (0, 0)
        self.final_position = (29, 29)
        self.current_state = self.start_position
        self.world = [[(i,j) for j in range(self.n)] for i in range(self.n)]
        
        # self.stone_list = [(10,5),(20,15)]
        self.stone_list = [(10, i) for i in range(20)]
        self.action = ["up", "down", "left", "right","left_up","left_down","right_up","right_down"]
        self.q_table = [[[0 for j in range(len(self.world))] for i in range(len(self.world))] for k in range(len(self.action))]
        self.policy = [[0 for j in range(len(self.world))] for i in range(len(self.world))]
        self.matrix = [[[0 for j in range(len(self.world))] for i in range(len(self.world))] for k in range(len(self.action))]
        
        # Reset
        self.save = [self.current_state]
        self.reward_list = []
        self.y_step = 0
        self.corner = 0
        self.actionindex = []
        self.area = 1
        self.epsilon = 0.3

    def plot_world(self, result=None):
        
        plt.figure(figsize=(8.4 / 2.54, 8.4 / 2.54), dpi=600)
        plt.ylim([-1, len(self.world)])
        plt.xlim([-1, len(self.world)])
        x = range(0, len(self.world) + 1, 5)

        plt.xticks(x)
        plt.yticks(x)

        plt.scatter(self.start_position[0], self.start_position[1], s=30, color="green", marker="o")
        plt.scatter(self.final_position[0], self.final_position[1], s=30, color="red", marker="o")
        
        for eve in self.stone_list:
            plt.scatter(eve[0], eve[1], s=30, color="black", marker="s")

        if result != None:

            for i in range(len(result) - 1):
                plt.plot([result[i][0], result[i + 1][0]], [result[i][1], result[i + 1][1]], color="green", linestyle="--")
            plt.savefig("map1.jpg", dpi=600)
            plt.show()

        else:
            plt.savefig("grid.jpg", dpi=600)
            plt.show()

            


    def action_result(self, action_index):
        action = self.action[action_index]
        max_trick = self.n-1
        if action == "up":
            if self.current_state[1] == max_trick:
                return self.current_state
            else:
                return (self.current_state[0], self.current_state[1]+1)
        elif action == "down":
            if self.current_state[1] == 0:
                return self.current_state
            else:
                return (self.current_state[0], self.current_state[1]-1)
        elif action == "left":
            if self.current_state[0] == 0:
                return self.current_state
            else:
                return (self.current_state[0]-1, self.current_state[1])
        elif action == "right":
            if self.current_state[0] == max_trick:
                return self.current_state
            else:
                return (self.current_state[0]+1, self.current_state[1])
        elif action == "left_up":
            if self.current_state[0] == 0 or self.current_state[1] == max_trick:
                return self.current_state
            else:
                return (self.current_state[0]-1,self.current_state[1]+1)
        elif action == "left_down":
            if self.current_state[0] == 0 or self.current_state[1] == 0:
                return self.current_state
            else:
                return (self.current_state[0] - 1, self.current_state[1] - 1)
        elif action == "right_up":
            if self.current_state[0] == max_trick or self.current_state[1] == max_trick:
                return self.current_state
            else:
                return (self.current_state[0] + 1, self.current_state[1] + 1)
        elif action == "right_down":
            if self.current_state[0] == max_trick or self.current_state[1] == 0:
                return self.current_state
            else:
                return (self.current_state[0] + 1, self.current_state[1] - 1)
        else:
            raise IOError

    def get_area(self, area,successepisode,C_list):
        c = 0
        if (successepisode == 0):
            area += 1
        for i in C_list:
            if i > 15:
                c += 1
        if c == len(C_list):
            area += 1

        return area


    def get_areareward(self, start_point,final_point,state,area):
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



    def get_reward(self, state,start_point,final_point, stonelist, currentstate,area):
        areareward = self.get_areareward(start_point,final_point,state,area)

        if state == currentstate:
            return -3+areareward

        if state == final_point:
            return 10+areareward

        elif state in stonelist:
            return -10+areareward
        else:
            return -1+areareward

    def get_maxq(self, qtable, state):
        temp = []
        index = []
        index_list = [0,1,2,3,4,5,6,7]

        for j in range(len(self.matrix)):
            if self.matrix[j][state[0]][state[1]] < 0:
                index.append(j)

        for m in range(len(index)):
            index_list.remove(index[m])

        for i in index_list:
            temp.append(qtable[i][state[0]][state[1]])

        maxone = max(temp)
        arg = np.argmax(temp)
        argmax = index_list[arg]
        return maxone, argmax


    def get_randaction(self, current_matrix,state):
        index02 = []
        index_list02 = [0,1,2,3,4,5,6,7]

        for i in range(len(index_list02)):
            if current_matrix[i][state[0]][state[1]] < 0:
                index02.append(i)

        for j in range(len(index02)):
            index_list02.remove(index02[j])

        index = random.randint(0, len(index_list02) - 1)
        actionindex = index_list02[index]

        return actionindex

    def get_epsilon(self,fuction_epsilon0,function_episode,function_episodes):

        if function_episode <= function_episodes*0.1854:
            epsilon = fuction_epsilon0*(1-function_episode/(function_episodes*0.1854))**2
        else:
            epsilon = 0.01

        return epsilon

    def reset(self):
        self.current_state = self.start_position
        self.save = [self.current_state]

        self.reward_list = []
        self.y_step = 0
        self.corner = 0
        self.actionindex = []
        self.area = self.get_area(self.area, self.success_episode, self.C)
        self.epsilon = self.get_epsilon(self.epsilon0, episode, self.episodes)

    def run(self):
        while True:
            if random.randint(1,100)/100 > self.epsilon:
                action_index = self.policy[self.current_state[0]][self.current_state[1]]
            else:
                action_index = self.get_randaction(self.matrix,self.current_state)

            next_state = self.action_result(action_index)
            reward = self.get_reward(next_state, self.start_position, self.final_position, self.stone_list, self.current_state, self.area)

            maxone, _ = self.get_maxq(self.q_table, next_state)
            self.q_table[action_index][self.current_state[0]][self.current_state[1]] += self.alpha*(reward + self.gamma*maxone - self.q_table[action_index][self.current_state[0]][self.current_state[1]])

            _, argmax = self.get_maxq(self.q_table, self.current_state)
            self.policy[self.current_state[0]][self.current_state[1]] = argmax


            if next_state == self.current_state:
                self.y_step += 0
            elif action_index in [0, 1, 2, 3]:
                self.y_step += 1.00
            elif action_index in [4, 5, 6, 7]:
                self.y_step += 1.41

            self.actionindex.append(action_index)

            test_temp = self.current_state
            self.current_state = next_state
            self.save.append(self.current_state)
            self.reward_list.append(reward)

            if reward == -10 or reward == -20:
                self.matrix[action_index][test_temp[0]][test_temp[1]] = -1

            if reward == 10:
                sum_reward = 0
                for i in self.reward_list:
                    sum_reward += i

                self.sum_reward_list.append(sum_reward)

                for i in range(len(self.actionindex) - 1):
                    if self.actionindex[i + 1] != self.actionindex[i]:
                        self.corner += 1

                num = 0
                for i in self.save:
                    if i not in self.stone_list:
                        num += 1
                    if num == len(self.save):
                        self.success_episode += 1
                        self.C.append(self.corner)

                self.Y.append(self.y_step)

                break

        

    def inference(self):
        
        state = self.start_position
        res = [state]
        for i in range(100):
            a_index = self.policy[state[0]][state[1]]
            next_state = self.action_result(a_index)
            res.append(next_state)

            if next_state == self.final_position or True:
                print(" ... Final Position ...")
                self.plot_world(res)
                break
            state = next_state

if __name__ == "__main__":

    agent = Agent()
    learn = True
    for episode in range(agent.episodes):
        
        # RESET
        print("Episode : " ,episode)
        agent.reset()
        counter = 0
        
        agent.run()

    agent.inference()
            
    
