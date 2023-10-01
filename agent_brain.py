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
        self.alpha = 0.9
        self.gamma = 0.7
        self.epsilon0 = 10.1
        self.episodes = 5000
        self.success_episode = 0
        self.sum_reward_list = []
        self.start_position = (0, 0)
        self.final_position = (29, 29)
        self.current_state = self.start_position
        self.world = [[(i,j) for j in range(self.n)] for i in range(self.n)]
        
        # self.stone_list = [(10,5),(20,15)]
        self.stone_list = [(10, i) for i in range(25)]
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


    def action_result(self, action_index, state):
        action = self.action[action_index]
        max_trick = self.n-1
        if action == "up":
            if state[1] == max_trick:
                return state
            else:
                return (state[0], state[1]+1)
        elif action == "down":
            if state[1] == 0:
                return state
            else:
                return (state[0], state[1]-1)
        elif action == "left":
            if state[0] == 0:
                return state
            else:
                return (state[0]-1, state[1])
        elif action == "right":
            if state[0] == max_trick:
                return state
            else:
                return (state[0]+1, state[1])
        elif action == "left_up":
            if state[0] == 0 or state[1] == max_trick:
                return state
            else:
                return (state[0]-1,state[1]+1)
        elif action == "left_down":
            if state[0] == 0 or state[1] == 0:
                return state
            else:
                return (state[0] - 1, state[1] - 1)
        elif action == "right_up":
            if state[0] == max_trick or state[1] == max_trick:
                return state
            else:
                return (state[0] + 1, state[1] + 1)
        elif action == "right_down":
            if state[0] == max_trick or state[1] == 0:
                return state
            else:
                return (state[0] + 1, state[1] - 1)
        else:
            raise IOError

    def get_area(self):
        area = self.area
        successepisode = self.success_episode
        C_list = self.C

        c = 0
        if (successepisode == 0):
            area += 1
        for i in C_list:
            if i > 15:
                c += 1
        if c == len(C_list):
            area += 1

        return area


    def get_areareward(self, next_state):
        
        ES = (self.final_position[0] - self.start_position[0], self.final_position[1] - self.start_position[1])
        ES2 = ES[0] ** 2 + ES[1] ** 2

        EP = (self.final_position[0] - next_state[0], self.final_position[1] - next_state[1])
        EP2 = EP[0] ** 2 + EP[1] ** 2
        ES_EP = ES[0] * EP[0] + ES[1] * EP[1]
        ES_EP2 = ES_EP ** 2

        distance = (EP2 - ES_EP2 / ES2)**0.5
        if distance < self.area:
            area_reward = 0
        else:
            area_reward = -10

        return area_reward



    def get_reward(self, next_state):
        areareward = self.get_areareward(next_state)
        # print("areareward : ",areareward)
        if next_state == self.current_state:
            return -3+areareward

        if next_state == self.final_position:
            return 10+areareward

        elif next_state in self.stone_list:
            return -10+areareward
        else:
            return -1+areareward

    
    def get_maxq(self, state):
        """ current ve next icin farkli olabilir """
        temp = []
        index = []
        index_list = [0,1,2,3,4,5,6,7]

        for j in range(len(self.matrix)):
            if self.matrix[j][state[0]][state[1]] < 0:
                index.append(j)

        for m in range(len(index)):
            index_list.remove(index[m])

        for i in index_list:
            temp.append(self.q_table[i][state[0]][state[1]])

        maxone = max(temp)
        arg = np.argmax(temp)
        argmax = index_list[arg]
        return maxone, argmax


    def get_randaction(self):
        
        index02 = []
        index_list02 = [0,1,2,3,4,5,6,7]

        for i in range(len(index_list02)):
            if self.matrix[i][self.current_state[0]][self.current_state[1]] < 0:
                index02.append(i)

        for j in range(len(index02)):
            index_list02.remove(index02[j])

        index = random.randint(0, len(index_list02) - 1)
        actionindex = index_list02[index]

        return actionindex

    def get_epsilon(self,fuction_epsilon0,function_episode,function_episodes):
        
        if function_episode < function_episodes*0.1854:
            epsilon = fuction_epsilon0*(1-function_episode/(function_episodes*0.1854))**2
        else:
            epsilon = 0.01

        return max(0.01,epsilon)

    def reset(self, episode):
        self.current_state = self.start_position
        self.save = [self.current_state]

        self.reward_list = []
        self.y_step = 0
        self.corner = 0
        self.actionindex = []
        # self.area = self.get_area(self.area, self.success_episode, self.C)

        if (episode % 100 == 0):
            self.area = self.get_area()

        self.epsilon = self.get_epsilon(self.epsilon0, episode, self.episodes)

    def run(self, episode):
        
        counter = 0
        while True:
            counter +=1
            if random.randint(1,100)/100 > self.epsilon:
                action_index = self.policy[self.current_state[0]][self.current_state[1]]
            else:
                action_index = self.get_randaction()

            next_state = self.action_result(action_index, self.current_state)
            reward = self.get_reward(next_state)
            
            maxone, _ = self.get_maxq(next_state)
            self.q_table[action_index][self.current_state[0]][self.current_state[1]] += self.alpha*(reward + self.gamma*maxone - self.q_table[action_index][self.current_state[0]][self.current_state[1]])

            _, argmax = self.get_maxq(self.current_state)
            self.policy[self.current_state[0]][self.current_state[1]] = argmax

            # @TODO: GEREKLİ Mİ ?
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


            if reward == -10 or reward <= -20:
                self.matrix[action_index][test_temp[0]][test_temp[1]] = -1
                # arw = self.get_areareward(self.current_state)
                # print("Episode : " ,episode, "   area : ",self.area, "  REWARD: ",reward, "  ARW: ",arw)
                break

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
                arw = self.get_areareward(self.current_state)
                print("Episode : " ,episode, "   area : ",self.area, "  REWARD: ",reward, "  ARW: ",arw)
                break

        

    def inference(self):
        
        state = self.start_position
        res = [state]
        for i in range(100):
            a_index = self.policy[state[0]][state[1]]
            # print("state : ",state)
            # print("a_index: ",a_index)
            
            next_state = self.action_result(a_index, state)
            res.append(next_state)
            # print("....  ",i," - ", a_index)
            if next_state == self.final_position:
                # print("next_state: ",next_state , "  self.final_position: ",self.final_position)
                print(" ... Final Position ...")
                self.plot_world(res)
                break
            state = next_state
            

if __name__ == "__main__":

    agent = Agent()
    learn = True
    for episode in range(agent.episodes):
        
        # RESET
        
        agent.reset(episode)
        if episode % 100 == 1:
            agent.inference()
        print("Episode : " ,episode, "   area : ",agent.area, "  REWARD: ",agent.get_reward(agent.current_state), " epsilon: ",agent.epsilon)
        agent.run(episode)



    agent.inference()
            
    
# 758 0.0099