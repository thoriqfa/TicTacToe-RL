import numpy as np
from collections import defaultdict
import random


class q_learning(object):
    '''q learning agent for playing tic tac toe'''
    def __init__(self, epsilon, alpha, gamma, episodes):
        super().__init__()

        self.q_value = defaultdict(lambda: 0)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.episodes = episodes

    def avail_actions(self, state):
        actions = [i for i, v in enumerate(state) if v == 0]
        return actions

    def stateToStr(self, state):
        '''turns state represented in array into string'''
        s = ""
        for i in state:
            s += str(i)
        return s

    def choose_action(self, state, episode = 0, eps_reduc = False):
        actions = self.avail_actions(state)
        action = random.choice(actions)

        #print("Player 1 availabe action: ", actions)

        if eps_reduc and episode % (0.1 * self.episodes) == 0:
            self.epsilon = 0.7 * self.epsilon

        if np.random.rand() > self.epsilon:
            stateStr = self.stateToStr(state)
            q_val_of_state = []
            for a in actions:
                q_val_of_state.append(self.q_value[(stateStr, a)])
            #extracting action with highest value in case of multiple occurrence
            highest_value = max(q_val_of_state)
            highest_actions = [i for i, v in enumerate(q_val_of_state) if v == highest_value]
            action = random.choice([actions[i] for i in highest_actions])
        
        return action
    
    def learn(self, state, action, reward, next_state):
        state1Str = self.stateToStr(state)
        state2Str = self.stateToStr(next_state)
        #print("State: ", state)
        #print("Action: ", action)
        #print("Reward: ", reward)
        #print("Next state: ", next_state)
        #choosing max q value over state-actions pairs of state2
        q_val_of_state2 = []
        actions2 = self.avail_actions(next_state)
        for a2 in actions2:
            q_val_of_state2.append(self.q_value[(state2Str, a2)])
        if len(q_val_of_state2) == 0:
            max_q = reward
        else:
            max_q = max(q_val_of_state2)
        
        #update q value
        self.q_value[(state1Str, action)] += self.alpha * (reward + self.gamma * max_q - self.q_value[(state1Str, action)])


class random_comp(object):
    def __init__(self):
        super().__init__()

    def choose_action(self, state):
        actions = [i for i, v in enumerate(state) if v == 0]
        #print("Player 2 availabe action: ", actions)
        if len(actions) == 0:
            return -1
        return random.choice(actions)

class first_available_pos(object):
    def __init__(self):
        super().__init__()
    
    def choose_action(self, state):
        actions = [i for i, v in enumerate(state) if v == 0]
        if len(actions) == 0:
            return -1
        return actions[0]