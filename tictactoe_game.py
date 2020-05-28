import random
import numpy as np

class Tictactoe(object):  
        
    def __init__(self):
        #array for state representation
        self.state = np.zeros(9, dtype = "int")
        #self.actions = [i for i in range(len(self.state))]
        
        self.game_done = False
        self.player1_win = False
        self.player2_win = False
    
    def avail_actions(self):
        return [index for index, value in enumerate(self.state) if value == 0]

    def player1_mark(self, position):
        if position == -1:
            self.game_done = True
        else:
            self.state[position] = 1
            self.check_win()
        (reward, next_state, game_done, winner) = self.observation()
        return (reward, next_state, game_done, winner)

    def player2_mark(self, position):
        if position == -1:
            self.game_done = True
        else:
            self.state[position] = 2
            self.check_win()
        (reward, next_state, game_done, winner) = self.observation()
        return (reward, next_state, game_done, winner)

    def check_win(self):
        '''Checks for win'''
        #Checks for horizontal/row win
        repr = self.state.reshape(3,3)
        for i in range(3):
            if np.all(repr[i,:] == 1):
                self.player1_win = True
                self.game_done = True
            if np.all(repr[i,:] == 2):
                self.player2_win = True
                self.game_done = True
    
        #Checkes for vertical/column win
        for i in range(3):
            if np.all(repr[:,i] == 1):
                self.player1_win = True
                self.game_done = True
            if np.all(repr[:,i] == 2):
                self.player2_win = True
                self.game_done = True

        #Checks for diagonal win
        #check 1st diagonal
        S = np.zeros(3, dtype = int)
        for i in range(3):
            S[i] = repr[i,i] 
        if np.all(S == 1):
            self.player1_win = True
            self.game_done = True
        elif np.all(S == 2):
            self.player2_win = True
            self.game_done = True
        else:
            #check 2nd diagonal
            S = np.zeros(3, dtype = int)
            for i in range(3):
                S[i] = repr[i,2-i]
            if np.all(S==1):
                self.player1_win = True
                self.game_done = True
            elif np.all(S == 2):
                self.player2_win = True
                self.game_done = True

    def reset(self):
        self.state = np.zeros(9, dtype = "int")
        self.game_done = False
        self.player1_win = False
        self.player2_win = False
        return self.state.copy()

    def observation(self):
        
        #defines reward assuming learning agent is player1
        reward = 0              #applies when draw or game not yet ended
        winner = 0
        if self.player1_win:
            reward = 1
            winner = 1
        elif self.player2_win:
            reward = -1
            winner = 2
        
        if len(self.avail_actions()) == 0:
            self.game_done = True

        return (reward, self.state.copy(), self.game_done, winner)

    def print_board(self):
        board ='''
                         |     |   
                      {a}  |  {b}  |  {c}
                        0|    1|    2
                    -----------------
                         |     |
                      {d}  |  {e}  |  {f}
                        3|    4|    5
                    -----------------
                         |     |
                      {g}  |  {h}  |  {i}
                        6|    7|    8  
                '''    
        icon = ""
        for pos in list(self.state):
            if pos == 1:
                icon += "X"
            elif pos == 2:
                icon += "O"
            else:
                icon += " "
        #print("Icon: ", icon)
        #print(len(icon))
        board = board.format(a = icon[0], b = icon[1], c = icon[2], d = icon[3], e = icon[4], f = icon[5], g = icon[6], h = icon[7], i = icon[8])
        print(board)