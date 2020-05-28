from tictactoe_game import Tictactoe
from agents import q_learning
from agents import random_comp
import matplotlib.pyplot as plt
import numpy as np

episodes = 10000

game = Tictactoe()
player1 = q_learning(epsilon=0.5, alpha=0.6, gamma=1, episodes = episodes)
player2 = random_comp()

#do training
results = []
for episode in range(episodes):
    winner = 0
    game_done = False
    state = game.reset()
    while not game_done:
        #q learning agent choosing move
        player1_action = player1.choose_action(state, episode, eps_reduc=True)
        #print("Player 1 move: ", player1_action)
        #q learning agent make the move
        reward, new_state, game_done, winner = game.player1_mark(player1_action)
        #print("State: ", game.state)
        #game.print_board()
        
        
        #comp agent move using state after q learning agent move
        player2_action = player2.choose_action(new_state)
        #print("Player 2 move: ", player2_action)
        reward, next_state, game_done, winner = game.player2_mark(player2_action)
        #print("State: ", game.state)
        #game.print_board()

        player1.learn(state, player1_action, reward, next_state)
        state = next_state

    if winner == 1:
        print("Win")
        results.append(1)
    elif winner == 2:
        print("Lost")
        results.append(-1)
    else:
        print("Draw")
        results.append(0)

    #game.print_board()
    #print("Episode ", i, " passed.")

#print(player1.q_value)

q_file = open(("Q_val_"+str(episodes)+"_10_eprd.txt"), "w")
q_file.write("Num of state-action pair: "+str(len(player1.q_value))+"\n")
q_file.write(str(player1.q_value))
q_file.close()

x_axis = np.linspace(10, episodes, int(episodes/10), dtype=int)
results_10 = [sum(results[i:i+10]) for i in range(0, episodes, 10)]
#print("Results_10: ", results_10)
plt.plot(x_axis, results_10)
plt.show()

print("Testing for git")

play = 2
for i in range(play):
    game_done = False
    state = game.reset()
    winner = 0
    print("Game begins!")
    while not game_done:
        print("Agent move")
        player1_action = player1.choose_action(state)
        game.player1_mark(player1_action)
        
        reward, new_state, game_done, winner = game.observation()
        game.print_board()
        if game_done:
            break

        player2_action = int(input('Enter space to occupy: '))
        print("Your move")
        game.player2_mark(player2_action)
        
        reward, new_state, game_done, winner = game.observation()
        game.print_board()
        state = new_state
    if winner == 1:
        print("Q Learning agent wins!")
    elif winner == 2:
        print("You win!")
    else:
        print("Game Draws")
