from game import HexGame
from mcts import MonteCarloTreeSearch
from ANN import ANN
import math
import numpy as np

def play(mcts, sim, ann, env, top_moves):
    while True:
        print("\n{} to move".format("Red" if env.player == 1 else "Black"))
        D = mcts.search(env, sim)
        best_mcts_move = np.argmax(D)
        probs, model_move, index = ann.get_move(env)
        val = dict(zip(np.arange(env.size**2), probs))
        sorted_moves = {k: v for k, v in sorted(val.items(), key=lambda item: item[1])}
        print("Top {} ANN moves:".format(top_moves))
        for move in list(sorted_moves.keys())[-1:-top_moves-1:-1]:
            print("{:>2}: {:>5.2f}%".format(move, sorted_moves[move]*100))
        print("MCTS would have chosen: {}, ({:.2f}% confidence)".format(best_mcts_move, D[best_mcts_move]*100))
        print("The model would have chosen: {}, ({:.2f}% confidence)".format(index, probs[index]*100))
        env.draw()
        i = input("Choose move (press enter for model move): ")
        if i == '':
            move = index
        elif i == ' ':
            move = best_mcts_move
        else:
            move = int(i)
        print("Chose move {}".format(move))
        env.move(env.all_moves[move])
        winning_path = env.is_game_over()
        if winning_path:
            break
    print("Player", 3 - env.player, "won")
    env.draw(path=winning_path)


if __name__ == '__main__':
    board_size = 5
    level = 400

    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.01  # learning rate
    H_dims = [math.floor(2*(1+board_size**2)/3)+board_size**2] * 3
    io_dim = board_size * board_size  # input and output layer sizes
    activation = activation_functions[3]
    optimizer = optimizers[3]
    epochs = 10

    ann = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
    ann.load(board_size, level)

    sim = 2000
    mcts = MonteCarloTreeSearch(ann, c=1.4, eps=1)
    env = HexGame(board_size)
    top_moves = 5

    play(mcts, sim, ann, env, top_moves)
