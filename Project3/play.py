from game import HexGame
from mcts import MonteCarloTreeSearch
from ANN import ANN
from CNN import CNN
import math
import numpy as np

def play(mcts, sim, ann, env, top_moves):
    while True:
        print("\n{} to move".format("Red" if env.player == 1 else "Black"))
        D = mcts.search(env, sim)
        best_mcts_move = np.argmax(D)
        probs, model_move, index = ann.get_move(env)

        mcts_moves = {i: p for i, p in enumerate(D)}
        sorted_mcts = {k: v for k, v in sorted(mcts_moves.items(), key=lambda item: item[1])}
        val = {i: p for i, p in enumerate(probs)}
        sorted_moves = {k: v for k, v in sorted(val.items(), key=lambda item: item[1])}
        print("Top {} ANN moves: \t Top MCTS moves:".format(top_moves, top_moves))
        for i in range(1,6):
            model_move = list(sorted_moves.keys())[-i]
            mcts_move = list(sorted_mcts.keys())[-i]
            print("{:>2}: {:>5.2f}% \t {:>10}: {:>5.2f}%".format(model_move, sorted_moves[model_move] * 100,
                                                             mcts_move, sorted_mcts[mcts_move] * 100))
        print("\nMCTS would have chosen: {}, ({:.2f}% confidence)".format(best_mcts_move, D[best_mcts_move]*100))
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
    level = 200

    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.001  # learning rate
    H_dims = [128,64]
    io_dim = board_size * board_size  # input and output layer sizes
    activation = activation_functions[3]
    optimizer = optimizers[3]
    epochs = 500

    #ann = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
    #ann.load(board_size, level)
    cnn = CNN(board_size, alpha, epochs, activation, optimizer)
    cnn.load(board_size, level)

    sim = 500
    mcts = MonteCarloTreeSearch(cnn, c=1., eps=1)
    env = HexGame(board_size)
    top_moves = 5

    play(mcts, sim, cnn, env, top_moves)
