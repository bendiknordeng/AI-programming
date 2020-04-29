from game import HexGame
from mcts import MonteCarloTreeSearch
from CNN import CNN
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
        for i in range(1,top_moves+1):
            model_move = list(sorted_moves.keys())[-i]
            mcts_move = list(sorted_mcts.keys())[-i]
            print("{:>2}: {:>5.2f}% \t {:>10}: {:>5.2f}%".format(model_move, sorted_moves[model_move] * 100,
                                                             mcts_move, sorted_mcts[mcts_move] * 100))
        print("\nThe model would have chosen: {}, ({:.2f}% confidence)".format(index, probs[index]*100))
        print("MCTS would have chosen: {}, ({:.2f}% confidence)\n".format(best_mcts_move, D[best_mcts_move]*100))
        env.draw()
        while True:
            i = input("Choose move (press enter for model move): ")
            if i == '':
                move = index
            elif i == ' ':
                move = best_mcts_move
            else:
                move = int(i)
            print("\nChose move {}".format(move))
            try:
                env.move(env.all_moves[move])
                break
            except:
                print("Invalid move, try again")
                env.draw()
                continue
        winning_path = env.is_game_over()
        if winning_path:
            break
    print("\nPlayer", 3 - env.player, "won")
    env.draw(path=winning_path)


if __name__ == '__main__':
    board_size = 5
    level = 500

    cnn = CNN(board_size)
    cnn.load(board_size, level)

    sim = 10
    mcts = MonteCarloTreeSearch(cnn, c=1.4, eps=1)
    env = HexGame(board_size)
    top_moves = 5

    play(mcts, sim, cnn, env, top_moves)
