import chess
import chess.engine
from stockfish import Stockfish


class ChessBBModel:

    def __init__(self, env, model_path):
        self.model_path = model_path
        self.env = env
        self.stockfish = Stockfish(path=self.model_path, depth=10, parameters={"Threads": 2})

    def predict(self, x):
        fen = self.env.from_array_to_fen(x)
        board = chess.Board(fen)

        try:
            self.stockfish.set_fen_position(board.fen())
            action = self.stockfish.get_best_move_time(1000)
        except:
            self.stockfish = Stockfish(path=self.model_path, depth=10, parameters={"Threads": 2})
            action = ''

        return action