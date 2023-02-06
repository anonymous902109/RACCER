import gym
import chess
import chess.engine
import numpy as np
from stockfish import Stockfish


class ChessEnv(gym.Env):

    def __init__(self, expert):
        self.board = chess.Board()
        self.state = self.from_fen_to_array(self.board.fen())

        self.state_dim = 64 + 1

        self.lows = np.zeros((self.state_dim, ))
        self.highs = np.array([12] * self.state_dim)

        self.expert = Stockfish(path='trained_models/stockfish_15.exe', depth=10, parameters={"Threads": 2})

    def step(self, action):
        if isinstance(action, str):
            action = chess.Move.from_uci(action)

        self.board.push(action)

        # play for the opponent -- source of stochasticity
        possible_opp_moves = self.get_actions(self.board)

        if len(possible_opp_moves) == 0:
            rew = self.board.is_checkmate()
            return self.state, rew*1000, True, {}

        opp_move = np.random.choice(possible_opp_moves)
        if isinstance(opp_move, str):
            opp_move = chess.Move.from_uci(opp_move)

        self.board.push(opp_move)

        new_state = self.from_fen_to_array(self.board.fen())

        done = self.board.is_game_over()

        rew = self.calculate_reward(self.state, new_state)

        self.state = new_state

        return self.state, rew, done, {}

    def render(self):
        print(self.board)

    def render_state(self, state):
        if not isinstance(state, str):
            state = self.from_array_to_fen(state)

        render_board = chess.Board(state)
        print(render_board)

    def reset(self):
        self.board = chess.Board()
        self.state = self.from_fen_to_array(self.board.fen())

        return self.state

    def close(self):
        pass

    def realistic(self, x):
        fen = self.from_array_to_fen(x)
        board = chess.Board(fen)

        return board.is_valid()

    def actionable(self, x, fact):
        return True

    def generate_state_from_json(self, json_dict):
        fen = json_dict['fen']

        state = self.from_fen_to_array(fen)

        return state

    def get_actions(self, state):
        if isinstance(state, chess.Board):
            board = state
        else:
            fen = self.from_array_to_fen(state)
            board = chess.Board(fen)

        legal_moves = board.legal_moves
        legal_actions = [m.uci() for m in legal_moves]
        return legal_actions

    def set_state(self, state):
        fen = self.from_array_to_fen(state)
        self.board = chess.Board(fen)
        self.state = state

    def check_done(self, state):
        fen = self.from_array_to_fen(state)
        board = chess.Board(fen)

        return board.is_game_over()

    def equal_states(self, s1, s2):
        return sum(s1.squeeze() != s2.squeeze())

    def calculate_reward(self, state, new_state):
        old_fen = self.from_array_to_fen(state)
        new_fen = self.from_array_to_fen(new_state)


        self.expert.set_fen_position(old_fen)

        win_old = self.expert.get_wdl_stats()[0] / 1000.0

        self.expert.set_fen_position(new_fen)

        try:
            win_new = self.expert.get_wdl_stats()
            win_new = win_new[0] / 1000.0
        except TypeError:
            board = chess.Board(new_fen)
            if board.is_checkmate():
                win_new = 0
            else:
                win_new = 0.5

        return (win_old - win_new + 1) * 0.5

    def from_array_to_fen(self, array):
        board = chess.Board(fen=None)  # initializing empty board

        # filling the board with pieces
        for i, sn in enumerate(chess.SQUARE_NAMES):
            if array[i] != 0:
                piece_type = ((array[i]-1) % 6) + 1
                piece_color = array[i] > 6
                piece = chess.Piece(piece_type, piece_color)

                square = chess.SQUARE_NAMES.index(sn)
                board.set_piece_at(square, piece)

        turn = array[-1]
        board.turn = turn

        return board.fen()

    def from_fen_to_array(self, fen):
        board = chess.Board(fen)
        array = []

        for sn in chess.SQUARE_NAMES:
            square = chess.SQUARE_NAMES.index(sn)
            piece = board.piece_at(square)
            if piece is not None:
                array.append(piece.piece_type + piece.color * 6)
            else:
                array.append(0)

        array.append(board.turn)

        return np.array(array)

    def writable_state(self, s):
        return self.from_array_to_fen(s)