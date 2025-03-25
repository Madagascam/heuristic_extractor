import chess
import chess.pgn

def nice_mate(board, move):
    if board.is_checkmate():
        print(move.to_square)
        if board.piece_at(move.to_square).piece_type in [chess.KNIGHT, chess.BISHOP, chess.KING]:
            return True
        return False
    return False

import chess
import chess.pgn
pgn_file_path = "C:/Users/Thinkpad/Downloads/lichess_pgn_knight_mate.pgn"

with open(pgn_file_path) as pgn_file:
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        c = 0

        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            c += 1
        last_move = list(game.mainline_moves())[-1]
        if nice_mate(board, last_move):
            print("Красивый мат на позиции:")
            print(board)
