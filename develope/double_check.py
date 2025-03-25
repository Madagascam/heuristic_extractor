import chess
import chess.pgn


def is_discovered_check(board, move):
    if not board.is_check():
        return False

    last_move = move.to_square
    black_king_square = board.king(chess.BLACK)

    if black_king_square not in board.attacks(last_move):
        return True

    return False

def is_double_check(board):

    if not board.is_check():
        return False
    black_king_square = board.king(chess.BLACK)

    white_attackers = board.attackers(chess.WHITE, black_king_square)

    return len(white_attackers) > 1


#pgn_file_path = "C:/Users/Thinkpad/Downloads/lichess_pgn_2021.09.25_sres20let_vs_Zoso71.NkgyzoS4.pgn"
pgn_file_path = "C:/Users/Thinkpad/Downloads/lichess_pgn.pgn"

with open(pgn_file_path) as pgn_file:
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        c = 0
        board = game.board()
        for move in game.mainline_moves():
            previous_board = board.copy()
            board.push(move)
            c += 1
            if is_discovered_check(board, move):
                print(f"Вскрытый шах после хода {move}")
                print(c)

            if is_double_check(board):
                print(f"Двойной шах после хода {move}")
                print(c)
