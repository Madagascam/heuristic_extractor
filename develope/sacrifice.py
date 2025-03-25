import chess
import chess.pgn


def piece_sacrifice(board, previous_board, move):
    # Получаем фигуру, которую взяли
    if board.is_capture(move):
        square = move.to_square
        captured_piece = previous_board.piece_at(square)
        if captured_piece is None:
            return False
        moving_piece = board.piece_at(square)
        # Определяем фигуру, которая сделала ход
        white_figures = ['Q', 'R', 'B', 'N']
        if str(moving_piece) not in white_figures:
            return False
        # Фигура, за которую жертвуют, должна быть защищена
        if not board.is_attacked_by(chess.BLACK, square):
            return False

        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 200
        }
        if piece_values[captured_piece.piece_type] < piece_values[moving_piece.piece_type]:
            return True
        return False

    return False

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
            if piece_sacrifice(board, previous_board, move):
                print(c)
