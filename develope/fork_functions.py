import chess
from develope.safe import is_white_protected, is_black_protected,is_bishop_safe

def has_knight_fork(board, figure_positions):

    knight_positions = figure_positions['N']

    if not knight_positions:
        return False

    for knight_position in knight_positions:
        if not is_white_knight_safe(board, knight_position, figure_positions):
            continue  # Пропускаем коня, если он под атакой пешки, слона или коня

        attacked_squares = [move for move in board.attacks(knight_position)]
        # for move in board.attacks(knight_position):
        #     attacked_squares.append(move)

        attacked_pieces_number = 0
        for target_square in attacked_squares:

            attacked_piece = board.piece_at(target_square)

            if attacked_piece is not None and attacked_piece.color == chess.BLACK:

                if attacked_piece.piece_type == chess.PAWN:
                    continue
                if attacked_piece.piece_type == chess.BISHOP and is_black_protected(board, target_square, figure_positions):
                    continue
                attacked_pieces_number += 1

        if attacked_pieces_number > 1:
            return True

    return False

def has_bishop_fork(fen):
    board = chess.Board(fen)

    # Находим позиции всех белых слонов
    bishop_positions = [square for square in chess.SQUARES if board.piece_at(square) == chess.Piece(chess.BISHOP, chess.WHITE)]

    if not bishop_positions:
        return False

    for bishop_position in bishop_positions:

        if not is_bishop_safe(board, bishop_position):
            continue


        attacked_squares = []
        for move in board.attacks(bishop_position):
            attacked_squares.append(move)


        attacked_pieces = []
        for target_square in attacked_squares:
            attacked_piece = board.piece_at(target_square)
            if attacked_piece and attacked_piece.color == chess.BLACK:
                if attacked_piece.piece_type == chess.PAWN:
                    continue
                if attacked_piece.piece_type == chess.KNIGHT and is_black_protected(board, target_square):
                    continue
                if attacked_piece.piece_type == chess.QUEEN and not is_white_protected(board, bishop_position):
                    continue
                attacked_pieces.append(attacked_piece)

        if len(attacked_pieces) >= 2:
            return True
    return False

def has_rook_fork(board, figure_positions):

    rook_positions = figure_positions['R']

    if not rook_positions:
        return False

    for rook_position in rook_positions:

        if not is_white_rook_safe(board, rook_position, figure_positions):
            continue

        attacked_squares = [move for move in board.attacks(rook_position)]

        attacked_pieces_number = 0
        for target_square in attacked_squares:
            attacked_piece = board.piece_at(target_square)
            if attacked_piece and attacked_piece.color == chess.BLACK:
                if attacked_piece.piece_type == chess.PAWN:
                    continue
                if is_black_protected(board, target_square, figure_positions):
                    if attacked_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        continue

                attacked_pieces_number += 1

        if attacked_pieces_number > 1:
            return True
    return False

def has_queen_fork(board, figure_positions):

    queen_positions = figure_positions['Q']
    if not queen_positions:
        return False

    for queen_position in queen_positions:

        if not is_white_queen_safe(board, queen_position):
            continue

        attacked_squares = [move for move in board.attacks(queen_position)]

        attacked_pieces_number = 0
        for target_square in attacked_squares:
            attacked_piece = board.piece_at(target_square)
            if attacked_piece and attacked_piece.color == chess.BLACK:
                if attacked_piece.piece_type == chess.PAWN:
                    continue
                if is_black_protected(board, target_square, figure_positions):
                    if attacked_piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                        continue

                attacked_pieces_number += 1

        if attacked_pieces_number > 1:
            return True
    return False
