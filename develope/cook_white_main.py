from typing import List, Optional
import chess
from chess import (
    square_rank,
    square_file,
    Board,
    SquareSet,
    Piece,
    PieceType,
    square_distance,
)
from chess import KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN
from chess import WHITE, BLACK
from chess.pgn import ChildNode
from develope.model import Puzzle, TagKind
import develope.util_prev as util_prev
import time


start = time.perf_counter()


def cook_white(puzzle: Puzzle) -> dict:
    highlights = dict()

    if mate_in(puzzle):
        if smothered_mate(puzzle):
            highlights["mate"] = "smotheredMate"
        elif back_rank_mate(puzzle):
            highlights["mate"] = "backRankMate"
        elif anastasia_mate(puzzle):
            highlights["mate"] = "anastasiaMate"
        elif hook_mate(puzzle):
            highlights["mate"] = "hookMate"
        elif arabian_mate(puzzle):
            highlights["mate"] = "arabianMate"
        else:
            found = boden_or_double_bishop_mate(puzzle)
            if found:
                highlights["mate"] = found
            elif dovetail_mate(puzzle):
                highlights["mate"] = "dovetailMate"

    moves = deflection(puzzle)
    if moves:
        highlights["deflection"] = moves

    moves = double_check(puzzle)
    if moves:
        highlights["doubleCheck"] = moves

    moves = fork(puzzle)
    if moves:
        highlights["fork"] = moves

    moves = trapped_piece(puzzle)
    if moves:
        highlights["trappedPiece"] = moves

    moves = skewer(puzzle)
    if moves:
        highlights["skewer"] = moves

    moves = self_interference(puzzle) + interference(puzzle)
    if moves:
        highlights["interference"] = moves

    moves = intermezzo(puzzle)
    if moves:
        highlights["intermezzo"] = moves

    moves = promotion(puzzle)
    if moves:
        highlights["promotion"] = moves

    moves = under_promotion(puzzle)
    if moves:
        highlights["underPromotion"] = moves

    moves = capturing_defender(puzzle)
    if moves:
        highlights["capturingDefender"] = moves

    return highlights



def double_check(puzzle: Puzzle) -> list:
    double_checks = []
    c = 0
    for node in puzzle.mainline[0::2]:
        c += 1
        if len(node.board().checkers()) > 1:
            double_checks.append(f"{c}W")
    return double_checks


def fork(puzzle: Puzzle) -> list:
    forks = []
    c = 0
    for node in puzzle.mainline[0::2][:-1]:
        c += 1
        if util_prev.moved_piece_type(node) is not KING:
            board = node.board()
            if util_prev.is_in_bad_spot(board, node.move.to_square) or util_prev.can_be_taken_by_same_piece(board, board.piece_at(node.move.to_square), node.move.to_square):
                continue
            if board.piece_at(node.move.to_square).piece_type == 1 and util_prev.may_be_winned_black_pawn(board, node.move.to_square):
                continue
            nb = 0
            for piece, square in util_prev.attacked_opponent_squares(
                board, node.move.to_square, puzzle.pov
            ):
                if piece.piece_type == PAWN:
                    continue
                if util_prev.king_values[piece.piece_type] > util_prev.king_values[
                    util_prev.moved_piece_type(node)
                ]: #or (
                #     util.is_hanging(board, piece, square)
                #     and square
                #     not in board.attackers(not puzzle.pov, node.move.to_square)
                # ):
                    nb += 1
            if nb > 1:
                forks.append(f"{c}W")
    return forks

def trapped_piece(puzzle: Puzzle) -> list:
    trapped_piece_moves = []
    c = 0
    for node in puzzle.mainline[0::2][1:]:
        c += 1
        square = node.move.to_square
        captured = node.parent.board().piece_at(square)
        if captured and captured.piece_type != PAWN:
            prev = node.parent
            assert isinstance(prev, ChildNode)
            if prev.move.to_square == square:
                square = prev.move.from_square
            if util_prev.is_trapped(prev.parent.board(), square):
                trapped_piece_moves.append(f"{c}W")
    return trapped_piece_moves


def discovered_check(puzzle: Puzzle) -> bool:
    discovered_check_moves = []
    c = 0
    for node in puzzle.mainline[0::2]:
        c += 1
        board = node.board()
        checkers = board.checkers()
        if checkers and not node.move.to_square in checkers:
            print(f"discovered_check: {c}")
            discovered_check_moves.append(f"{c}W")
    return discovered_check_moves

def deflection(puzzle: Puzzle) -> list:
    deflection_moves = []
    c = 0
    for node in puzzle.mainline[0::2][1:]:
        c += 1
        captured_piece = node.parent.board().piece_at(node.move.to_square)
        if captured_piece and captured_piece.piece_type == 1: # Пешка
            continue
        if captured_piece or node.move.promotion:
            capturing_piece = util_prev.moved_piece_type(node)
            if (
                captured_piece
                and util_prev.king_values[captured_piece.piece_type]
                > util_prev.king_values[capturing_piece]
            ):
                continue
            square = node.move.to_square
            prev_op_move = node.parent.move
            assert prev_op_move
            grandpa = node.parent.parent
            assert isinstance(grandpa, ChildNode)
            prev_player_move = grandpa.move
            prev_player_capture = grandpa.parent.board().piece_at(
                prev_player_move.to_square
            )
            if (
                (
                    not prev_player_capture
                    or util_prev.values[prev_player_capture.piece_type]
                    < util_prev.moved_piece_type(grandpa)
                )
                and square != prev_op_move.to_square
                and square != prev_player_move.to_square
                and (
                    prev_op_move.to_square == prev_player_move.to_square
                    or grandpa.board().is_check()
                )
                and (
                    square in grandpa.board().attacks(prev_op_move.from_square)
                    or node.move.promotion
                    and square_file(node.move.to_square)
                    == square_file(prev_op_move.from_square)
                    and node.move.from_square
                    in grandpa.board().attacks(prev_op_move.from_square)
                )
                and (not square in node.parent.board().attacks(prev_op_move.to_square))
            ):
                deflection_moves.append(f"{c}W")
    return deflection_moves

def skewer(puzzle: Puzzle) -> list:
    skewer_moves = []
    c = 0
    for node in puzzle.mainline[0::2][1:]:
        c += 1
        prev = node.parent
        assert isinstance(prev, ChildNode)
        capture = prev.board().piece_at(node.move.to_square)
        if (
            capture
            and util_prev.moved_piece_type(node) in util_prev.ray_piece_types
            and not node.board().is_checkmate()
        ):
            if capture.piece_type == chess.PAWN:
                continue
            between = SquareSet.between(node.move.from_square, node.move.to_square)
            op_move = prev.move
            assert op_move
            if (
                op_move.to_square == node.move.to_square
                or not op_move.from_square in between
            ):
                continue
            if util_prev.king_values[util_prev.moved_piece_type(prev)] > util_prev.king_values[
                capture.piece_type
            ] and util_prev.is_in_bad_spot(prev.board(), node.move.to_square):
                skewer_moves.append(f"{c}W")
    return skewer_moves



def self_interference(puzzle: Puzzle) -> list:
    # intereference by opponent piece
    self_interference_moves = []
    c = 0
    for node in puzzle.mainline[0::2][1:]:
        c += 1
        prev_board = node.parent.board()
        square = node.move.to_square
        capture = prev_board.piece_at(square)
        if capture and util_prev.is_hanging(prev_board, capture, square):
            grandpa = node.parent.parent
            assert grandpa
            init_board = grandpa.board()
            defenders = init_board.attackers(capture.color, square)
            defender = defenders.pop() if defenders else None
            defender_piece = init_board.piece_at(defender) if defender else None
            if (
                defender
                and defender_piece
                and defender_piece.piece_type in util_prev.ray_piece_types
            ):
                if node.parent.move and node.parent.move.to_square in SquareSet.between(
                    square, defender
                ):
                    self_interference_moves.append(f"{c}W")
    return self_interference_moves


def interference(puzzle: Puzzle) -> list:
    # intereference by player piece
    interference_moves = []
    c = 0
    for node in puzzle.mainline[0::2][1:]:
        c += 1
        prev_board = node.parent.board()
        square = node.move.to_square
        capture = prev_board.piece_at(square)
        assert node.parent.move
        if (
            capture
            and square != node.parent.move.to_square
            and util_prev.is_hanging(prev_board, capture, square)
        ):
            assert node.parent
            assert node.parent.parent
            assert node.parent.parent.parent
            init_board = node.parent.parent.parent.board()
            defenders = init_board.attackers(capture.color, square)
            defender = defenders.pop() if defenders else None
            defender_piece = init_board.piece_at(defender) if defender else None
            if (
                defender
                and defender_piece
                and defender_piece.piece_type in util_prev.ray_piece_types
            ):
                interfering = node.parent.parent
                if interfering.move and interfering.move.to_square in SquareSet.between(
                    square, defender
                ):
                    interference_moves.append(f"{c}W")
    return interference_moves


def intermezzo(puzzle: Puzzle) -> list:
    intermezzo_moves = []
    c = 0
    for node in puzzle.mainline[0::2][1:]:
        c += 1
        if util_prev.is_capture(node):
            capture_move = node.move
            capture_square = node.move.to_square
            op_node = node.parent
            assert isinstance(op_node, ChildNode)
            if op_node.board().piece_type_at(capture_square) == PAWN:
                continue
            prev_pov_node = node.parent.parent
            assert isinstance(prev_pov_node, ChildNode)
            if not op_node.move.from_square in prev_pov_node.board().attackers(
                not puzzle.pov, capture_square
            ):
                if prev_pov_node.move.to_square != capture_square:
                    prev_op_node = prev_pov_node.parent
                    assert isinstance(prev_op_node, ChildNode)
                    if (prev_op_node.move.to_square == capture_square
                        and util_prev.is_capture(prev_op_node)
                        and capture_move in prev_op_node.board().legal_moves):
                        intermezzo_moves.append(f"{c}W")

                    # return (
                    #     prev_op_node.move.to_square == capture_square
                    #     and util.is_capture(prev_op_node)
                    #     and capture_move in prev_op_node.board().legal_moves
                    # )
    return intermezzo_moves


def promotion(puzzle: Puzzle) -> list:
    promotions = []
    c = 0
    for node in puzzle.mainline[0::2]:
        c += 1
        if node.move.promotion:
            promotions.append(f"{c}W")
    return promotions


def under_promotion(puzzle: Puzzle) -> list:
    under_promotion = []
    c = 0
    for node in puzzle.mainline[0::2]:
        c += 1
        if node.board().is_checkmate():
            return True if node.move.promotion == KNIGHT else False
        elif node.move.promotion and node.move.promotion != QUEEN:
            under_promotion.append(f"{c}W")
    return under_promotion


def capturing_defender(puzzle: Puzzle) -> list:
    capturing_defender_moves = []
    c = 0
    for node in puzzle.mainline[0::2][1:]:
        c += 1
        board = node.board()
        capture = node.parent.board().piece_at(node.move.to_square)
        assert isinstance(node.parent, ChildNode)

        if board.is_checkmate() or (
            capture and util_prev.moved_piece_type(node) != KING
            and util_prev.values[capture.piece_type] <= util_prev.values[util_prev.moved_piece_type(node)]
            and util_prev.is_hanging(node.parent.board(), capture, node.move.to_square)
            and node.parent.move.to_square != node.move.to_square
        ):
            prev = node.parent.parent
            assert isinstance(prev, ChildNode)

            if (
                not prev.board().is_check()
                and prev.move.to_square != node.move.from_square
            ):
                assert prev.parent
                init_board = prev.parent.board()
                defender_square = prev.move.to_square
                defender = init_board.piece_at(defender_square)

                if (
                    defender
                    and defender_square in init_board.attackers(defender.color, node.move.to_square)
                    and not init_board.is_check()
                ):
                    # Проверка, защищаемая фигура не является пешкой
                    protected_piece = init_board.piece_at(node.move.to_square)
                    if protected_piece and protected_piece.piece_type == PAWN:
                        continue

                    capturing_defender_moves.append(f"{c}W")
    return capturing_defender_moves



def back_rank_mate(puzzle: Puzzle) -> bool:
    node = puzzle.game.end()
    board = node.board()
    king = board.king(not puzzle.pov)
    assert king is not None
    assert isinstance(node, ChildNode)
    back_rank = 7 if puzzle.pov else 0
    if board.is_checkmate() and square_rank(king) == back_rank:
        squares = SquareSet.from_square(king + (-8 if puzzle.pov else 8))
        if puzzle.pov:
            if chess.square_file(king) < 7:
                squares.add(king - 7)
            if chess.square_file(king) > 0:
                squares.add(king - 9)
        else:
            if chess.square_file(king) < 7:
                squares.add(king + 9)
            if chess.square_file(king) > 0:
                squares.add(king + 7)
        for square in squares:
            piece = board.piece_at(square)
            if (
                piece is None
                or piece.color == puzzle.pov
                or board.attackers(puzzle.pov, square)
            ):
                return False
        return any(square_rank(checker) == back_rank for checker in board.checkers())
    return False


def anastasia_mate(puzzle: Puzzle) -> bool:
    node = puzzle.game.end()
    board = node.board()
    king = board.king(not puzzle.pov)
    assert king is not None
    assert isinstance(node, ChildNode)
    if square_file(king) in [0, 7] and square_rank(king) not in [0, 7]:
        if square_file(node.move.to_square) == square_file(
            king
        ) and util_prev.moved_piece_type(node) in [QUEEN, ROOK]:
            if square_file(king) != 0:
                board.apply_transform(chess.flip_horizontal)
            king = board.king(not puzzle.pov)
            assert king is not None
            blocker = board.piece_at(king + 1)
            if blocker is not None and blocker.color != puzzle.pov:
                knight = board.piece_at(king + 3)
                if (
                    knight is not None
                    and knight.color == puzzle.pov
                    and knight.piece_type == KNIGHT
                ):
                    return True
    return False


def hook_mate(puzzle: Puzzle) -> bool:
    node = puzzle.game.end()
    board = node.board()
    king = board.king(not puzzle.pov)
    assert king is not None
    assert isinstance(node, ChildNode)
    if (
        util_prev.moved_piece_type(node) == ROOK
        and square_distance(node.move.to_square, king) == 1
    ):
        for rook_defender_square in board.attackers(puzzle.pov, node.move.to_square):
            defender = board.piece_at(rook_defender_square)
            if (
                defender
                and defender.piece_type == KNIGHT
                and square_distance(rook_defender_square, king) == 1
            ):
                for knight_defender_square in board.attackers(
                    puzzle.pov, rook_defender_square
                ):
                    pawn = board.piece_at(knight_defender_square)
                    if pawn and pawn.piece_type == PAWN:
                        return True
    return False


def arabian_mate(puzzle: Puzzle) -> bool:
    node = puzzle.game.end()
    board = node.board()
    king = board.king(not puzzle.pov)
    assert king is not None
    assert isinstance(node, ChildNode)
    if (
        square_file(king) in [0, 7]
        and square_rank(king) in [0, 7]
        and util_prev.moved_piece_type(node) == ROOK
        and square_distance(node.move.to_square, king) == 1
    ):
        for knight_square in board.attackers(puzzle.pov, node.move.to_square):
            knight = board.piece_at(knight_square)
            if (
                knight
                and knight.piece_type == KNIGHT
                and (
                    abs(square_rank(knight_square) - square_rank(king)) == 2
                    and abs(square_file(knight_square) - square_file(king)) == 2
                )
            ):
                return True
    return False


def boden_or_double_bishop_mate(puzzle: Puzzle) -> Optional[TagKind]:
    node = puzzle.game.end()
    board = node.board()
    king = board.king(not puzzle.pov)
    assert king is not None
    assert isinstance(node, ChildNode)
    bishop_squares = list(board.pieces(BISHOP, puzzle.pov))
    if len(bishop_squares) < 2:
        return None
    for square in [s for s in SquareSet(chess.BB_ALL) if square_distance(s, king) < 2]:
        if not all(
            [
                p.piece_type == BISHOP
                for p in util_prev.attacker_pieces(board, puzzle.pov, square)
            ]
        ):
            return None
    if (square_file(bishop_squares[0]) < square_file(king)) == (
        square_file(bishop_squares[1]) > square_file(king)
    ):
        return "bodenMate"
    else:
        return "doubleBishopMate"


def dovetail_mate(puzzle: Puzzle) -> bool:
    node = puzzle.game.end()
    board = node.board()
    king = board.king(not puzzle.pov)
    assert king is not None
    assert isinstance(node, ChildNode)
    if square_file(king) in [0, 7] or square_rank(king) in [0, 7]:
        return False
    queen_square = node.move.to_square
    if (
        util_prev.moved_piece_type(node) != QUEEN
        or square_file(queen_square) == square_file(king)
        or square_rank(queen_square) == square_rank(king)
        or square_distance(queen_square, king) > 1
    ):
        return False
    for square in [s for s in SquareSet(chess.BB_ALL) if square_distance(s, king) == 1]:
        if square == queen_square:
            continue
        attackers = list(board.attackers(puzzle.pov, square))
        if attackers == [queen_square]:
            if board.piece_at(square):
                return False
        elif attackers:
            return False
    return True


def smothered_mate(puzzle: Puzzle) -> bool:
    board = puzzle.game.end().board()
    king_square = board.king(not puzzle.pov)
    assert king_square is not None
    for checker_square in board.checkers():
        piece = board.piece_at(checker_square)
        assert piece
        if piece.piece_type == KNIGHT:
            for escape_square in [
                s for s in chess.SQUARES if square_distance(s, king_square) == 1
            ]:
                blocker = board.piece_at(escape_square)
                if not blocker or blocker.color == puzzle.pov:
                    return False
            return True
    return False

def mate_in(puzzle: Puzzle) -> Optional[TagKind]:
    return puzzle.game.end().board().is_checkmate()

import chess.pgn

#pgn_file_path = "C:/Users/Thinkpad/Downloads/lichess_broadcast_round-8_2024.10.22.pgn"
#pgn_file_path = "C:/Users/Thinkpad/Downloads/Tur_1.pgn (1).pgn"
#pgn_file_path = "C:/Users/Thinkpad/Downloads/Tur_9.MO_women.pgn"
#pgn_file_path = "C:/Users/Thinkpad/Downloads/lichess_broadcast_7th-autumn-esbjerg-open-championship-2024_b0v1T44h_2024.10.20.pgn"
#pgn_file_path = "C:/Users/Thinkpad/Downloads/lichess_pgn.pgn"
#pgn_file_path = "C:/Users/Thinkpad/Downloads/lichess_broadcast_round-2_hye-allan-loftgaard-mikkel-vinh_2024.10.18.pgn"
pgn_file_path = "C:/Users/Thinkpad/Downloads/lichess_swiss_2024.10.15_xwlmrKOM_grand-prix-mai-n4-oct-2024.pgn"

if __name__=="__main__":
    with open(pgn_file_path, encoding='utf-8') as pgn_file:
        d = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            # Создание объекта Puzzle
            puzzle = Puzzle(id="puzzle_1", game=game)
            puzzle.pov = BLACK

            print(f"game: {d // 8 + 1}.{d % 8 + 1}")
            result = cook_white(puzzle)
            print(result)
            d += 1
    finish = time.perf_counter()
    print('Время работы: ' + str(finish - start))
