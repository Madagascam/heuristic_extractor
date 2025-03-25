import chess
import chess.pgn
import time
from develope.safe import possible_block_pin_square, is_white_bishop_safe, get_all_positions

start = time.perf_counter()

def white_bishop_pin(board, figure_positions):
    black_rook_positions = figure_positions['r']
    white_bishop_positions = figure_positions['B']
    for rook_square in black_rook_positions:
        if board.is_pinned(chess.BLACK, rook_square):
            # direction = board.pin(chess.BLACK, rook_square)
            for bishop_square in white_bishop_positions:
                # if rook_square in board.attacks(bishop_square) and bishop_square in direction:
                # Есть сомнения в необходимости проверки направления, т.к. если ладью связывает ферзь и на нее нападает слон, то это тоже связка
                if rook_square in board.attacks(bishop_square):
                    if is_white_bishop_safe(board, bishop_square, figure_positions):
                        if abs(rook_square - bishop_square) % 9 == 0:
                            start = min(rook_square, bishop_square) + 9
                            end = max(rook_square, bishop_square)
                            for square in range(start, end, 9):
                                if possible_block_pin_square(board, square, figure_positions):
                                    return False

                            return True
                        if abs(rook_square - bishop_square) % 7 == 0:
                            start = min(rook_square, bishop_square) + 7
                            end = max(rook_square, bishop_square)
                            for square in range(start, end, 7):
                                if possible_block_pin_square(board, square, figure_positions):
                                    return False
                            return True
    return False



pgn_file_path = "C:/Users/Thinkpad/Downloads/lichess_pgn_bishop_pin.pgn"

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
            figure_positions = get_all_positions(board)
            if white_bishop_pin(board, figure_positions):
                # print("Слон связывает ладью на позиции:\n")
                #print(board)
                print(c)

finish = time.perf_counter()
print('Время работы: ' + str(finish - start))
