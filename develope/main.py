import chess.pgn
from chess import WHITE, BLACK
from develope.cook_black_main import cook_black
from develope.cook_white_main import cook_white
from develope.model import Puzzle
import time

start = time.perf_counter()

#pgn_file_path = "C:/Users/Thinkpad/Downloads/lichess_broadcast_7th-autumn-esbjerg-open-championship-2024_b0v1T44h_2024.10.20.pgn"
#pgn_file_path = "C:/Users/Thinkpad/Downloads/lichess_pgn.pgn"
#pgn_file_path = "C:/Users/Thinkpad/Downloads/lichess_swiss_2024.10.24_qqrfjd21_grand-prix-mai-n6-oct-2024.pgn"
pgn_file_path = "C:/Users/Thinkpad/Downloads/lichess_broadcast_bharatiya-naveena-kreeda-utsav-chess-festival-2024_FlhXUXrp_2024.10.25.pgn"
#pgn_file_path = "C:/Users/Thinkpad/Downloads/lichess_broadcast_round-8_2024.10.22_(2).pgn"


def find_highlights_in_one_game(pgn_file_path):
    with open(pgn_file_path, encoding='utf-8') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            puzzle_w = Puzzle(id="puzzle_w", game=game)
            puzzle_w.pov = WHITE
            result_w = cook_white(puzzle_w)

            puzzle_b = Puzzle(id="puzzle_b", game=game)
            puzzle_b.pov = BLACK
            result_b = cook_black(puzzle_b)

            result = {}

            if "mate" in result_w.keys():
                result["mate"] = result_w["mate"]
                del result_w["mate"]
            elif "mate" in result_b.keys():
                result["mate"] = result_b["mate"]
                del result_b["mate"]

            all_keys = set(result_w.keys()).union(set(result_b.keys()))

            for key in all_keys:
                # Проверяем, есть ли ключ в обоих словарях
                values = []
                if key in result_w:
                    values.extend(result_w[key])
                if key in result_b:
                    values.extend(result_b[key])
                result[key] = values
    return result

def find_highlights_in_many_games(pgn_file_path):
    with open(pgn_file_path, encoding='utf-8') as pgn_file:
        d = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            print(f"game {d // 15 + 1}.{d % 15 + 1}")
            #print(f"game: {d}")
            # Создание объекта Puzzle
            puzzle_w = Puzzle(id="puzzle_w", game=game)
            puzzle_w.pov = WHITE
            result_w = cook_white(puzzle_w)

            puzzle_b = Puzzle(id="puzzle_b", game=game)
            puzzle_b.pov = BLACK
            result_b = cook_black(puzzle_b)

            result = {}

            if "mate" in result_w.keys():
                result["mate"] = result_w["mate"]
                del result_w["mate"]
            elif "mate" in result_b.keys():
                result["mate"] = result_b["mate"]
                del result_b["mate"]

            all_keys = set(result_w.keys()).union(set(result_b.keys()))

            for key in all_keys:
                # Проверяем, есть ли ключ в обоих словарях
                values = []
                if key in result_w:
                    values.extend(result_w[key])
                if key in result_b:
                    values.extend(result_b[key])
                result[key] = values
            print(result)
            d += 1
#print(find_highlights_in_one_game(pgn_file_path))
find_highlights_in_many_games(pgn_file_path)
finish = time.perf_counter()
print('Время работы: ' + str(finish - start))
