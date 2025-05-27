from highlighter import find_highlight
import os

def create_pgn(moves_str, output_file):
    """Преобразует последовательность ходов в pgn-файл
    
    Пример использования:
    moves_str = "e2e4 c7c6 f2f4 d7d5 e4d5 c6d5 g1f3 b8c6"
    output_file = "output.pgn"
    create_pgn(moves_str, output_file)
    """
    pgn_header = "[Event \"?\"]\n[Site \"?\"]\n[Date \"????.??.??\"]\n[Round \"?\"]\n[White \"?\"]\n[Black \"?\"]\n[Result \"*\"]\n\n"
    moves = moves_str.split()

    move_number = 1
    pgn_moves = []
    for i in range(0, len(moves), 2):
        if i + 1 < len(moves):
            pgn_moves.append(f"{move_number}. {moves[i]} {moves[i+1]}")
            move_number += 1
        else:
            pgn_moves.append(f"{move_number}. {moves[i]}")

    pgn_content = pgn_header + " ".join(pgn_moves) + " *"

    # Записываем в файл
    with open(output_file, 'w') as f:
        f.write(pgn_content)

moves = 'e2e4 c7c6 f2f4 d7d5 e4d5 c6d5 g1f3 b8c6 d2d3 g8f6 f1e2 d8c7 e1g1 e7e5 f4e5 c6e5 f3e5 c7e5 c1f4 e5b2 b1d2 f8c5 g1h1 e8g8 d2b3 c5b6 f4d6 f8d8 d6e7 d8d7 e7f6 g7f6 f1f3 d7c7 f3g3 g8h8 e2h5 c8f5 h5f7 c7f7 d1h5 f5e6 a1e1 f7e7 h5h6 b6f2 e1e6 f2g3 e6e7 b2b1 b3c1 b1c1 h6c1 g3d6 e7b7 a8e8 h2h3 d6c5 c1f4 e8e1 h1h2 c5g1 h2g3 e1e3 g3g4 h7h5 g4h5 e3e5 h5g6 e5g5 g6f6 g5g2 f4h6 h8g8 h6h7 g8f8 b7b8'
create_pgn(moves, 'temp.pgn')

# Получаем интересный момент
interesting_moment = find_highlight("temp.pgn")
start, end = interesting_moment['start'], interesting_moment['end']
os.remove('temp.pgn')