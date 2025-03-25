from heuristic_functions import find_moments_without_stockfish, stockfish_moments
from util import merge_intervals, transform_format

pgn_file_path = "lichess_pgn.pgn"
engine_path = "C:/Users/Thinkpad/Desktop/Гоша/Friflex/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
heuristics_without_stockfish = find_moments_without_stockfish(pgn_file_path)
stockfish_moves = stockfish_moments(pgn_file_path, engine_path)
heuristics = heuristics_without_stockfish + stockfish_moves
heuristics = merge_intervals(heuristics)
print(heuristics)
print(transform_format(heuristics))
