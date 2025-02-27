"""
Модуль для обработки уже готовых пазлов с
https://database.lichess.org/#puzzles.
"""

import argparse
import chess
import chess.pgn
import csv
import requests
import random
import json
from io import StringIO
from typing import List, Union


class Loader:
    def __init__(self, path_to_puzzles: str):
        self.__puzzle_file = open(path_to_puzzles, 'r', newline='')
        self.reader = csv.DictReader(self.__puzzle_file)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.__puzzle_file.close()
        if exc_type is not None:
            print(f"Было возбуждено исключение: {exc_type.__name__}: {exc_value}")
        return False
    
    def close(self):
        if self.__puzzle_file and not self.__puzzle_file.closed:
            self.__puzzle_file.close()

    def get_game(self, game_url: str) -> Union[chess.pgn.Game, None]:
        # Получаем ID игры. Данные в базе хранятся в одном формате, поэтому хардкод:
        try:
            game_id = game_url.split('/')[3]
        except Exception as e:
            raise ValueError('Некорректный адрес')
        
        # Эндпоинт для получения партии по её ID
        api_url = f'https://lichess.org/game/export/{game_id}'

        # Пробуем получить игру
        try:
            response = requests.get(api_url)
        except requests.RequestException as e:
            return None
        
        if response.status_code != 200:
            return None

        # Преобразуем в объект игры
        try:
            game = chess.pgn.read_game(StringIO(response.text))
            if not game:
                return None
            return game
        except Exception as e:
            return None
    
    def find_position(self, game: chess.pgn.Game, fen: str) -> List[chess.Move]:
        need_board = chess.Board(fen=fen)
        cur_board = chess.Board()
        moves = []
        for move in game.mainline_moves():
            moves.append(move)
            cur_board.push(move)
            if cur_board == need_board:
                return moves
        raise LookupError('Не найдено указанной позиции в игре')

    def make_marked(self, row: dict):
        try:
            game_url = row['GameUrl']
            game = self.get_game(game_url)
        except (KeyError, ValueError):
            game = None
        
        if game is None:
            return None
        
        moves_before = self.find_position(game, row['FEN'])
        moves_after = [chess.Move.from_uci(str_move) for str_move in row['Moves'].split()]

        game_id = game.headers['GameId'] if 'GameId' in game.headers else f'unknown{random.randint(1, 10**9)}'

        # Рейтинг игроков
        white_elo = 1500
        black_elo = 1500
        if 'WhiteElo' in game.headers:
            white_elo = int(game.headers['WhiteElo'])
        if 'BlackElo' in game.headers:
            black_elo = int(game.headers['BlackElo'])

        marked_game = {
            'id': game_id,
            'white_elo': white_elo,
            'black_elo': black_elo,
            'moves': [move.uci() for move in moves_before + moves_after],
            'marks': [len(moves_before), len(moves_before) + len(moves_after)]
        }

        return marked_game


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='loader.py',
        description='takes csv file with lichess-puzzles and creates labeled data in json'
    )
    parser.add_argument('--input', '-i', help='input csv file', required=True)
    parser.add_argument('--output', '-o', help='output json-file with labeled games', required=True)
    parser.add_argument('--skip', help='how many puzzles would skipped from the beginning of the file', default=0)
    parser.add_argument('--quantity', help='how many puzzles would processed', default=1)

    args = parser.parse_args()

    marked_games = []
    with Loader(args.input) as loader:
        # Скипаем
        cnt = int(args.skip)
        if cnt > 0:
            for row in loader.reader:
                cnt -= 1
                if cnt == 0:
                    break
        
        # Обрабатываем
        cnt = int(args.quantity)
        if cnt > 0:
            for row in loader.reader:
                marked_game = loader.make_marked(row)
                if marked_game:
                    marked_games.append(marked_game)
                    cnt -= 1
                if cnt == 0:
                    break

    
    with open(args.output, 'a') as file:
        json.dump(marked_games, file)
