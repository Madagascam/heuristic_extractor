"""
Модуль генерирует разметку из партий в формате pgn.
Последовательность ходов после генерации может
обрывать партию, не завершая её матом.
"""

import argparse
import copy
import chess
import chess.engine
import chess.pgn
import csv
import random
import logging
from typing import Optional, List, Union, Tuple
from util import *

mate_soon = chess.engine.Mate(15)
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def material_count(board: chess.Board, side: chess.Color) -> int:
    values = { chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9 }
    return sum(len(board.pieces(piece_type, side)) * value for piece_type, value in values.items())


def material_diff(board: chess.Board, side: chess.Color) -> int:
    return material_count(board, side) - material_count(board, not side)


class Generator:
    def __init__(self, engine: chess.engine.SimpleEngine):
        """
        Несмотря на то, что генератор принимает на вход движок,
        партии всё равно должны быть предварительно размечены с
        помощью модуля `annotator.py`
        """
        self.engine = engine
    
    def is_valid_mate_in_one(self, pair: NextMovePair) -> bool:
        if pair.best.score != chess.engine.Mate(1):
            return False
        non_mate_win_threshold = 0.6
        if not pair.second or win_chances(pair.second.score) <= non_mate_win_threshold:
            return True
        if pair.second.score == chess.engine.Mate(1):
            # if there's more than one mate in one, gotta look if the best non-mating move is bad enough
            logger.debug('Looking for best non-mating move...')
            mates = count_mates(copy.deepcopy(pair.node.board()))
            info = self.engine.analyse(pair.node.board(), multipv = mates + 1, limit = pair_limit)
            scores =  [pv["score"].pov(pair.winner) for pv in info]
            # the first non-matein1 move is the last element
            if scores[-1] < chess.engine.Mate(1) and win_chances(scores[-1]) > non_mate_win_threshold:
                return False
            return True
        return False

    # is pair.best the only continuation?
    def is_valid_attack(self, pair: NextMovePair) -> bool:
        return (
            pair.second is None or
            self.is_valid_mate_in_one(pair) or
            win_chances(pair.best.score) > win_chances(pair.second.score) + 0.7
        )

    def get_next_pair(self, node: chess.pgn.ChildNode, winner: chess.Color) -> Optional[NextMovePair]:
        pair = get_next_move_pair(self.engine, node, winner, pair_limit)
        if node.board().turn == winner and not self.is_valid_attack(pair):
            logger.debug("No valid attack {}".format(pair))
            return None
        return pair

    def get_next_move(self, node: chess.pgn.ChildNode, limit: chess.engine.Limit) -> Optional[chess.Move]:
        result = self.engine.play(node.board(), limit = limit)
        return result.move if result else None
    
    def cook_advantage(self, node: chess.pgn.GameNode, winner: chess.Color) -> Optional[List[NextMovePair]]:
        board = node.board()

        if board.is_repetition(2):
            logger.debug("Found repetition, canceling")
            return None

        pair = self.get_next_pair(node, winner)
        if not pair:
            return []
        if pair.best.score < chess.engine.Cp(200):
            logger.debug("Not winning enough, aborting")
            return None

        follow_up = self.cook_advantage(node.add_main_variation(pair.best.move), winner)

        if follow_up is None:
            return None

        return [pair] + follow_up

    def cook_mate(self, node: chess.pgn.Game, winner: chess.Color) -> Optional[List[NextMovePair]]:
        board = node.board()

        if board.is_game_over():
            return []

        if board.turn == winner:
            pair = self.get_next_pair(node, winner)
            if not pair:
                return None
            if pair.best.score < mate_soon:
                logger.debug("Best move is not a mate, we're probably not searching deep enough")
                return None
            move = pair.best.move
        else:
            next = self.get_next_move(node, mate_defense_limit)
            if not next:
                return None
            move = next

        follow_up = self.cook_mate(node.add_main_variation(move), winner)

        if follow_up is None:
            return None

        return [move] + follow_up

    def generate_interesting(self, input_pgn: str, output_file: str, skip: int = 0, quantity: int = 1):
        with open(input_pgn, 'r') as pgn:
            # Пропускаем партии
            for i in range(skip):
                game = chess.pgn.read_game(pgn)
            logger.info(f'Пропущено {skip}')
            
            # Определяем заголовки для CSV-файла
            fieldnames = ['id', 'white_elo', 'black_elo', 'moves', 'marks']
            
            # Открываем выходной CSV-файл для записи
            with open(output_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Записываем заголовки только если файл пустой
                if csvfile.tell() == 0:
                    writer.writeheader()
                
                # Обрабатываем партии
                for i in range(quantity):
                    logger.info(f'Обработано {i}/{quantity}, обрабатывается {i + 1}...')
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        logger.warning("Достигнут конец файла до обработки всех игр.")
                        break
                    
                    result = self.cook_interesting(game, get_tier(game))
                    if result is not None:
                        game_id = game.headers['GameId'] if 'GameId' in game.headers else f'unknown{random.randint(1, 10**9)}'

                        # Ходы и разметка
                        moves: List[chess.Move] = result[0]
                        marks: Tuple[int, int] = result[1]

                        # Рейтинг игроков
                        white_elo = None
                        black_elo = None
                        if 'WhiteElo' in game.headers:
                            try:
                                white_elo = int(game.headers['WhiteElo'])
                            except ValueError:
                                pass
                        
                        if 'BlackElo' in game.headers:
                            try:
                                black_elo = int(game.headers['BlackElo'])
                            except ValueError:
                                pass
                        
                        # Формируем строку для записи в CSV
                        marked_game = {
                            'id': game_id,
                            'white_elo': white_elo,
                            'black_elo': black_elo,
                            'moves': " ".join([move.uci() for move in moves]),  # Ходы как строка через пробел
                            'marks': f"{marks[0]},{marks[1]}"  # Разметка как строка через запятую
                        }
                        
                        # Записываем строку в CSV
                        writer.writerow(marked_game)

    def cook_interesting(self, game: chess.pgn.Game, tier: int) -> Union[Tuple[List[chess.Move], Tuple[int, int]], None]:
        moves = []
        previous_score = chess.engine.Cp(20)
        for node in game.mainline():
            # Сохраняем ходы
            moves.append(node.move)

            # Анализируем ход на интересность
            current_score = node.eval()
            result = self.cook_interesting_position(
                node,
                previous_score,
                current_score,
                tier
            )

            # Если можно построить интересный момент
            if result is not None:
                # То возвращаем видоизменённую партию
                return moves + result, (len(moves), len(moves) + len(result) - 1)
            else:
                # Иначе просто продолжаем
                previous_score = current_score.pov(not node.board().turn)
        
        return None
    
    def cook_interesting_position(
            self,
            node: chess.pgn.GameNode,
            previous_score: chess.engine.Score,
            current_score: chess.engine.PovScore,
            tier: int
    ):
        board = node.board()
        winner = board.turn
        score = current_score.pov(winner)

        # Если возможен только один ход, ничего интересного нет
        if board.legal_moves.count() <= 1:
            return None

        # Если позиция слишком лёгкая (большое преимущество или мат очень скоро)
        if previous_score > chess.engine.Cp(300) and score < mate_soon:
            logger.debug(f"Позиция {'белых' if winner == chess.WHITE else 'черных'}#{node.ply()} "
                  f"слишком выигрышная: previous_score={previous_score}, score={score}")
            return None

        # Если сторона имеет большое материальное преимущество
        eps = 5 # Надо подбирать, я наобум поставил
        if abs(material_diff(board, winner)) > eps:
            logger.debug(f"{'Белые' if winner == chess.WHITE else 'Черные'}#{node.ply()} имеют "
                  f"большое материальное преимущество: white={material_count(board, chess.WHITE)}, "
                  f"black={material_count(board, chess.BLACK)}")
            return None

        # Скип, если мат в один ход
        if score >= chess.engine.Mate(1) and tier > 2:
            logger.debug(f"{node.ply()} Мат в один ход: tier={tier}, score={score}")
            return None
        
        # Если намечается мат, пробуем его найти
        if score >= mate_soon:
            logger.debug(f"{node.ply()} Ищем мат...")

            # Ищем решение
            mate_solution = self.cook_mate(copy.deepcopy(node), winner)

            # Если решения нет ИЛИ для первого тира это обычный мат в два хода
            if mate_solution is None or (tier > 2 and len(mate_solution) <= 3):
                return None
            
            return [pair.best.move for pair in mate_solution]
    
        # Наконец, пробуем выиграть преимущество
        if score >= chess.engine.Cp(200) and win_chances(score) > win_chances(previous_score) + 0.6:
            if score < chess.engine.Cp(400) and material_diff(board, winner) > -1:
                logger.debug("Not clearly winning and not from being down in material, aborting")
                return None
            logger.debug('Пробуем получить преимущество...')
            adv_solution = self.cook_advantage(copy.deepcopy(node), winner)
            logger.debug(adv_solution)
            # Если преимущество получить невозможно
            if adv_solution is None or len(adv_solution) == 0:
                return None
            
            while len(adv_solution) % 2 == 0 or not adv_solution[-1].second:
                if not adv_solution[-1].second:
                    logger.debug("Remove final only-move")
                adv_solution = adv_solution[:-1]
            
            # if tier > 2 and len(adv_solution) <= 3:
            #     return None

            moves = [pair.best.move for pair in adv_solution]
            if len(moves) < 3:
                return None
            return moves

        return None
        

def main():
    # Указываем аргументы командной строки
    parser = argparse.ArgumentParser(
        prog='generator.py',
        description='takes a pgn file, find and add interesting moment to it'
    )
    parser.add_argument('--input', '-i', help='input pgn file', required=True)
    parser.add_argument('--output', '-o', help='output csv-file with marked game', required=True)
    parser.add_argument('--skip', help='how many games would skipped from the beginning of the file', default=0)
    parser.add_argument('--quantity', help='how many games would annotated', default=1)
    parser.add_argument('--stockfish', '-s', help='(engine settings) path to stockfish executable file', required=True)
    parser.add_argument('--threads', '-t', help='(engine settings) threads to stockfish', default=1)
    parser.add_argument("--verbose", "-v", help="increase verbosity", action="count")
    
    # Парсим аргументы
    args = parser.parse_args()
    
    # Настройка уровня логирования
    if args.verbose == 2:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # Генерируем партию с интересным моментом
    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as engine:
        engine.configure({"Threads": int(args.threads)})
        generator = Generator(engine)
        generator.generate_interesting(
            input_file=args.input,
            output_file=args.output,
            skip=int(args.skip),
            quantity=int(args.quantity)
        )


if __name__ == '__main__':
    main()