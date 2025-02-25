import argparse
import chess
import chess.pgn
import chess.engine
import os
from exceptions import UnexpectedEOF


class Annotator:
    def __init__(self, engine: chess.engine.SimpleEngine, depth: int = 16):
        self.engine = engine
        self.limit = chess.engine.Limit(depth=depth)

    def cook(self, game: chess.pgn.Game):
        """Добавляет аннотации в игре"""
        for node in game.mainline():
            info = self.engine.analyse(node.board(), limit=self.limit)
            score = info['score'].pov(chess.WHITE)

            eval_str = 'nan'
            if score.is_mate():
                eval_str = f'#{score.mate()}'
            else:
                eval_str = f'{score.score() / 100:.2f}'
            
            nag_comment = f'[%eval {eval_str}]'
            node.comment = nag_comment
        
        return game

    def add_annotations(self, input_pgn: str, output_pgn: str, skip: int = 0, quantity: int = 1, clean: bool = False) -> None:
        """
        Читает партию из input_pgn, делает анализ и
        сохраняет результат в output_pgn
        
        Параметры
        ---------
        input_pgn: str
            Путь к файлу, для которого нужно сделать аннотацию.
        output_pgn: str
            Путь, по которому следует сохранить партию с аннотациями.
        clean: bool = False
            Очищает файл output_pgn перед записью
        skip: int = 0
            Количество партий, которые нужно пропустить от начала партий
        quantity: int = 1
            Количество партий, которые нужно обработать

        Исключения
        ----------
        ValueError:
            1. Если не удалось создать файл или путь некорректный
            2. Если не удалось очистить файл
        UnexpectedEOF:
            1. Если количество партий в файле меньше, чем требуется пропустить (skip)
            2. Если количество партий в файле меньше, чем требуется пропустить и обработать (skip + quantity)
        """

        # Создание файла (если его нет)
        try:
            if not os.path.exists(output_pgn):
                # Если файл не существует, создаём его
                with open(output_pgn, 'x') as f:
                    pass
        except Exception as e:
            raise ValueError(f"Не удалось создать файл {output_pgn}. Ошибка: {e}")

        # Очистка файла, если указан флаг clean
        if clean:
            try:
                with open(output_pgn, 'w') as f:
                    f.write('')  # Явно очищаем содержимое файла
            except Exception as e:
                raise ValueError(f"Не удалось очистить файл {output_pgn}. Ошибка: {e}")

        with open(input_pgn, 'r') as pgn:
            # Скипаем часть игр
            for i in range(skip):
                game = chess.pgn.read_game(pgn)
                if game is None:
                    raise UnexpectedEOF('В файле оказалось меньше партий, чем нужно пропустить')
                print(f'Пропущено {i + 1}/{skip}', end='\r' if i < skip - 1 else '\n')
            
            # Обрабатываем игры
            with open(output_pgn, 'a') as output_file:
                for i in range(quantity):
                    print(f'Обработка... {i + 1}/{quantity}', end=' ')

                    # Аннотируем
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        raise UnexpectedEOF('В файле оказалось меньше партий, чем нужно пропустить и обработать')
                    game = self.cook(game)
            
                    # Пишем в файл
                    exporter = chess.pgn.FileExporter(output_file, headers=True, variations=True, comments=True)
                    game.accept(exporter)
                    print(f'Обработано {i + 1}/{quantity}', end='\r' if i < skip - 1 else '\n')


def main():
    # Указываем аргументы командной строки
    parser = argparse.ArgumentParser(
        prog='annotator.py',
        description='takes a pgn file and adds annotations to it'
    )
    parser.add_argument('--input', '-i', help='input pgn file', required=True)
    parser.add_argument('--output', '-o', help='output png file', required=True)
    parser.add_argument('--skip', help='how many games would skipped from the beginning of the file', default=0)
    parser.add_argument('--quantity', help='how many games would annotated', default=1)
    parser.add_argument('--clean', help='WARNING! Clears the output file before processing', action='store_true', default=False)
    parser.add_argument('-y', help='answer "yes" on all warnings', action='store_true', default=False)
    parser.add_argument('--stockfish', '-s', help='(engine settings) path to stockfish executable file', required=True)
    parser.add_argument('--depth', '-d', help='(engine settings) depth of analysis', default=16)
    parser.add_argument('--threads', '-t', help='(engine settings) threads to stockfish', default=1)

    # Парсим их
    args = parser.parse_args()

    # Проверка опасного аргумента --clean
    if args.clean and not args.y:
        confirmation = input("WARNING: The '--clean' option will clear the output file. Are you sure? (Y/n): ").strip()
        if confirmation != 'Y':
            print("Operation cancelled. Exiting.")
            return

    # Добавляем аннотации
    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as engine:
        engine.configure({'Threads': args.threads})
        annotator = Annotator(engine, args.depth)
        annotator.add_annotations(args.input, args.output, skip=int(args.skip), quantity=int(args.quantity), clean=args.clean)


if __name__ == '__main__':
    main()