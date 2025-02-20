import argparse
import chess
import chess.pgn
import chess.engine


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

    def add_annotations(self, input_pgn: str, output_pgn: str = None, overwrite: bool = False) -> None:
        """
        Читает партию из input_pgn, делает анализ и
        сохраняет результат в output_pgn
        
        Параметры
        ---------
        input_pgn: str
            Путь к файлу, для которого нужно сделать аннотацию.
        output_pgn: str = None
            Путь, по которому следует сохранить партию с аннотациями.
        overwrite: bool = False
            Если не указывать output_pgn, зато указать overwrite = True,
            файл с аннотациями перезапишет исходный файл.
        """

        # Если путь некорректный, будем считать, что output_pgn = None
        try:
            with open(output_pgn, 'w') as file:
                pass
        except:
            output_pgn = None

        # Если пользователь забыл разрешить перезапись, при этом путь некорректный, вызываем ошибку
        if overwrite == False and output_pgn is None:
            raise ValueError("Укажите или проверьте файл для сохранения, либо разрешите перезапись.")
        if overwrite == True:
            output_pgn = input_pgn

        # Читаем игру
        with open(input_pgn, 'r') as pgn:
            game = chess.pgn.read_game(pgn)
        
        # Аннотируем
        game = self.cook(game)

        # Пишем в файл
        exporter = chess.pgn.FileExporter(open(output_pgn, 'w'), headers=True, variations=True, comments=True)
        game.accept(exporter)


def main():
    # Указываем аргументы командной строки
    parser = argparse.ArgumentParser(
        prog='annotator.py',
        description='takes a pgn file and adds annotations to it'
    )
    parser.add_argument('--input', '-i', help='input pgn file', required=True)
    parser.add_argument('--output', '-o', help='output png file', default='None')
    parser.add_argument('--overwrite', help='overwrites the original file', action='store_true', default=False)
    parser.add_argument('--stockfish', '-s', help='(engine settings) path to stockfish executable file', required=True)
    parser.add_argument('--depth', '-d', help='(engine settings) depth of analysis', default=16)
    parser.add_argument('--threads', '-t', help='(engine settings) threads to stockfish', default=1)

    # Парсим их
    args = parser.parse_args()

    # Добавляем аннотации
    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as engine:
        engine.configure({'Threads': args.threads})
        annotator = Annotator(engine, args.depth)
        annotator.add_annotations(args.input, args.output, overwrite=args.overwrite)


if __name__ == '__main__':
    main()