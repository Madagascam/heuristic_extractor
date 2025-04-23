import chess.pgn
import io
from .interfaces import PGNParserInterface, PGNParsingError

class PGNParser(PGNParserInterface):
    """Parses PGN string using python-chess."""

    def parse(self, pgn_string: str) -> str:
        """
        Parses PGN string into a numbered list of moves (SAN notation).
        Returns an empty string if parsing fails.
        """
        try:
            pgn = io.StringIO(pgn_string)
            game = chess.pgn.read_game(pgn)
            if game is None:
                raise PGNParsingError("Could not read game from PGN string.")

            moves_formatted = []
            board = game.board()
            half_move_index = 0
            node = game


            while not node.is_end():
                 next_node = node.variation(0)
                 if next_node is None:
                     break
                 move = next_node.move
                 if move:
                     try:
                         san = board.san(move)
                     except ValueError:
                         san = str(move)

                     move_num = board.fullmove_number
                     turn = 'w' if board.turn == chess.WHITE else 'b'
                     prefix = f"{move_num}." if turn == 'w' else f"{move_num}..."
                     moves_formatted.append(f"{half_move_index}. {prefix} {san}")
                     board.push(move)
                     half_move_index += 1
                 node = next_node

            if not moves_formatted:
                 return "No moves found in PGN."

            return "\n".join(moves_formatted)

        except Exception as e:
            raise PGNParsingError(f"Failed to parse PGN: {e}") from e