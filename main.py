if __name__ == '__main__':
    from board2vec import board2vec, pgn_to_boards

    path = 'D:/Program Files/JupyterLabWorkspace/chess_data/example_games.pgn'
    list_of_boards = pgn_to_boards(path)

    for boards in list_of_boards:
        print(board2vec(boards))