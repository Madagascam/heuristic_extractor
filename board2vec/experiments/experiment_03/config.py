"""Архитектура модели"""

# Размер скрытого слоя
hidden_dim = 128

# Размер готового эмбеддинга
output_dim = 64

"""Гиперпараметры"""

# Начальный шаг обучения
learning_rate = 0.001

# (lr_scheduler) Один раз в step_size эпох шаг обучения будет уменьшаться в gamma раз
step_size = 30
gamma = 0.95

# Количество эпох обучения
num_epochs = 10000

"""Настройки даталоадера"""

# Ширина окна контекста (сколько партий влево и вправо нужно уметь предсказывать)
WINDOW_SIZE = 8

# Количество итераций в эпохе (количество игр, участвующих в эпохе)
GAME_COUNT = 50

# Размер батча (количество пар, получаемых из игры)
PAIR_CNT = 50

# Размер подмножества игр, которое участвует в генерации негативных примеров
SUBSET_SIZE = 1000

# Количество негативных примеров
NEGATIVES_COUNT = 5

"""Пути для сохранения файлов"""

# Путь к размеченным данным (https://disk.yandex.ru/d/z1rfxW7UZmnAhw)
data_path = 'C:/Users/matvey/Documents/chess_data/games.csv'

# Путь к исполняемому движку stockfish
stockfish_path = 'C:/Users/matvey/Documents/chess_data/stockfish/stockfish.exe'

# Директория для сохранения весов
weight_dir = 'C:/Users/matvey/workspace/heuristic_extractor/board2vec/experiments/experiment_03/weights/'