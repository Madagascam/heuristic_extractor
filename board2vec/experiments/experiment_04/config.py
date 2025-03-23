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
num_epochs = 5

"""Пути для сохранения файлов"""

# Путь к размеченным данным
data_path = 'C:/Users/matvey/Documents/chess_data/small_labeled.h5'

# Директория для сохранения весов
weight_dir = 'C:/Users/matvey/workspace/heuristic_extractor/board2vec/experiments/experiment_04/weights/'