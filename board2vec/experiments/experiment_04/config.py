"""Архитектура модели"""

# Размер скрытого слоя
hidden_dim = 512

# Размер готового эмбеддинга
output_dim = 128

"""Гиперпараметры"""

# Начальный шаг обучения
learning_rate = 1

# (lr_scheduler) Один раз в step_size эпох шаг обучения будет уменьшаться в gamma раз
step_size = 1
gamma = 0.95

# Количество эпох обучения
num_epochs = 50

"""Пути для сохранения файлов"""

# Директория для сохранения весов
weight_dir = 'D:/Program Files/JupyterLabWorkspace/heuristic_extractor/board2vec/experiments/experiment_04/weights/'