# Импорты библиотек
import chess.pgn  # Для работы с PGN файлами шахматных партий
import numpy as np  # Для работы с массивами и математическими операциями
import torch  # Для работы с PyTorch моделями
from scipy.signal import find_peaks, peak_widths  # Для поиска пиков и их характеристик

# Импорты пользовательских модулей
from .encoder import MatrixEncoder  # Кодировщик досок
from .board2vec import Board2Vec  # Модель для преобразования досок в векторы
from .transformer import BinaryClassifierTransformer

def pgn_to_tensors(pgn_path, encoder):
    """
    Преобразует единственную партию из PGN-файла в список тензоров (закодированных досок)
    
    Аргументы:
        pgn_path: str - путь к PGN-файлу с одной партией
        encoder - кодировщик досок (должен иметь метод encode)
        
    Возвращает:
        list[np.ndarray] - список закодированных досок для каждого хода партии
    """
    with open(pgn_path, encoding='utf-8') as pgn_file:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            raise ValueError("PGN файл не содержит партий или пуст")
            
        # Проверяем, что в файле только одна партия
        if chess.pgn.read_game(pgn_file) is not None:
            raise ValueError("PGN файл содержит более одной партии")
            
        board = game.board()
        moves = list(game.mainline_moves())
        encoded_boards = []
        
        # Кодируем начальную позицию
        encoded_boards.append(encoder.encode(board))
        
        # Кодируем позиции после каждого хода
        for move in moves:
            board.push(move)
            encoded_boards.append(encoder.encode(board))
            
        return encoded_boards


def inference(y_pred):
    """
    Выполняет инференс на основе предсказаний модели.
    Обнаруживает пики и формирует маску активности.

    Аргументы:
        y_pred: np.ndarray - массив предсказаний модели
        
    Возвращает:
        np.ndarray - бинарная маска активности
    """
    # Применяем сглаживание
    kernel = np.ones(5) / 5
    smoothed = np.convolve(y_pred, kernel, mode='same')
    
    threshold = np.percentile(smoothed, 70)
    peaks, _ = find_peaks(smoothed, height=threshold, distance=10)
    
    # Если пики не найдены, используем простой порог
    if len(peaks) == 0:
        y_result = (smoothed > threshold).astype(int)
        return y_result
    
    rel_heights = np.minimum(1, smoothed[peaks] * 8)
    widths_list, left_ips_list, right_ips_list = [], [], []
    
    for i, peak in enumerate(peaks):
        w, _, l, r = peak_widths(smoothed, [peak], rel_height=rel_heights[i])
        widths_list.append(w[0])
        left_ips_list.append(l[0])
        right_ips_list.append(r[0])
    
    widths = np.array(widths_list)
    left_ips = np.array(left_ips_list)
    right_ips = np.array(right_ips_list)
    
    valid_mask = (widths > 4) & (widths < 50)
    left_ips = left_ips[valid_mask]
    right_ips = right_ips[valid_mask]
    
    y_result = np.zeros_like(y_pred)
    
    left_bounds = np.floor(left_ips).astype(int)
    right_bounds = np.ceil(right_ips).astype(int) + 1
    
    for start, end in zip(left_bounds, right_bounds):
        y_result[start:end] = 1
    
    # Дополнительная проверка: если после всех фильтров остались нули
    # возвращаем результат по порогу
    if np.sum(y_result) == 0:
        threshold = np.percentile(smoothed, 80)
        y_result = (smoothed > threshold).astype(int)
    
    return y_result


def find_longest_segment_of_ones(arr):
    # Находим индексы, где происходят переходы между 0 и 1
    changes = np.diff(np.concatenate(([0], arr, [0])))
    
    # Начала отрезков (индексы, где 0 переходит в 1)
    starts = np.where(changes == 1)[0]
    # Концы отрезков (индексы, где 1 переходит в 0)
    ends = np.where(changes == -1)[0]
    
    # Если нет отрезков из единиц, возвращаем (0, 0)
    if len(starts) == 0:
        return (0, 0)
    
    # Находим самый длинный отрезок
    lengths = ends - starts
    longest_idx = np.argmax(lengths)
    
    return (starts[longest_idx], ends[longest_idx])


# Определение устройства для вычислений (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка предобученной модели Board2Vec
board2vec = Board2Vec(128, 64)
board2vec.load_state_dict(torch.load('C:/Users/matvey/workspace/FriflexChess/highlighter/highlighter/checkpoints/board2vec_epoch1.pt', map_location=device))
board2vec.eval()

# Загрузка предобученной модели BinaryClassifierTransformer
model = BinaryClassifierTransformer(64).to(device)
model.load_state_dict(torch.load('C:/Users/matvey/workspace/FriflexChess/highlighter/highlighter/checkpoints/transformer_epoch50.pt', map_location=device))


def find_highlight(pgn_path):
    """
    Находит ключевые моменты в шахматной партии на основе анализа PGN-файла.

    Аргументы:
        pgn_path: str - путь к PGN-файлу с партией
        
    Возвращает:
        tuple[int, int] - временные границы ключевого момента (в полуходах)
    """
    tensors = pgn_to_tensors(pgn_path, MatrixEncoder())
    embeds = np.array([board2vec(torch.tensor(x).unsqueeze(0)).detach().numpy() for x in tensors]).reshape((len(tensors), 64))
    
    # Паддинг до фиксированной длины
    if len(tensors) < 200:
        pad = torch.zeros(200 - len(tensors), embeds.shape[1])
        embeds = torch.cat([torch.FloatTensor(embeds), pad])
    else:
        embeds = torch.FloatTensor(embeds[:200])
    
    result = model(torch.tensor(embeds))[0][:len(tensors)].detach().numpy()
    y_result = inference(result)

    start, end = find_longest_segment_of_ones(y_result)

    # print(start, end)
    return {
        'start': start,
        'end': end
    }