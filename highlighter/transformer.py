# Импорты необходимых библиотек
import torch  # Основная библиотека для работы с тензорами и нейронными сетями
import torch.nn as nn  # Модуль PyTorch для создания нейронных сетей
import torch.nn.init as init  # Модуль для инициализации весов


class BinaryClassifierTransformer(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=8, dim_feedforward=256, dropout=0.1):
        """
        Инициализация модели классификатора на основе Transformer.
        
        Параметры:
        - input_dim: Размерность входных данных (число признаков).
        - d_model: Размерность эмбеддингов (по умолчанию 256).
        - nhead: Число голов в механизме внимания MultiHeadAttention (по умолчанию 8).
        - num_layers: Количество слоев TransformerEncoder (по умолчанию 8).
        - dim_feedforward: Размерность скрытого слоя в feedforward сети Transformer (по умолчанию 256).
        - dropout: Вероятность dropout для регуляризации (по умолчанию 0.1).
        """
        super().__init__()
        
        # 1. Улучшенный embedding слой
        # Преобразует входные данные размерности input_dim в пространство d_model
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),  # Линейное преобразование входных данных
            nn.LayerNorm(d_model),          # Нормализация по слою для стабилизации обучения
            nn.Dropout(dropout),            # Dropout для предотвращения переобучения
            nn.GELU(),                      # Активационная функция GELU
            nn.Linear(d_model, d_model),    # Дополнительное линейное преобразование
            nn.LayerNorm(d_model),          # Нормализация по слою
            nn.GELU()                       # Активационная функция GELU
        )
        
        # 2. Позиционные эмбеддинги
        # Добавляются к входным данным для учета порядка последовательности
        self.pos_encoder = nn.Parameter(torch.randn(1, 200, d_model))  # Рандомные позиционные эмбеддинги
        
        # 3. Увеличенный трансформер
        # Создание одного слоя TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,                # Размерность эмбеддингов
            nhead=nhead,                    # Число голов в механизме внимания
            dim_feedforward=dim_feedforward,  # Размерность скрытого слоя
            dropout=dropout,                # Dropout для регуляризации
            batch_first=True,               # Входные данные имеют формат [batch_size, seq_len, features]
            activation='gelu'               # Активационная функция GELU
        )
        # Стек из num_layers слоев TransformerEncoder
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Расширенный классификатор
        # Выполняет классификацию на основе выходов трансформера
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),    # Линейное преобразование
            nn.GELU(),                      # Активационная функция GELU
            nn.Linear(d_model, d_model // 2),  # Снижение размерности
            nn.GELU(),                      # Активационная функция GELU
            nn.Dropout(dropout),            # Dropout для регуляризации
            nn.Linear(d_model // 2, 1)      # Финальный слой для бинарной классификации
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """
        Инициализация весов модели с использованием Xavier uniform initialization.
        Это помогает улучшить сходимость модели.
        """
        for p in self.parameters():
            if p.dim() > 1:  # Инициализируем только тензоры с размерностью больше 1
                init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Прямой проход модели.
        
        Параметры:
        - x: Входные данные размерности [batch_size, seq_len, input_dim].
        
        Возвращает:
        - output: Вероятности классов размерности [batch_size, seq_len].
        """
        # 1. Преобразование входных данных через embedding слой
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        
        # 2. Добавление позиционных эмбеддингов
        x = x + self.pos_encoder[:, :x.size(1), :]  # Добавляем позиционные эмбеддинги
        
        # 3. Пропуск данных через TransformerEncoder
        x = self.transformer(x)  # [batch_size, seq_len, d_model]
        
        # 4. Классификация и применение сигмоиды
        return torch.sigmoid(self.classifier(x)).squeeze(-1)  # [batch_size, seq_len]