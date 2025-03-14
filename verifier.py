import json

class ResponseVerifier:
    def __init__(self, expected_format: str, valid_score_range: tuple = (1, 10)):
        """
        Инициализация верификатора.

        :param expected_format: Ожидаемый формат ответа. Возможные значения:
            - "single": список из двух целых чисел.
            - "single_with_score": список из трёх целых чисел (последний — оценка).
            - "multiple": список списков из двух целых чисел.
            - "multiple_with_score": список списков из трёх целых чисел (последний — оценка).
        :param valid_score_range: Допустимый диапазон для оценки (score), по умолчанию (1, 10).
        """
        self.expected_format = expected_format
        self.valid_score_range = valid_score_range

    def _check_segment(self, segment, length: int) -> bool:
        """
        Проверяет, что сегмент имеет нужную длину и состоит из целых чисел.
        Если ожидается оценка (length == 3), дополнительно проверяет, что оценка входит в допустимый диапазон.
        """
        if not isinstance(segment, list):
            return False
        if len(segment) != length:
            return False
        for i in range(length):
            if not isinstance(segment[i], int):
                return False
        if length == 3:
            score = segment[2]
            if not (self.valid_score_range[0] <= score <= self.valid_score_range[1]):
                return False
        return True

    def verify(self, response: str):
        """
        Проверяет, что ответ соответствует ожидаемому формату.

        :param response: Строка с ответом (в формате JSON).
        :return: Декодированный объект, если проверка прошла успешно.
        :raises ValueError: если ответ не соответствует ожидаемому формату.
        """
        try:
            data = json.loads(response)
        except Exception as e:
            raise ValueError("Ответ не является валидным JSON.") from e

        if self.expected_format == "single":
            if not self._check_segment(data, 2):
                raise ValueError("Ответ должен быть списком из двух целых чисел.")
            return data
        elif self.expected_format == "single_with_score":
            if not self._check_segment(data, 3):
                raise ValueError(
                    "Ответ должен быть списком из трёх целых чисел, где третий элемент — оценка от {} до {}."
                    .format(self.valid_score_range[0], self.valid_score_range[1])
                )
            return data
        elif self.expected_format == "multiple":
            if not isinstance(data, list):
                raise ValueError("Ответ должен быть списком.")
            for item in data:
                if not self._check_segment(item, 2):
                    raise ValueError("Каждый элемент ответа должен быть списком из двух целых чисел.")
            return data
        elif self.expected_format == "multiple_with_score":
            if not isinstance(data, list):
                raise ValueError("Ответ должен быть списком.")
            for item in data:
                if not self._check_segment(item, 3):
                    raise ValueError(
                        "Каждый элемент ответа должен быть списком из трёх целых чисел, где третий элемент — оценка от {} до {}."
                        .format(self.valid_score_range[0], self.valid_score_range[1])
                    )
            return data
        else:
            raise ValueError(f"Неизвестный формат верификации: {self.expected_format}")
