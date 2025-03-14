import json
from g4f import ChatCompletion, Provider

class Prompt:
    def __init__(self, prompt_text: str, model: str, provider: Provider, verify_func=None):
        """
        Инициализация промпта.

        :param prompt_text: Текст промпта.
        :param model: Название модели.
        :param provider: Провайдер для модели.
        :param verify_func: Функция верификации ответа. Принимает строку (ответ) и возвращает преобразованные данные.
        """
        self.prompt_text = prompt_text
        self.model = model
        self.provider = provider
        self.verify_func = verify_func

    def _process_response(self, response: str) -> str:
        """
        Обрабатывает ответ модели.
        Если в ответе содержится тег '</think>', возвращает часть после него, иначе – очищенный ответ.
        """
        if "</think>" in response:
            return response.split("</think>", 1)[1].strip()
        return response.strip()

    def get_response(self, additional_context: str = "", stream: bool = False, max_attempts: int = 4) -> str:
        """
        Получает ответ от модели, объединяя основной текст промпта с дополнительным контекстом.
        При неудаче повторяет запрос до max_attempts.
        """
        full_prompt = self.prompt_text + (f"\n{additional_context}" if additional_context else "")
        attempt = 0
        while attempt < max_attempts:
            try:
                response = ChatCompletion.create(
                    model=self.model,
                    provider=self.provider,
                    messages=[{"role": "user", "content": full_prompt}],
                    stream=stream,
                )
                return self._process_response(response)
            except Exception as e:
                attempt += 1
                if attempt < max_attempts:
                    print(f"Попытка {attempt} не удалась, пробую еще раз...")
                else:
                    raise RuntimeError(
                        f"Ошибка при получении ответа от модели {self.model} после {max_attempts} попыток: {e}"
                    ) from e

    def get_verified_response(self, additional_context: str = "", stream: bool = False, max_attempts: int = 4):
        """
        Получает ответ от модели и проводит его верификацию с помощью заданной функции.
        Если функция верификации не задана, по умолчанию пытается разобрать JSON.
        """
        response = self.get_response(additional_context=additional_context, stream=stream, max_attempts=max_attempts)
        if self.verify_func is not None:
            try:
                return self.verify_func(response)
            except Exception as e:
                raise RuntimeError(f"Верификация ответа не удалась: {e}") from e
        else:
            try:
                return json.loads(response)
            except Exception as e:
                raise RuntimeError(
                    f"Ответ не является валидным JSON и функция верификации не задана: {e}"
                ) from e
