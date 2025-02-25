# Автоматическая разметка

Данный модуль позволяет автоматически размечать данные на основе [алгоритма](https://github.com/ornicar/lichess-puzzler/tree/master/generator), с помощью которого генерируются задачи на lichess.org

# Структура

## annotator.py

Этот скрипт добавляет аннотации к партии, то есть сопоставляет каждому ходу оценку движка.

Есть два сценария использования этого скрипта:

1. **Командная строка**

Вы просто запускаете python файл с необходимыми параметрами, а именно:

- `--input` или `-i` - путь к pgn файлу, для которого вы хотите добавить аннотации;
- `--output` или `-o` - путь к выходному pgn файлу, куда сохранится результат;
- `--clean` - флаг, очищает файл `--output` перед обработкой, требует подтверждения;
- `-y` - флаг, отвечает 'Y' всегда, если программа просит подтверждения;
- `--skip` - количество партий, которые нужно пропустить от начала файла;
- `--quantity` - количество партий, которые нужно обработать;
- `--stockfish` или `-s` - путь к исполняемому файлу движка;
- `--depth` или `-d` - глубина анализа. Для многих задач хватает 20, но если позволяют ресурсы, указывайте больше;
- `--threads` или `-t` - количество потоков, которое будет использовать движок.

Пример использования:
```bash
python annotator.py --insert input.pgn --output output.pgn --stockfish .\stockfish.exe -d 23 -t 4
```

2. **Напрямую в коде**

Вы должны самостоятельно создать движок, задать ему необходимые параметры, а потом создать объект класса `Annotator` и вызвать метод `add_annotations`.

Например:

```python
import chess.engine
from annotator import Annotator

with chess.engine.SimpleEngine.popen_uci(path_to_stockfish) as engine:
    engine.configure({'Threads': num_of_threads})
    annotator = Annotator(engine, depth)
    annotator.add_annotations(input_pgn, output_pgn, overwrite=overwrite_flag)
```

## generator.py

Этот скрипт анализирует партию из PGN файла и выделяет интересные моменты (возможность поставить мат или получить преимущество), сохраняя результат в JSON файл.

Есть два сценария использования этого скрипта:

1. **Командная строка**
Вы можете запустить скрипт через командную строку, указав следующие параметры:
- `--input` или `-i` - путь к входному PGN файлу, который содержит партии для анализа;
- `--output` или `-o` - путь к выходному JSON файлу, куда будут сохранены результаты анализа;
- `--stockfish` или `-s` - путь к исполняемому файлу Stockfish;
- `--threads` или `-t` - количество потоков, которые будут использоваться движком (по умолчанию 1).

Пример использования:
```bash
python generator.py --input games.pgn --output interesting_moments.json --stockfish ./stockfish --threads 4
```

2. **Напрямую в коде**
Вы можете интегрировать функциональность `generator.py` в свой код. Для этого нужно создать движок, настроить его параметры, создать объект класса `Generator` и вызвать метод `generate_interesting`. Пример:
```python
import chess.engine
from generator import Generator

with chess.engine.SimpleEngine.popen_uci(path_to_stockfish) as engine:
    engine.configure({"Threads": num_of_threads})
    generator = Generator(engine)
    generator.generate_interesting(input_pgn, output_json)
```


### finisher.py

Этот скрипт использует шахматный движок **MAIA Engine** для завершения партий, которые могли быть обрезаны после выполнения скрипта `generator.py`.

#### Требования
- Установите шахматный движок [lc0](https://lczero.org/play/download/).
- Разархивируйте архив в одну директорию со скриптом `finisher.py`.
- Убедитесь, что имя папки с движком — **`lc0`**.

#### Методы класса `MaiaEngine`
- **`upload_game(path_to_json)`** — загружает партию из JSON-файла.  
  Формат файла соответствует [примеру](https://github.com/Madagascam/heuristic_extractor/blob/feature/algo-gen-data/automatic-markup/example/example_markup.json).

- **`finish_game()`** — завершает загруженную партию.

- **`output_info(path_to_output_json)`** — сохраняет завершённую партию в JSON-файл. Формат выхода совпадает с форматом входного файла.

---

### Пример использования
```python
with MaiaEngine() as eng:
    eng.upload_game("example_markup1.json")
    eng.finish_game()
    eng.output_info("output1.json")

    eng.upload_game("example_markup2.json")
    eng.finish_game()
    eng.output_info("output2.json")
```
---
