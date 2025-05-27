from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from highlighter import find_highlight
import os

app = FastAPI()

def create_pgn(moves_str, output_file):
    """Преобразует последовательность ходов в pgn-файл
    
    Пример использования:
    moves_str = "e2e4 c7c6 f2f4 d7d5 e4d5 c6d5 g1f3 b8c6"
    output_file = "output.pgn"
    create_pgn(moves_str, output_file)
    """
    pgn_header = "[Event \"?\"]\n[Site \"?\"]\n[Date \"????.??.??\"]\n[Round \"?\"]\n[White \"?\"]\n[Black \"?\"]\n[Result \"*\"]\n\n"
    moves = moves_str.split()

    move_number = 1
    pgn_moves = []
    for i in range(0, len(moves), 2):
        if i + 1 < len(moves):
            pgn_moves.append(f"{move_number}. {moves[i]} {moves[i+1]}")
            move_number += 1
        else:
            pgn_moves.append(f"{move_number}. {moves[i]}")

    pgn_content = pgn_header + " ".join(pgn_moves) + " *"

    # Записываем в файл
    with open(output_file, 'w') as f:
        f.write(pgn_content)


@app.post("/upload-pgn/")
async def upload_pgn(file: UploadFile = File(...)):
    try:
        # Сохраняем загруженный файл временно
        with open("temp.pgn", "wb") as buffer:
            buffer.write(await file.read())

        # Получаем интересный момент
        interesting_moment = find_highlight("temp.pgn")
        start, end = interesting_moment['start'], interesting_moment['end']

        # Удаляем временный файл
        os.remove("temp.pgn")

        return JSONResponse(content={"start": start, "end": end})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
