from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from highlighter import find_highlight
import os

app = FastAPI()

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
