from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse

from model.predictor import predict_image

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Pasar a función de predicción
    result = predict_image(contents)

    return JSONResponse(content=result)