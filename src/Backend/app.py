from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Query, Response
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import time
from PIL import Image
from io import BytesIO
import io
from util import (
    load_models,
    download_artifacts,
    run_model,
    upload_file_to_s3,
    retrive_file,
    upload_video_to_s3,
)
import tempfile

app = FastAPI()
id = 0

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# downloading artifacts
encoder_local_path, decoder_local_path = download_artifacts()
# loading models
encoder, decoder = load_models(encoder_local_path, decoder_local_path)


@app.post("/upload_images")
async def upload_images(files: List[UploadFile] = File(...)):
    try:
        # save image to s3-> upgrade
        global id
        # os.makedirs(os.path.join("images"),exist_ok=True)
        id += 1

        i = 0
        for file in files:
            content = await file.read()
            # img = Image.open(io.BytesIO(content))

            if i == 0:
                # img.save(os.path.join(f"images/{id}_style.png"))
                upload_file_to_s3(f"images_{id}_style.png", content, file.content_type)
                i += 1
            else:
                # img.save(os.path.join(f"images/{id}_content.png"))
                upload_file_to_s3(f"images_{id}_content.png", content, file.content_type)

        return JSONResponse({"id": str(id)}, status_code=200)
    except Exception as E:
        print(E)
        return Response(status_code=500)


@app.post("/upload_video")
async def upload_video(files: List[UploadFile] = File(...)):
    try:
        # save image to s3-> upgrade
        global id
        # os.makedirs(os.path.join("videos"),exist_ok=True)
        id += 1

        i = 0
        for file in files:
            content = await file.read()

            if i == 0:
                # img = Image.open(io.BytesIO(content))
                # img.save(os.path.join("videos",f"{id}_style.png"))
                upload_file_to_s3(f"videos_{id}_style.png", content, file.content_type)
                i += 1
            else:
                # with open(os.path.join("videos",f"{id}_content.mp4"), "wb") as buffer:
                #     shutil.copyfileobj(file.file, buffer)
                upload_file_to_s3(f"videos_{id}_content.mp4", content, file.content_type)

        return JSONResponse({"id": str(id)}, status_code=200)
    except Exception as E:
        print(E)
        return Response(status_code=500)


@app.get("/process_images")
async def process_image(
    id: str = Query(..., title="id", description="automaticall generated id ")
):
    try:
        start_time = time.time()
        style = np.array(Image.open(BytesIO(retrive_file(f"images_{id}_style.png"))))
        content = np.array(Image.open(BytesIO(retrive_file(f"images_{id}_content.png"))))

        ouput_image = run_model(style, content, encoder, decoder)

        buf = io.BytesIO()
        ouput_image.save(buf, format="PNG")
        bytes_data = buf.getvalue()

        upload_file_to_s3(f"images_{id}_output.png", bytes_data, "image/png")

        print("time_taken", time.time() - start_time)
        return Response(bytes_data, media_type="image/png")

    except Exception as E:
        print()
        print(E)
        return Response(status_code=500)


@app.get("/process_video")
def process_video(id: str = Query(..., title="id", description="automaticall generated id ")):
    try:
        start_time = time.time()

        style = np.array(Image.open(BytesIO(retrive_file(f"videos_{id}_style.png"))))
        video_bytes = retrive_file(f"videos_{id}_content.mp4")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            content_path = tmp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            output_path = tmp.name

        cap = cv2.VideoCapture(content_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (512, 512))

        while True:
            ret, frame = cap.read()
            if not ret:  # video ended or frame not readable
                break
            content = np.array(frame)
            ouput_image = np.array(run_model(style, content, encoder, decoder))
            out.write(ouput_image)

        out.release()

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        upload_file_to_s3(f"videos_{id}_output.mp4", video_bytes, "video/mp4")

        print("time_taken", time.time() - start_time)
        return Response(video_bytes, media_type="video/mp4")

    except Exception as E:
        print()
        print(E)
        return Response(status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
