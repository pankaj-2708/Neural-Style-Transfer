from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File ,Query,Response
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
import shutil
import os
import cv2
import time
from PIL import Image
import io
from util import load_models,download_artifacts,run_model

app = FastAPI()
id=0

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# downloading artifacts
encoder_local_path,decoder_local_path=download_artifacts()

# loading models
encoder,decoder=load_models(encoder_local_path,decoder_local_path)

# input output names
input_name_encoder = encoder.get_inputs()[0].name
output_name_encoder = encoder.get_outputs()[0].name

input_name_decoder = decoder.get_inputs()[0].name
output_name_decoder = decoder.get_outputs()[0].name

@app.post("/upload_images")
async def upload_images(files:List[UploadFile] = File(...)):
    try:
        # save image to s3-> upgrade
        global id
        os.makedirs(os.path.join("images"),exist_ok=True)
        id+=1
        
        i=0
        for file in files:
            content = await file.read()
            img = Image.open(io.BytesIO(content))
            
            if i==0:       
                img.save(os.path.join(f"images/{id}_style.png"))
                i+=1
            else:
                img.save(os.path.join(f"images/{id}_content.png"))
                
        return JSONResponse({"id":str(id)},status_code=200)
    except Exception as E:
        print(E)
        return Response(status_code=500)
    
    
@app.post("/upload_video")
async def upload_video(files:List[UploadFile] = File(...)):
    try:
        # save image to s3-> upgrade
        global id
        os.makedirs(os.path.join("videos"),exist_ok=True)
        id+=1
        
        i=0
        for file in files:
            
            if i==0:       
                content = await file.read()
                img = Image.open(io.BytesIO(content))
                img.save(os.path.join("videos",f"{id}_style.png"))
                i+=1
            else:
                with open(os.path.join("videos",f"{id}_content.mp4"), "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                
        return JSONResponse({"id":str(id)},status_code=200)
    except Exception as E:
        print(E)
        return Response(status_code=500)

@app.get("/process_images")
def process_image(id: str = Query(..., title="id", description="automaticall generated id ")):
    try:
        start_time=time.time()
        style=np.array(Image.open(os.path.join(f"images/{id}_style.png")))
        content=np.array(Image.open(os.path.join(f"images/{id}_content.png")))
        
        
        ouput_image=run_model(style,content,encoder,decoder)
        ouput_image.save(os.path.join(f"images/{id}_output.png"))
        
        
        buf = io.BytesIO()
        ouput_image.save(buf, format="PNG")
        bytes_data = buf.getvalue()
        
        print("time_taken",time.time()-start_time)
        return Response(bytes_data, media_type="image/png")
    
    except Exception as E:
        print()
        print(E)
        return Response(status_code=500)
        
        
@app.get("/process_video")
def process_video(id: str = Query(..., title="id", description="automaticall generated id ")):
    try:
        start_time=time.time()
        
        style=np.array(Image.open(os.path.join(f"videos/{id}_style.png")))
        content_path=os.path.join(f"videos/{id}_content.mp4")
        
        cap=cv2.VideoCapture(content_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        fourcc=cv2.VideoWriter_fourcc(*"mp4v")
        out=cv2.VideoWriter(os.path.join(f"videos/{id}_output.mp4"),fourcc,fps,(512,512))

        while True:
            ret , frame = cap.read()
            if not ret:      # video ended or frame not readable
                break
            content=np.array(frame)
            ouput_image=np.array(run_model(style,content,encoder,decoder))
            out.write(ouput_image)
        
        out.release()
        
        with open(f"videos/{id}_output.mp4", "rb") as f:
            video_bytes = f.read()
        
        print("time_taken",time.time()-start_time)
        return Response(video_bytes, media_type="video/mp4")
    
    except Exception as E:
        print()
        print(E)
        return Response(status_code=500)
        
        
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)