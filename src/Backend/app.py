from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File ,Query,Response
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
from src.Backend.util import load_models,download_artifacts,run_model
import os
from PIL import Image
import io

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
    # save image to s3-> upgrade
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


@app.get("/process_images")
def process_image(id: str = Query(..., title="id", description="automaticall generated id ")):
    try:
        style=Image.open(io.BytesIO(os.path.join(f"images/{id}_style.png")))
        content=Image.open(io.BytesIO(os.path.join(f"images/{id}_content.png")))
        
        ouput_image=run_model(style,content,encoder,decoder,input_name_encoder,output_name_encoder,input_name_decoder,output_name_decoder)
        ouput_image.save(os.path.join(f"images/{id}_output.png"))
        buf = io.BytesIO()
        ouput_image.save(buf, format="PNG")
        bytes_data = buf.getvalue()
        
        return Response(bytes_data, media_type="image/png")
    
    except Exception as E:
        print(E)
        return Response(status_code=500)
        
        