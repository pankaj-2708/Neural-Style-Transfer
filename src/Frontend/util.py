from PIL import Image
import requests
import base64

backend_uri=""
def predict_output(id):
    r = requests.get("http://localhost:8000/upload_images", json={"id":id})
    if(r.status_code==500):
        return -1
    return r.content

def upload_images(style,content):
    files = []
    files.append(
            ("files", (style.name, style.getvalue(), style.type))
        )
    files.append(
            ("files", (content.name, content.getvalue(), content.type))
        )
    
    r = requests.post("http://localhost:8000/process_images", files=files)
    return r['id']