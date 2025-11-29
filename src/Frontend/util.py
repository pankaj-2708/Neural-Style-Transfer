from PIL import Image
import requests
import base64

backend_uri = "http://16.16.78.98:8000/"


def predict_output(id):
    r = requests.get(f"{backend_uri}/process_images/?id={id}")
    if r.status_code == 500:
        return -1
    return r.content


def predict_output_video(id):
    r = requests.get(f"{backend_uri}/process_video/?id={id}")
    if r.status_code == 500:
        return -1
    return r.content


def upload_images(style, content):
    files = []
    files.append(("files", (style.name, style.getvalue(), style.type)))
    files.append(("files", (content.name, content.getvalue(), content.type)))

    r = requests.post(f"{backend_uri}/upload_images", files=files).json()
    return r["id"]


def upload_video(style, content):
    files = []
    files.append(("files", (style.name, style.getvalue(), style.type)))
    files.append(("files", (content.name, content.getvalue(), content.type)))

    r = requests.post(f"{backend_uri}/upload_video", files=files).json()
    return r["id"]
