import boto3
import onnxruntime as ort
import mlflow
import cv2
import os
from PIL import Image
import numpy as np


mlflow.set_tracking_uri("http://ec2-13-51-234-7.eu-north-1.compute.amazonaws.com:5000/")

print(mlflow.get_tracking_uri())


def prepare(img):
    # img shape: (H, W, 3), dtype=uint8
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # CHW → NCHW
    return img


def get_mean_std(x, epsilon=1e-5):
    axes = (2, 3)
    mean, variance = np.mean(x, axis=axes, keepdims=True), np.var(x, axis=axes, keepdims=True)
    standard_deviation = np.sqrt(variance + epsilon)
    return mean, standard_deviation


def ada_in(style, content):
    mean_style, std_style = get_mean_std(style)
    mean_content, std_content = get_mean_std(content)

    return std_style * (content - mean_content) / std_content + mean_style


def download_artifacts(
    encoder_run_id="0179cde372fb4f60bfb93deb104e682f",
    decoder_run_id="d0c790f0a1fe43fa9218e6b238a1d930",
):
    encoder_uri = f"runs:/{encoder_run_id}/encoder"
    decoder_uri = f"runs:/{decoder_run_id}/decoder"
    encoder_local_path = mlflow.artifacts.download_artifacts(encoder_uri)
    decoder_local_path = mlflow.artifacts.download_artifacts(decoder_uri)
    print("Encoder at ", encoder_local_path)
    print("Decoder at ", decoder_local_path)
    return encoder_local_path, decoder_local_path


def load_models(encoder_local_path, decoder_local_path):
    encoder = ort.InferenceSession(
        encoder_local_path + "model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    decoder = ort.InferenceSession(
        decoder_local_path + "model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    print(encoder.get_providers())
    print(decoder.get_providers())
    return encoder, decoder


def run_model(style, content, encoder, decoder):
    style = prepare(cv2.resize(style, (512, 512)))
    content = prepare(cv2.resize(content, (512, 512)))

    style_encoded = encoder.run(["relu4_1"], {"input": style})[0]
    content_encoded = encoder.run(["relu4_1"], {"input": content})[0]

    t = ada_in(style_encoded, content_encoded).astype(np.float32)

    # alpha can be added for cutomisation-> read paper

    out = decoder.run(["output"], {"features": t})[0]
    img = out[0].transpose(1, 2, 0)  # (3,512,512) → (512,512,3)
    img = np.clip(img, 0, 255).astype(np.uint8)

    return Image.fromarray(img)


s3 = boto3.client("s3")


def upload_file_to_s3(filename, file_body, file_content_type, bucket="pankaj-nst"):
    s3.put_object(Bucket=bucket, Key=filename, Body=file_body, ContentType=file_content_type)
    print(filename, " uploaded sucessfully to s3")


def upload_video_to_s3(file, filename, bucket="pankaj-nst"):
    # If S3 object_name was not specified, use file_name
    s3.put_object(Bucket=bucket, Key=filename, Body=file, ContentType="video/mp4")
    print(filename, " uploaded sucessfully to s3")


def retrive_file(filename, bucket="pankaj-nst"):
    return s3.get_object(Bucket="pankaj-nst", Key=filename)["Body"].read()
