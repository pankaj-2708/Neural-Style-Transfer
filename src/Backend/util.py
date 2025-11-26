import onnxruntime as ort
import mlflow
import numpy as np

def get_mean_std(x, epsilon=1e-5):
    axes = (1, 2)
    mean, variance = np.mean(x, axis=axes,keepdims=True), np.var(x, axis=axes, keepdims=True)
    standard_deviation = np.sqrt(variance + epsilon)
    return mean, standard_deviation

def ada_in(style,content):
    mean_style,std_style=get_mean_std(style)
    mean_content,std_content=get_mean_std(content)

    return std_style*(content-mean_content)/std_content+mean_style


def download_artifacts(encoder_run_id="7ded6fa5e95c43bbae26eca6a5c807d9",decoder_run_id="972d39c3b7d94ea394bb4bb04403cdbb"):
    encoder_uri = f"runs:/{encoder_run_id}/encoder"
    decoder_uri = f"runs:/{decoder_run_id}/decoder"
    encoder_local_path=mlflow.artifacts.download_artifacts(encoder_uri)
    decoder_local_path=mlflow.artifacts.download_artifacts(decoder_uri)
    return encoder_local_path,decoder_local_path



def load_models(encoder_local_path,decoder_local_path):
    encoder=ort.InferenceSession(
        encoder_local_path+'model.onnx',
        providers=["CUDAExecutionProvider"]   # GPU
    )
    decoder=ort.InferenceSession(
        decoder_local_path+'model.onnx',
        providers=["CUDAExecutionProvider"]   # GPU
    )
    
    return encoder,decoder
    

def run_model(style,content,encoder,decoder,input_name_encoder,output_name_encoder,input_name_decoder,output_name_decoder):
    style_encoded=encoder.run([output_name_encoder],{input_name_encoder:np.expand_dims(np.array(style),axis=0).astype(np.float32)})[0][0]
    content_encoded=encoder.run([output_name_encoder],{input_name_encoder:np.expand_dims(np.array(content),axis=0).astype(np.float32)})[0][0]

    bot=ada_in(style_encoded,content_encoded)

    out=decoder.run([output_name_decoder],{input_name_decoder:np.expand_dims(np.array(bot),axis=0).astype(np.float32)})[0]

    return out[0]