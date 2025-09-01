
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms,datasets
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
from torch import nn
from function import load
import torch

# TODO: replace with your own architecture
from torchvision.models import efficientnet_b0

app = FastAPI()
app.mount("/static", StaticFiles(directory="."), name="static")  # serves index.html at /
# @app.get("/")
# def home():
#     return {"message": "Server jalan"}
@app.get("/", response_class=FileResponse)
def read_index():
    return "static/index.html"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Load model ----
# Adjust to your architecture & number of classes
num_classes = 3
class_names = ["rafa", "master", "owi"]

# Create model architecture
weight = EfficientNet_B0_Weights.IMAGENET1K_V1
loadedmodel = efficientnet_b0(weights=weight)
loadedmodel.classifier = nn.Sequential(
    torch.nn.Dropout(p=0.2,inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=num_classes,
                    bias=True).to(device))

# Load your trained weights
load(model=loadedmodel,Saved_path="preTrainedModel.pth")
loadedmodel = loadedmodel.to(device)
loadedmodel.eval()

# ---- Preprocessing (match your training!) ----
# Change size/mean/std to match how you trained the model
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@torch.inference_mode()
def predict_image_bytes(b: bytes):
    img = Image.open(BytesIO(b)).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(device)  # shape [1, C, H, W]
    logits = loadedmodel(x)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    label = class_names[int(idx)] if int(idx) < len(class_names) else str(int(idx))
    return {"label": label, "confidence": float(conf)}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    b = await file.read()
    return predict_image_bytes(b)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
