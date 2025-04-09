from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import torch.nn as nn

# Define EAST model
class EAST(nn.Module):
    def __init__(self):
        super(EAST, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)  
        )
        self.merge = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU()
        )
        self.score = nn.Conv2d(32, 1, 1)
        self.geo = nn.Conv2d(32, 5, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        x = self.merge(x)
        score = torch.sigmoid(self.score(x))  
        geo = self.geo(x)                     
        return score, geo

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EAST().to(device)

# Load the correct model checkpoint
checkpoint = torch.load("east_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

app = FastAPI()

# Image preprocessing function
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (256, 256)) 
    image = image / 255.0  
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    return image
@app.post("/debug-upload")
async def debug_upload(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "received": True
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
        image_tensor = preprocess_image(image)

        print("Image successfully received")
        print("Image tensor shape:", image_tensor.shape)

        with torch.no_grad():
            score, geo = model(image_tensor)
            prediction = score.squeeze().cpu().numpy().tolist() 

        print("Model prediction complete")
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app using: uvicorn app:app --reload

