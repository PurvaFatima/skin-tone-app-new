from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
from model import SkinToneModel

app = FastAPI()

# Allow CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to your Flutter app’s domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = SkinToneModel(num_classes=10)
model.load_state_dict(torch.load('resnet_checkpoint.pth', map_location=torch.device('cpu')))
model.eval()

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Color palettes
palettes = {
    0: ("Porcelain", ["#F5D0C3", "#FFE4E1", "#E6E6FA", "#D8BFD8", "#98FF98", "#FFB6C1", "#F0FFFF", "#87CEEB", "#FFF0F5", "#FFDAB9"]),
    1: ("Fair", ["#FFC0CB", "#B0E0E6", "#E0B0FF", "#FAFAD2", "#FFD700", "#D2B48C", "#AFEEEE", "#E6E6FA", "#FFA07A", "#FFE4B5"]),
    2: ("Light Beige", ["#FFE5B4", "#C19A6B", "#B2AC88", "#AFDBF5", "#FF7F50", "#E6E6FA", "#FFDB58", "#008080", "#F5DEB3", "#DAA520"]),
    3: ("Warm Beige", ["#CC5500", "#50C878", "#FFDB58", "#800000", "#E2725B", "#556B2F", "#CD853F", "#FF6347", "#F5DEB3", "#8B4513"]),
    4: ("Honey", ["#800020", "#228B22", "#FFD700", "#D35400", "#00008B", "#8B0000", "#F5DEB3", "#8B4513", "#673147", "#FFA500"]),
    5: ("Golden Brown", ["#722F37", "#014421", "#FFD700", "#556B2F", "#4B3621", "#950714", "#D2691E", "#E97451", "#FFA500", "#7B3F00"]),
    6: ("Deep Tan", ["#9B111E", "#0F52BA", "#E2725B", "#CD7F32", "#F5DEB3", "#800000", "#3C341F", "#FF7518", "#580F41", "#A0522D"]),
    7: ("Rich Mocha", ["#811A21", "#50C878", "#B87333", "#191970", "#FF6700", "#8B0000", "#7B3F00", "#D2B48C", "#E97451", "#CD7F32"]),
    8: ("Espresso", ["#4169E1", "#FFD700", "#D2691E", "#DC143C", "#008080", "#3C341F", "#B7410E", "#673147", "#800020", "#C0C0C0"]),
    9: ("Ebony", ["#FFFF00", "#FFFFFF", "#FF69B4", "#00FF00", "#FFD700", "#0000FF", "#A020F0", "#FF0000", "#00FFFF", "#FF8C00"])
}

def predict_skin_tone(image: Image.Image) -> dict:
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        pred_class = output.argmax().item()  # 0-based class index
    skin_tone, colors = palettes.get(pred_class, ("Unknown", ["#FFFFFF"]))
    return {
        "skin_tone": f"Class {pred_class + 1} ({skin_tone})",
        "recommended_colors": colors
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert('RGB')
    result = predict_skin_tone(image)
    return result 
