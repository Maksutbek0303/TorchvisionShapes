from fastapi import FastAPI, UploadFile, File, HTTPException
import torch.nn as nn
import io, torch
import uvicorn
from torchvision import transforms
from PIL import Image

classes = ['circle', 'square', 'star', 'triangle']

class CheckImage(nn.Module):
   def __init__(self):
     super().__init__()
     self.first = nn.Sequential(
          nn.Conv2d(3, 32, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2),

          nn.Conv2d(32, 64, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2),

          nn.Conv2d(64, 128, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2),
          )

     self.second = nn.Sequential(
          nn.Flatten(),
          nn.Linear(128 * 16 * 16, 256),
          nn.ReLU(),
          nn.Linear(256, 6),

          )
   def forward(self, x):
     x = self.first(x)
     x = self.second(x)
     return x

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()])

check_router = FastAPI()

model_path = 'model.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CheckImage()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

@check_router.post('/predict/')
async def check_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=404, detail="No image")
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tenso = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(image_tenso)
            pred = y_pred.argmax(dim=1).item()
        return {'Answer': classes[pred]}


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(check_router, host='127.0.0.1', port=8000)