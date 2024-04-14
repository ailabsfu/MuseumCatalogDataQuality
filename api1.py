from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from io import BytesIO
import os
from torch import nn

app = FastAPI()

# Transformation for prediction
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class ImageFolderDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        idx = 0
        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx
                idx += 1
                for img_file in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_file)
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class SimpleResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Load the model (adjust path and settings as necessary)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleResNet(num_classes=15)
model.load_state_dict(torch.load(r"C:\Users\user\Desktop\госкатлог\api\best_model.pth", map_location=device))
model.to(device)
model.eval()


@app.get("/", response_class=HTMLResponse)
def main():
    return """
    <html>
        <head>
            <title>Загрузка изображения для предсказания</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 600px;
                    margin: 50px auto;
                    padding: 20px;
                    background-color: #fff;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }
                h1 {
                    text-align: center;
                    margin-bottom: 30px;
                }
                form {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }
                input[type="file"] {
                    margin-bottom: 20px;
                }
                input[type="submit"] {
                    padding: 10px 20px;
                    background-color: #007bff;
                    color: #fff;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }
                input[type="submit"]:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Загрузка изображения для предсказания</h1>
                <form action="/predict/" enctype="multipart/form-data" method="post">
                    <input name="file" type="file" accept="image/*">
                    <input type="submit" value="Загрузить">
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/predict/", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    # Ensure the directory path here is correct
    root_dir = r"C:\Users\user\Desktop\госкатлог\train_train_2"
    local_dataset = ImageFolderDataset(root_dir, transform=transform)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()

    predicted_class_name = list(local_dataset.class_to_idx.keys())[predicted_class]

    return f"""
    <html>
        <head>
            <title>Результат предсказания</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 600px;
                    margin: 50px auto;
                    padding: 20px;
                    background-color: #fff;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }}
                h1 {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                p {{
                    text-align: center;
                    font-size: 20px;
                    margin-bottom: 20px;
                }}
                strong {{
                    color: #007bff;
                }}
                a {{
                    display: block;
                    text-align: center;
                    text-decoration: none;
                    color: #007bff;
                    font-weight: bold;
                    transition: color 0.3s;
                }}
                a:hover {{
                    color: #0056b3;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Результат предсказания</h1>
                <p>Изображение классифицировано как: <strong>{predicted_class_name}</strong></p>
                <a href="/">Загрузить ещё одно изображение</a>
            </div>
        </body>
    </html>
    """
