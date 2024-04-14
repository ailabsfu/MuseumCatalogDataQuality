from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn
import io
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Define the ResNet model as a PyTorch module
class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.base_model = resnet50(pretrained=True)
        self.base_model.fc = nn.Identity()  # Removing the classification layer

    def forward(self, x):
        return self.base_model(x)

# Initialize the model and move it to the available device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleResNet().to(device)
model.eval()

def image_to_embedding(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image)
    return embedding.cpu().numpy()

# Load CSV to map image names to object IDs
df = pd.read_csv('subm_train.csv', sep=';', usecols=['img_name', 'object_id'])
img_to_id = dict(zip(df.img_name, df.object_id))

# Load saved embeddings
embeddings = np.load('saved_embeddings.npy')

app = FastAPI()

# Serve static files from the directory where the similar images are stored
app.mount("/images", StaticFiles(directory=r"C:\Users\user\Desktop\госкатлог\train-t"), name="images")

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Similarity Viewer</title>
    <style>
        #drop-area {
            border: 2px dashed #ccc;
            margin: 10px;
            padding: 20px;
            width: 95%;
            height: 150px;
            text-align: center;
            align-self: center;
        }
        #uploaded-image {
            width: 95%;
            margin: 10px auto;
            text-align: center;
        }
        #uploaded-image img {
            max-width: 100%;
            height: auto;
        }
       #similar-images {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); /* This will create a responsive grid with a minimum column size of 100px */
    grid-gap: 10px; /* This sets the gap between the images */
    margin-top: 10px;
}
#similar-images img {
    width: 100%; /* This will ensure that the images take up the full width of their grid column */
    height: auto; /* This maintains the aspect ratio of the images */
}
    </style>
</head>
<body>
    <div id="drop-area">Загрузите изображение, перетащив его сюда</div>
    <div id="uploaded-image"></div>
    <div id="similar-images"></div>
    
    <!-- Добавляем ползунок -->
    <div>
        <label for="n_neighbors">Количество схожих изображений:</label>
        <input type="range" id="n_neighbors" name="n_neighbors" min="1" max="20" value="10" onchange="updateNeighbors(this.value)">
        <span id="neighbor-value">10</span>
    </div>

    <script>
        const dropArea = document.getElementById('drop-area');
        const uploadedImageDiv = document.getElementById('uploaded-image');
        const similarImagesDiv = document.getElementById('similar-images');
        const neighborValueSpan = document.getElementById('neighbor-value');
        const nNeighborsInput = document.getElementById('n_neighbors');

        dropArea.addEventListener('dragover', (event) => {
            event.stopPropagation();
            event.preventDefault();
            event.dataTransfer.dropEffect = 'copy';
        });

        dropArea.addEventListener('drop', (event) => {
            event.stopPropagation();
            event.preventDefault();
            const files = event.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                uploadFile(file);
            }
        });

        function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);

            fetch('/upload/', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.json())
            .then(data => {
                // Display uploaded image
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedImageDiv.innerHTML = '<img src="' + e.target.result + '" alt="Uploaded Image"/>';
                };
                reader.readAsDataURL(file);

                // Display similar images
                similarImagesDiv.innerHTML = '';
                data.closest_images.forEach(([filename, objectId]) => {
                    const clickableImage = createClickableImage(filename, objectId);
                    similarImagesDiv.appendChild(clickableImage);
                });
            })
            .catch(error => console.error('Error uploading file:', error));
        }

        function createClickableImage(filename, objectId) {
            const imgPath = 'images/' + filename;
            const link = document.createElement('a');
            link.href = `https://goskatalog.ru/portal/#/collections?id=${objectId}`;
            link.target = '_blank';
            const imgElement = document.createElement('img');
            imgElement.src = imgPath;
            link.appendChild(imgElement);
            return link;
        }

        // Функция для обновления значения количества соседей
        function updateNeighbors(value) {
            neighborValueSpan.textContent = value;
            // Отправляем запрос на обновление количества соседей только после завершения изменения ползунка
            clearTimeout(this.timer);
            this.timer = setTimeout(() => {
                fetch('/update_neighbors/' + value)
                    .then(response => response.json())
                    .then(data => {
                        console.log(data);
                        // Вызываем функцию для поиска соседей с новым значением n_neighbors
                        findNeighbors();
                    })
                    .catch(error => console.error('Error updating neighbors:', error));
            }, 500); // 500 мс - задержка перед отправкой запроса
        }

        // Функция для поиска соседей с текущим значением n_neighbors
        function findNeighbors() {
            const formData = new FormData();
            const fileInput = document.querySelector('input[type="file"]');
            formData.append('file', fileInput.files[0]);

            fetch('/upload/', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.json())
            .then(data => {
                // Display uploaded image
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedImageDiv.innerHTML = '<img src="' + e.target.result + '" alt="Uploaded Image"/>';
                };
                reader.readAsDataURL(fileInput.files[0]);

                // Display similar images
                similarImagesDiv.innerHTML = '';
                data.closest_images.forEach(([filename, objectId]) => {
                    const clickableImage = createClickableImage(filename, objectId);
                    similarImagesDiv.appendChild(clickableImage);
                });
            })
            .catch(error => console.error('Error uploading file:', error));
        }
    </script>
</body>
</html>
    """

@app.post("/upload/", response_class=JSONResponse)
async def image_similarity(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        embedding = image_to_embedding(image, model, device)

        filenames = np.load('saved_labels.npy', allow_pickle=True)

        global neighbors  # Use global variable to update number of neighbors
        distances, indices = neighbors.kneighbors(embedding)

        closest_filenames = filenames[indices[0]]
        closest_ids = [img_to_id[filename] for filename in closest_filenames]  # Map filenames to object IDs

        return {"closest_images": list(zip(closest_filenames, closest_ids))}
    else:
        return {"error": "Unsupported file type"}

@app.get("/update_neighbors/{n_neighbors}", response_class=JSONResponse)
async def update_neighbors(n_neighbors: int):
    global neighbors  # Use global variable to update number of neighbors
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
    neighbors.fit(embeddings)
    return {"message": f"Neighbors updated to {n_neighbors}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)