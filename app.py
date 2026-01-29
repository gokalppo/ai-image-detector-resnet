import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr


# 1. UPLOADING MODEL
def load_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # 'map_location=torch.device('cpu')' This is important because we haven't adjusted the GPU settings on your computer.
    model.load_state_dict(torch.load('ai_image_detector_resnet18.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


model = load_model()

# 2. UPDATED CONVERTER
# We first reduce and degrade the quality photos to 32x32, then increase them to the model size (224).
transform_fix = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 3. PREDICTION FUNCTION
def predict(image):
    # The image from Gradio might sometimes be a NumPy array; let's convert it to a PIL Image.
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image_tensor = transform_fix(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    return {"FAKE": float(probabilities[0]), "REAL": float(probabilities[1])}


# 4. INTERFACE
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="AI Generated Image Detector 🕵️‍♂️",
    description="An AI detection tool trained with ResNet18 and using Transfer Learning."
)

if __name__ == "__main__":
    interface.launch()