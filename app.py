import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr


# 1. MODELİ YÜKLEME
def load_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # 'map_location=torch.device('cpu')' önemli, çünkü bilgisayarında GPU ayarı yapmadık
    model.load_state_dict(torch.load('ai_image_detector_resnet18.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


model = load_model()

# 2. GÜNCELLENMİŞ DÖNÜŞTÜRÜCÜ (SENİN BULDUĞUN ÇÖZÜM)
# Kaliteli fotoları önce 32x32'ye indirip bozuyoruz, sonra modelin boyutuna (224) çıkarıyoruz.
transform_fix = transforms.Compose([
    transforms.Resize((32, 32)),  # Kaliteyi kasıtlı düşür
    transforms.Resize((224, 224)),  # Modele uygun boyuta getir
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 3. TAHMİN FONKSİYONU
def predict(image):
    # Gradio'dan gelen resim bazen NumPy array olabilir, onu PIL Image'a çevirelim
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image_tensor = transform_fix(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Colab'deki class_to_idx çıktısına göre burayı güncellemen gerekebilir!
    # Genelde: 0=FAKE, 1=REAL
    return {"FAKE": float(probabilities[0]), "REAL": float(probabilities[1])}


# 4. ARAYÜZ
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="AI Generated Image Detector 🕵️‍♂️",
    description="An AI detection tool trained with ResNet18 and using Transfer Learning."
)

if __name__ == "__main__":
    interface.launch()