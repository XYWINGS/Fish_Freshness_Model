import os
import torch
import torchvision
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import transforms
import cv2
import numpy as np
from transformers import ViTModel, ViTConfig  # Use Hugging Face's ViTModel

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model/multi_attribute_fish_model_novel2.pth"

# Define class names
class_names = ["fresh", "non-fresh"]

# Custom Model Definition (same as in training script)
class MultiAttributeFishModel(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(MultiAttributeFishModel, self).__init__()
        
        # Pre-trained CNNs for feature extraction
        self.eye_cnn = torchvision.models.resnet18(pretrained=False)
        self.eye_cnn.fc = torch.nn.Linear(self.eye_cnn.fc.in_features, 128)
        
        self.gill_cnn = torchvision.models.resnet18(pretrained=False)
        self.gill_cnn.fc = torch.nn.Linear(self.gill_cnn.fc.in_features, 128)
        
        # Vision Transformer for global context (using Hugging Face's ViTModel)
        self.vit = ViTModel(ViTConfig.from_pretrained("google/vit-base-patch16-224"))
        self.vit_fc = torch.nn.Linear(self.vit.config.hidden_size, 128)
        
        # Attention Mechanism for Eyes and Gills
        self.eye_attention = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(128, 1),
            torch.nn.Softmax(dim=1)
        )
        self.gill_attention = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(128, 1),
            torch.nn.Softmax(dim=1)
        )
        
        # Weighted Fusion Layer
        self.fusion_fc = torch.nn.Sequential(
            torch.nn.Linear(128 * 3, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, eye_img, gill_img):
        # Extract features using CNNs
        eye_features = self.eye_cnn(eye_img)
        gill_features = self.gill_cnn(gill_img)
        
        # Apply Attention Mechanisms
        eye_weights = self.eye_attention(eye_features)
        gill_weights = self.gill_attention(gill_features)
        
        eye_features = eye_features * eye_weights
        gill_features = gill_features * gill_weights
        
        # Extract global context using ViT
        vit_outputs = self.vit(eye_img)
        vit_features = vit_outputs.last_hidden_state[:, 0]  # Use the [CLS] token output
        vit_features = self.vit_fc(vit_features)
        
        # Concatenate features
        combined_features = torch.cat([eye_features, gill_features, vit_features], dim=1)
        
        # Final classification
        output = self.fusion_fc(combined_features)
        return output

# Load the saved model
if os.path.exists(MODEL_PATH):
    model = MultiAttributeFishModel(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Model loaded from: {os.path.abspath(MODEL_PATH)}")
else:
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
model.eval()  # Set to evaluation mode

# Preprocessing Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Domain-specific preprocessing for fish eyes and gills
def preprocess_image(image, attribute):
    """
    Domain-specific preprocessing for fish eyes and gills.
    """
    image = np.array(image)
    
    if attribute == "eyes":
        # Reddish color detection for non-fresh eyes
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_red = np.array([0, 50, 50])  # Lower range for red
        upper_red = np.array([10, 255, 255])  # Upper range for red
        red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
        
        # Glitter/reflectivity detection for fresh eyes
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)  # Detect edges
        
        # Combine masks
        combined_mask = cv2.bitwise_or(red_mask, edges)
        image = cv2.bitwise_and(image, image, mask=combined_mask)
    
    elif attribute == "gills":
        # Enhance color contrast for gills
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab_image = cv2.merge((l_channel, a_channel, b_channel))
        image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(image)

# Run inference
def infer(image):
    # Preprocess the image for both eyes and gills
    eye_image = preprocess_image(image, "eyes")
    gill_image = preprocess_image(image, "gills")
    
    # Apply transformations
    eye_image = transform(eye_image).unsqueeze(0).to(DEVICE)
    gill_image = transform(gill_image).unsqueeze(0).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        outputs = model(eye_image, gill_image)
        _, predicted_class_idx = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0, predicted_class_idx].item()
    
    # Map index to class name
    predicted_class = class_names[predicted_class_idx.item()]
    return predicted_class, confidence

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    try:
        image = Image.open(file.stream)
        predicted_class, confidence = infer(image)
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)