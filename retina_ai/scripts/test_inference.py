"""
Test script to demonstrate single image inference functionality.
This script shows how to use the trained model for inference on individual images.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

def load_model(model_path="dr_resnet50.pth"):
    """Load the trained model."""
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model architecture
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)  # 5 classes
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def predict_single_image(model, image_path, device):
    """Run inference on a single image."""
    # Define transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Class names (same order as training)
    class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    all_probabilities = probabilities.cpu().numpy()
    
    return predicted_class, confidence_score, dict(zip(class_names, all_probabilities))

def main():
    print("🔬 Diabetic Retinopathy Single Image Inference Test")
    print("=" * 50)
    
    # Check if model exists
    model_path = "dr_resnet50.pth"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("Please make sure you have trained the model first by running the fundus training script.")
        return
    
    # Load model
    print("📥 Loading trained model...")
    model, device = load_model(model_path)
    print(f"✅ Model loaded successfully on device: {device}")
    
    # Example usage
    print("\n📋 Instructions:")
    print("1. Place your test image in the same directory as this script")
    print("2. Uncomment the example code below and update the image path")
    print("3. Run this script again to see the prediction")
    
    print("\n💡 Example usage:")
    print("""
    # Uncomment and modify the following lines:
    # test_image_path = "your_test_image.jpg"  # Update with your image path
    # if os.path.exists(test_image_path):
    #     predicted_class, confidence, all_probs = predict_single_image(model, test_image_path, device)
    #     
    #     print(f"\\n🎯 Prediction Results for: {test_image_path}")
    #     print(f"Predicted Class: {predicted_class}")
    #     print(f"Confidence: {confidence:.1%}")
    #     print("\\nAll Class Probabilities:")
    #     for class_name, prob in all_probs.items():
    #         print(f"  {class_name}: {prob:.1%}")
    # else:
    #     print(f"❌ Image file not found: {test_image_path}")
    """)
    
    # You can uncomment and test with an actual image:
    # test_image_path = "sample_retinal_image.jpg"  # Update this path
    # if os.path.exists(test_image_path):
    #     predicted_class, confidence, all_probs = predict_single_image(model, test_image_path, device)
    #     
    #     print(f"\n🎯 Prediction Results for: {test_image_path}")
    #     print(f"Predicted Class: {predicted_class}")
    #     print(f"Confidence: {confidence:.1%}")
    #     print("\nAll Class Probabilities:")
    #     for class_name, prob in all_probs.items():
    #         print(f"  {class_name}: {prob:.1%}")

if __name__ == "__main__":
    main()