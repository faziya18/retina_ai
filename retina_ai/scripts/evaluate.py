import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_trained_model(model_path, num_classes=5):
    """Load the trained ResNet50 model."""
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model architecture
    model = models.resnet50(weights=None)  # Don't load ImageNet weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def evaluate_model(model, val_loader, device, class_names):
    """Evaluate model on validation set and return metrics."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average=None, zero_division=0)
    
    # Print results
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nPer-class metrics:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plot_confusion_matrix(cm, class_names)
    
    return accuracy, precision, recall, f1, cm

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix using seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def inference_single_image(model, image_path, device, class_names):
    """Run inference on a single image."""
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score, probabilities.cpu().numpy()

def main():
    # Paths
    DATA_DIR = "/Users/srinivasanc/Downloads/datasets/APTOS2019_224x224/colored_images"
    MODEL_PATH = "dr_resnet50.pth"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        return
    
    # Transformations (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset to get validation split
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")
    
    # Split dataset (same as training)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Load trained model
    print("Loading trained model...")
    model, device = load_trained_model(MODEL_PATH)
    print(f"Model loaded successfully on device: {device}")
    
    # Evaluate model
    print("\nEvaluating model on validation set...")
    accuracy, precision, recall, f1, cm = evaluate_model(model, val_loader, device, class_names)
    
    # Example inference on a single image (if you have one)
    # Uncomment and modify the path below to test single image inference
    # test_image_path = "path/to/your/test/image.jpg"
    # if os.path.exists(test_image_path):
    #     predicted_class, confidence, probabilities = inference_single_image(
    #         model, test_image_path, device, class_names
    #     )
    #     print(f"\nSingle image inference:")
    #     print(f"Predicted class: {predicted_class}")
    #     print(f"Confidence: {confidence:.4f}")
    #     print(f"All probabilities: {dict(zip(class_names, probabilities))}")

if __name__ == "__main__":
    main()