import torch
import torch.nn as nn
import torch.optim as optim
from src.model import SimCLR, Classifier,Encoder
from src.data import get_Dataloader
from src.train_contrastive import train_simclr
from src.train_MLP import train_classifier
from src.test_MLP import test_classifier
from src.visualize import visualize_predictions

# Load CIFAR-10 dataset
train_loader, test_loader = get_Dataloader()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Model, Optimizer for SimCLR
model = SimCLR().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the SimCLR model
train_simclr(model, train_loader, optimizer, device, epochs=50)

# Save only the encoder
torch.save(model.encoder.state_dict(), "simclr_encoder.pth")
print("Encoder saved successfully!")

# Initialize Encoder, Classifier, Optimizer for linear evaluation
# Load pre-trained encoder and freeze it
encoder = Encoder()
encoder.load_state_dict(torch.load("simclr_encoder.pth"))
encoder.eval()  # Set encoder to evaluation mode (not necessary but good practice)

# Create classifier model
model = Classifier(encoder).to("cuda" if torch.cuda.is_available() else "cpu")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Train only the classifier

# Train the classifier
train_classifier(model, train_loader, criterion, optimizer, device, epochs=10)

# Test the classifier
test_classifier(model, test_loader, device)

# Visualize predictions
visualize_predictions(model, test_loader, device, num_images=10)



