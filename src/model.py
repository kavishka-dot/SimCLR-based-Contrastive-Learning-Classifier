import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models

# Encoder: ResNet18 (removing classification head)
class Encoder(nn.Module):
    '''ResNet18 as encoder for SimCLR'''
    def __init__(self):
        super(Encoder, self).__init__()
        base_model = models.resnet18(pretrained=False) # Load pre-trained ResNet18
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # Remove last FC layer

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # Flatten

# Projection Head: 2-layer MLP
class ProjectionHead(nn.Module):
    '''Projection head for SimCLR'''
    def __init__(self, input_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# SimCLR Model
class SimCLR(nn.Module):
    '''SimCLR model with encoder and projection head'''
    def __init__(self):
        super(SimCLR, self).__init__()
        self.encoder = Encoder()
        self.projection_head = ProjectionHead()

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return z
    
# Classifier: Shallow 1-layer MLP
class Classifier(nn.Module):
    '''Shallow 1-layer MLP for linear evaluation'''
    def __init__(self, encoder, num_classes=10):
        super(Classifier, self).__init__()
        self.encoder = encoder  # Pretrained encoder
        self.fc = nn.Linear(512, num_classes)  # Shallow classifier (1 layer)

    def forward(self, x):
        with torch.no_grad():  # Freeze encoder during forward pass
            x = self.encoder(x)
        return self.fc(x)  # Only train the classifier