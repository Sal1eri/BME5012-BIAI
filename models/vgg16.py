import torch
import torch.nn as nn
from torchvision import models

class VGG16ForAD(nn.Module):
    def __init__(self, num_classes=5):
        """        
        Args:
            num_classes (int): Number of classes (default: 5)
        """
        super(VGG16ForAD, self).__init__()
        
        self.num_classes = num_classes
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # Replace the final fully connected layer (4096 -> num_classes)
        in_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(in_features, num_classes)
        
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self.eval()

    def forward(self, x):
        return self.vgg16(x)
    
if __name__ == "__main__":
    model = VGG16ForAD(num_classes=5)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print("-" * 30)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print("-" * 30)