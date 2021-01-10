import torch
import torch.nn as nn
import torch.functional as F

class CNN_OShaugnessy(nn.Module):
    
    def __init__(self, img_channels=1, M=2):
        """ The MNIST CNN classifier introduced by OShaugnessy et al. (see Table 1, Appendix D).

        Args:
        - img_channels: int, the number of channels the MNIST images have
        - M: int, the number of output nodes
            
        """
        super().__init__()
        
        self.img_channels = img_channels
        self.M = M
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=self.img_channels, out_channels=32,
                      kernel_size=3, stride=1, padding=0),          # 28x28 -> 26x26
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=0),          # 26x26 -> 24x24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                            # 24x24 -> 23x23
            
            # Linear Layer
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(in_features=9216, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128,  out_features=self.M)
        )

    def forward(self, X):
        Y = self.net(X)
        
        return Y
