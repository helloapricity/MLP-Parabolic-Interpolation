import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # Khởi tạo ma trận A (784x512)
        self.hidden = nn.Linear(28*28, 512) 
        # Khởi tạo ma trận B (512x10)
        self.output = nn.Linear(512, 10)    
        
    def forward(self, x):
        x = x.view(-1, 28*28)           # Biến đổi ảnh 2D thành 1D
        x = F.relu(self.hidden(x))      # Lớp ẩn + ReLU
        x = self.output(x)              # Lớp đầu ra
        return x