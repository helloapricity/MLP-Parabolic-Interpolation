import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models.simple_mlp import SimpleMLP
from utils.parabolic import ParabolicOptimizer

# --- BƯỚC 1: Thiết lập thiết bị (CPU hoặc GPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chuẩn bị Dữ liệu
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=64, 
    shuffle=True,
    num_workers=2,      
    pin_memory=True     
)

# --- BƯỚC 2: Chuyển Mô hình lên GPU ---
model = SimpleMLP().to(device) 
criterion = nn.CrossEntropyLoss() 
optimizer = ParabolicOptimizer(model.parameters(), alpha=0.001)


if __name__ == '__main__':
    print(f"Đang chạy huấn luyện trên thiết bị: {device}")
    print("Bắt đầu tiến trình huấn luyện...")

    # Vòng lặp Huấn luyện
    epochs = 20 

    with open("training_results.txt", "w", encoding="utf-8") as f:
        f.write(f"BÁO CÁO KẾT QUẢ HUẤN LUYỆN (Thiết bị: {device})\n")
        f.write("=" * 60 + "\n")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for images, labels in train_loader:
                # --- BƯỚC 3: Đẩy dữ liệu (batch) lên GPU ---
                images, labels = images.to(device), labels.to(device)
                
                # 1. Lan truyền tiến
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # 2. Xóa đạo hàm
                model.zero_grad()
                
                # 3. Lan truyền ngược
                loss.backward()
                
                # 4. Cập nhật bằng Parabol
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            result_line = f"Chu kỳ (Epoch) [{epoch+1}/{epochs}] - Sai số (Loss): {avg_loss:.4f}"
            
            print(result_line)
            f.write(result_line + "\n")
            
        f.write("\nQuá trình huấn luyện đã hoàn tất!")

    # Lưu "Bộ não" của mô hình
    torch.save(model.state_dict(), "mlp_parabolic_weights.pth")
    print("Đã lưu trọng số thành công!")