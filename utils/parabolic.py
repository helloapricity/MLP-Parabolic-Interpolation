import torch

class ParabolicOptimizer:
    def __init__(self, parameters, alpha=0.01):
        self.params = list(parameters)
        self.alpha = alpha  # Hệ số học cơ sở dùng cho bước fallback
        
        # Khởi tạo danh sách tensor lưu trữ lịch sử cho toàn bộ kiến trúc mạng
        self.prev_grads = [torch.zeros_like(p) for p in self.params]
        self.prev_steps = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad(): # Vô hiệu hóa autograd để tinh chỉnh tham số thủ công
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                
                # Thu thập biến trạng thái từ bộ đệm
                delta_t = p.grad                 # Đạo hàm hiện tại
                delta_t_minus_1 = self.prev_grads[i] # Đạo hàm chu kỳ trước
                c_t_minus_1 = self.prev_steps[i]     # Bước nhảy chu kỳ trước
                
                # Tính toán sự chênh lệch (độ dốc đạo hàm)
                grad_diff = delta_t_minus_1 - delta_t
                c_t = torch.zeros_like(p) # Khởi tạo Tensor chứa bước cập nhật
    
                # Giải quyết bài toán chia cho 0 hoặc khi thuật toán mới chạy ở bước t=0
                mask_zero = (torch.abs(grad_diff) < 1e-6) | (c_t_minus_1 == 0)
                
                # Rút lui về Gradient Descent chuẩn
                c_t[mask_zero] = -self.alpha * delta_t[mask_zero]

                # CÔNG THỨC PARABOL CỐT LÕI (QuickProp)
                mask_valid = ~mask_zero
                c_t[mask_valid] = (delta_t[mask_valid] / grad_diff[mask_valid]) * c_t_minus_1[mask_valid]
                
                # Giới hạn bước nhảy không cho vượt quá đoạn [-0.1, 0.1]
                c_t = torch.clamp(c_t, min=-0.1, max=0.1)
                
                # Cập nhật trọng số của mạng
                p.add_(c_t)

                # Ghi đè bộ nhớ đệm lịch sử cho chu kỳ t+1 
                self.prev_grads[i].copy_(delta_t) 
                self.prev_steps[i].copy_(c_t)