# Lab 1: Giải phẫu Mạng Nơ-ron và Tối ưu hóa bằng Nội suy Parabol

## 1. Giới thiệu Dự án
Đây là mã nguồn thực hành cho bài Lab 01 - Khảo sát mạng nơ-ron học sâu (Deep Learning). Dự án sử dụng framework mã nguồn mở PyTorch để xây dựng một mạng Multilayer Perceptron (MLP) cơ bản, phân loại tập dữ liệu chữ số viết tay MNIST.

Điểm nhấn cốt lõi của dự án là việc **thay thế hoàn toàn phương pháp tính Gradient Descent truyền thống bằng phương pháp Nội suy Parabol (QuickProp)** để cập nhật và học tham số mạng, đáp ứng chính xác yêu cầu khảo sát cơ chế tối ưu hóa.

## 2. Cấu trúc Thư mục
- `models/simple_mlp.py`: Định nghĩa kiến trúc mạng nơ-ron 3 lớp (Input, Hidden, Output).
- `utils/parabolic.py`: Chứa class `ParabolicOptimizer`, tự tay triển khai logic toán học cập nhật trọng số bằng thuật toán Parabol và cơ chế xử lý biên.
- `train.py`: Vòng lặp huấn luyện chính, kết nối mô hình, dữ liệu và bộ tối ưu hóa. Có tích hợp tự động nhận diện thiết bị tính toán (CPU/GPU).
- `requirements.txt`: Danh sách các thư viện nền tảng cần thiết.
- `training_results.txt`: File báo cáo tự động sinh ra sau khi huấn luyện, ghi nhận sự sụt giảm của sai số (Loss).

## 3. Hướng dẫn Cài đặt & Sử dụng (Step-by-Step)

**Bước 1: Cài đặt thư viện**
Mở Terminal hoặc Command Prompt tại thư mục chứa dự án và chạy lệnh sau để thiết lập môi trường:
```bash
pip install -r requirements.txt
```

**Bước 2: Chạy chương trình huấn luyện**
Thực thi file chính để bắt đầu quá trình học (hệ thống sẽ tự động tải dữ liệu MNIST nếu thư mục `data/` chưa có):

```bash
python train.py
```

## 4. Kết quả
Chương trình sẽ tiến hành lan truyền tiến, lan truyền ngược và tự động gọi bộ tối ưu Parabol để tinh chỉnh ma trận trọng số. Mức độ sai số (Loss) giảm dần qua các chu kỳ sẽ được in trực tiếp ra màn hình và tự động lưu trữ vào tệp `training_results.txt`. Cuối cùng, trọng số học được sẽ được trích xuất ra file `mlp_parabolic_weights.pth`.