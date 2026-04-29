# Các Giải Thuật Học Máy Meta (Scikit-Learn Wrappers)

## 1. Giải thích Trực quan (The Intuition)
Cả LinUCB, LinTS hay Logistic đều yêu cầu mối quan hệ giữa ngữ cảnh và phần thưởng phải là một đường tuyến tính. Nhưng thực tế thì rối rắm hơn thế nhiều (Non-linear). Ví dụ: Một kết hợp feature đặc biệt (như giờ + ngày + vùng) có thể tạo ra hiệu ứng hoàn toàn khác so với từng feature riêng lẻ.

Các mô hình học máy hiện đại như **Random Forest** hay **LightGBM** là bậc thầy trong việc tìm ra những quy luật phi tuyến tính phức tạp này. Vấn đề duy nhất là chúng sinh ra chỉ để "Khai thác" (Exploitation) — chúng không có tư duy "Tò mò" (Exploration) của một Bandit.

Nhóm thuật toán **Meta-Heuristics** (bao gồm Epsilon Greedy và Bootstrapped) được sinh ra như một lớp "vỏ bọc" thông minh. Chúng bọc lấy các mô hình học máy trên, ép chúng phải khám phá bằng các mẹo toán học tinh vi.

## 2. Bootstrapped Thompson Sampling / UCB
**Cách hoạt động:**
Thuật toán này duy trì một danh sách (Ensemble) gồm nhiều mô hình giống hệt nhau (ví dụ 10 mô hình LightGBM nhỏ).
* Khi có dữ liệu mới, hệ thống không dạy cho tất cả 10 mô hình giống nhau. Thay vào đó, nó tung xúc xắc (phân phối Poisson) để quyết định "độ chú ý" của từng mô hình đối với dữ liệu đó. (Có mô hình học rất kỹ, có mô hình bỏ qua). Quá trình này gọi là *Online Bootstrapping*.
* Kết quả là, ta có 10 mô hình chuyên môn khác nhau một chút. Khi cần ra giá (Thompson Sampling), ta bốc ngẫu nhiên 1 trong 10 ông ra để tham khảo. Sự khác biệt giữa 10 ông chính là sự "Khám phá ngẫu nhiên" (Exploration).

## 3. Contextual Epsilon Greedy
**Cách hoạt động:**
Đây là mẹo đơn giản nhất quả đất. Thuật toán có đúng 1 mô hình LightGBM.
* 90% thời gian (Xác suất $1 - \epsilon$), nó sẽ ngoan ngoãn nghe lời mô hình để tối đa hoá lợi nhuận.
* 10% thời gian (Xác suất $\epsilon$), nó phớt lờ hoàn toàn mô hình, nhắm mắt đưa ra một mức giá bừa bãi cực kỳ lớn (hoặc ngẫu nhiên) để thử phản ứng của thị trường.

## 4. Ví dụ Chạy Độc Lập (Code Mẫu)

Dưới đây là cách sử dụng `BootstrappedTSArmModel` kết hợp với thư viện `LightGBM` đỉnh cao.

```python
import numpy as np
from lightgbm import LGBMRegressor
from coba.policies.sklearn_models import BootstrappedTSArmModel

# Khởi tạo mô hình Tree-based
# Cần cài đặt lightgbm: pip install lightgbm
base_model = LGBMRegressor(n_estimators=10, max_depth=3)

model = BootstrappedTSArmModel(
    arm="variant_A",
    rng=np.random.default_rng(),
    base_estimator=base_model,
    n_bootstraps=5  # Giữ 5 mô hình LightGBM trong một arm
)

context = np.array([[1.0, 0.5, 0.2]])  # LightGBM thường yêu cầu ma trận 2D

# Tính điểm ưu tiên (Mỗi lần gọi có thể bốc trúng 1 mô hình LightGBM khác nhau)
score = model.score(context)
print(f"Điểm từ LightGBM (Bootstrapped): {score}")

# Cập nhật kết quả vào ensemble
# Mỗi mô hình trong ensemble sẽ cập nhật với mức độ ảnh hưởng ngẫu nhiên
model.update(context, reward=1.0)
```

## 5. Nguồn Tham Khảo (References)

> Eckles, D., & Kaptein, M. (2014). *Thompson sampling with the online bootstrap*. arXiv preprint arXiv:1410.4009. (Mô tả lý thuyết cực kỳ nền tảng để biến một mô hình học máy bình thường thành Contextual Bandit).
