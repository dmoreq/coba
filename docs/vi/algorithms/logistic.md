# Logistic Bandits (Logistic TS / Logistic UCB)

## 1. Giải thích Trực quan (The Intuition)
Cả LinUCB và LinTS đều có một điểm yếu chết người: Chúng ngầm định rằng phần thưởng (reward) là một đường thẳng tuyến tính trải dài từ âm sang dương (Ví dụ: Doanh thu 10k, 20k, 50k). 
Nhưng trong thực tế của hệ thống Pricing, mục tiêu quan trọng nhất thường là **Tỷ lệ chuyển đổi (Conversion Rate)**: Khách hàng có CHẤP NHẬN mức giá này hay KHÔNG? (Reward chỉ có thể là 0 hoặc 1).

Nếu dùng đường thẳng (LinUCB) để dự đoán số 0 và 1, mô hình thường xuyên đoán ra các con số vô lý như xác suất = 1.5 hoặc xác suất = -0.2. 

**Logistic Bandits** ra đời để giải quyết bài toán này. Nó ép mọi dự đoán phải bẻ cong thành hình chữ S (hàm Sigmoid), đảm bảo mọi dự đoán luôn là một xác suất hợp lý từ 0% đến 100%.

## 2. Cơ chế Hoạt động (Online Laplace Approximation)
Việc cập nhật liên tục (online) cho mạng hình chữ S rất tốn kém tài nguyên tính toán (thường phải dùng Gradient Descent nhiều lần).

Trong `coba`, chúng ta sử dụng một kỹ thuật toán học gọi là **Online Laplace Approximation**. Thay vì tính toán lại toàn bộ, mỗi khi có khách hàng click hay bỏ qua, mô hình chỉ thực hiện đúng **1 bước cập nhật** (1-step Newton-Raphson). Điều này giúp Logistic Bandits chạy siêu nhanh (độ phức tạp $O(d^2)$) ngang ngửa với LinUCB, đáp ứng cực tốt cho môi trường Real-time API.

## 3. Toán học Cốt lõi & Tham số
Với ngữ cảnh $x$, xác suất khách hàng đồng ý mua được dự đoán bằng hàm Sigmoid:
$$ p = \frac{1}{1 + e^{-x^T w}} $$

* Khi thuật toán chọn **Logistic UCB**: Nó tính toán khoảng tin cậy của $x^T w$, rồi đưa qua hàm Sigmoid để ra giới hạn xác suất tối đa (Upper Bound).
* Khi thuật toán chọn **Logistic TS**: Nó bốc ngẫu nhiên một mẫu hệ số $w_{sample}$ từ phân phối Laplace Gaussian, nhân với $x$ và đưa qua hàm Sigmoid.

**Tham số:** Kế thừa toàn bộ các tham số của LinUCB (`alpha`) và LinTS (`v_sq`), đồng thời hỗ trợ cả cơ chế quên lãng (`gamma`).

## 4. Ví dụ Chạy Độc Lập (Code Mẫu)

```python
import numpy as np
from coba.policies.logistic import LogisticTSArmModel

# Khởi tạo mô hình Logistic Thompson Sampling
model = LogisticTSArmModel(
    arm="price_100k", 
    n_features=2, 
    v_sq=1.0, 
    gamma=0.99
)

# Khách hàng xuất hiện
context = np.array([1.0, 0.5])

# Dự đoán xác suất khách hàng mua hàng (Luôn nằm trong khoảng [0, 1])
# Do dùng TS, xác suất này chứa đựng cả tỷ lệ ngẫu nhiên để khám phá
prob = model.score(context)
print(f"Xác suất mua hàng dự kiến (Score): {prob * 100:.2f}%")

# Khách hàng TỪ CHỐI mức giá này (Reward = 0.0)
model.update(context, reward=0.0)

# Điểm số lần sau khi gặp ngữ cảnh tương tự sẽ giảm đi rõ rệt
new_prob = model.score(context)
print(f"Xác suất mua hàng cập nhật: {new_prob * 100:.2f}%")
```

## 5. Nguồn Tham Khảo (References)

> Chapelle, O., & Li, L. (2011). *An empirical evaluation of thompson sampling*. In Advances in neural information processing systems (NeurIPS). (Phân tích chi tiết về áp dụng Laplace Approximation cho Logistic Bandits).
