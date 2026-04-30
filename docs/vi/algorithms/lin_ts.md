# Thuật toán LinTS (Linear Thompson Sampling)

## 1. Giải thích Trực quan (The Intuition)
Thay vì giống LinUCB cố gắng định lượng chính xác "khoảng an toàn" cho từng quyết định, **LinTS** hành xử như một chuyên gia luôn có nhiều kịch bản (scenarios) trong đầu.

Mỗi khi cần ra quyết định, LinTS không đưa ra một đáp án cố định. Thay vào đó, nó tung xúc xắc (lấy mẫu ngẫu nhiên) để chọn ra một "giả thuyết" (hypothesis) từ vô số kịch bản nó đang nghĩ tới. Kịch bản nào càng phù hợp với dữ liệu lịch sử thì càng dễ được chọn (xác suất cao).
Nhờ tính chất ngẫu nhiên này, hệ thống sẽ tự nhiên "vô tình" thử nghiệm những arm chưa quen, nhưng vẫn luôn có xu hướng bám sát những arm đã chứng minh hiệu quả.

## 2. Cơ chế Hoạt động
LinTS hoạt động theo trường phái Thống kê Bayesian:
1. **Prior & Posterior:** Nó duy trì một "niềm tin" (phân phối xác suất Gaussian) về các hệ số quan hệ giữa thị trường (ngữ cảnh) và lợi nhuận. Khi chưa có dữ liệu, niềm tin này tỏa đều mọi hướng (khám phá nhiều). Khi có nhiều dữ liệu, niềm tin hội tụ lại quanh các giá trị chính xác (khai thác nhiều).
2. **Sampling (Lấy mẫu):** Tại thời điểm ra quyết định, nó bốc ngẫu nhiên một bộ hệ số từ phân phối niềm tin này, nhân với dữ liệu hiện tại để ra một dự đoán, và quyết định dựa trên dự đoán đó.

> **Ưu điểm thực tế:** Trong môi trường Production với hàng ngàn request mỗi giây, dữ liệu bị trễ (delayed feedback) là điều bình thường. LinUCB có thể liên tục chọn sai một arm hàng nghìn lần trước khi nhận được kết quả cập nhật. Nhưng LinTS, nhờ vào yếu tố lấy mẫu ngẫu nhiên, sẽ tự động phân tán các lựa chọn, giúp hệ thống không bị "kẹt" vào một arm sai trong thời gian chờ dữ liệu.

## 3. Toán học Cốt lõi & Tham số

Dựa trên phân phối hậu nghiệm (Posterior Distribution) của hệ số hồi quy:
$$\tilde{\beta} \sim \mathcal{N}(\hat{\beta}, v^2 A^{-1})$$

Trong đó:
- $\hat{\beta}$: Hệ số tốt nhất mô hình học được tính tới hiện tại (Mean).
- $A^{-1}$: Ma trận hiệp phương sai (Covariance) đại diện cho độ không chắc chắn.
- **Tham số `v_sq` ($v^2$):** Hệ số khuếch đại phương sai. `v_sq` càng lớn, phân phối càng phình to, hệ thống lấy mẫu càng phân tán (Khám phá nhiều hơn). Mặc định là `1.0`.
- Tham số `gamma`: Tương tự LinUCB, dùng để lãng quên quá khứ trong môi trường phân phối thay đổi liên tục.

Điểm cuối cùng để đánh giá tay đòn: $Score = x^T \tilde{\beta}$.

## 4. Ví dụ Chạy Độc Lập (Code Mẫu)

```python
import numpy as np
from coba.policies.lin_ts import LinTSArmModel

# Khởi tạo mô hình LinTS cho một arm
model = LinTSArmModel(
    arm="variant_A",
    n_features=3,
    v_sq=1.0,   # Hệ số ngẫu nhiên
    gamma=0.99  # Tốc độ lãng quên (decay)
)

context = np.array([0.8, 0.2, 0.9])

# Lấy điểm ưu tiên.
# LƯU Ý: Vì LinTS mang tính ngẫu nhiên, mỗi lần gọi hàm score với cùng
# một context sẽ trả về một điểm khác nhau! Điều này tạo ra sự khám phá.
score_1 = model.score(context)
score_2 = model.score(context)
print(f"Lần 1: {score_1:.4f} | Lần 2: {score_2:.4f}")

# Cập nhật kết quả từ thực tế
model.update(context, reward=1.0)

# Càng nhiều dữ liệu, độ chênh lệch giữa các lần lấy mẫu sẽ càng nhỏ dần.
```

## 5. Nguồn Tham Khảo (References)

Thuật toán Linear Thompson Sampling được chứng minh lý thuyết và áp dụng rộng rãi bởi Agrawal và Goyal.

> Agrawal, S., & Goyal, N. (2013). *Thompson sampling for contextual bandits with linear payoffs*. In International conference on machine learning (ICML).
>
> Chapelle, O., & Li, L. (2011). *An empirical evaluation of thompson sampling*. In Advances in neural information processing systems (NeurIPS).
