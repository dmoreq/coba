# Thuật toán LinUCB (Linear Upper Confidence Bound)

## 1. Giải thích Trực quan (The Intuition)
Hãy tưởng tượng bạn là một chuyên gia ra quyết định. Khi gặp một tình huống quen thuộc (dữ liệu phong phú), bạn khá tự tin về lựa chọn tốt nhất. Nhưng khi gặp một tình huống hiếm (ít dữ liệu), bạn không chắc chắn lắm.

**LinUCB** giải quyết vấn đề này bằng cách cộng thêm một "điểm thưởng tò mò" (exploration bonus) vào những tình huống mà nó chưa từng gặp. Nó làm theo triết lý: *"Lạc quan trong sự không chắc chắn"* (Optimism in the face of uncertainty). Nếu một tình huống quá mới lạ, nó sẵn sàng thử một arm chưa quen để thu thập thêm dữ liệu, nhưng nếu đã có kinh nghiệm, nó sẽ dùng dữ liệu đó để khai thác tối đa.

## 2. Cơ chế Hoạt động

LinUCB kết hợp hai yếu tố để ra quyết định:
1. **Exploitation (Khai thác):** Dự đoán phần thưởng trung bình dựa trên một mô hình hồi quy tuyến tính (Ridge Regression).
2. **Exploration (Khám phá):** Đo lường mức độ "chưa chắc chắn" (Uncertainty). Những tình huống (vector ngữ cảnh) càng khác biệt so với dữ liệu lịch sử, độ không chắc chắn càng cao, điểm tò mò càng lớn.

Điểm tổng (Score) = Khai thác + Khám phá. Hệ thống sẽ chọn arm có điểm tổng cao nhất.

## 3. Toán học Cốt lõi & Tham số

Với mỗi tay đòn $a$, khi nhận một vector ngữ cảnh $x \in \mathbb{R}^d$:

$$ Score_a(x) = x^T \hat{\beta}_a + \alpha \sqrt{x^T A_a^{-1} x} $$

- $x^T \hat{\beta}_a$: Dự đoán phần thưởng (Khai thác).
- $\sqrt{x^T A_a^{-1} x}$: Độ lệch chuẩn (phương sai) dự kiến của dự đoán (Khám phá). Ma trận $A_a$ là ma trận hiệp phương sai lưu trữ những vector ngữ cảnh mà mô hình đã từng học.
- **Tham số `alpha`:** Hệ số điều chỉnh. `alpha = 0` nghĩa là chỉ tin vào mô hình (Không tò mò). `alpha` càng lớn, thuật toán càng muốn thử nghiệm ở các vùng dữ liệu mới. (Mặc định thường từ 0.5 đến 2.0).
- **Tham số `gamma` (Non-stationary):** Hệ số lãng quên. Nếu `gamma < 1.0` (ví dụ 0.99), ma trận lịch sử $A$ sẽ dần bị triệt tiêu, giúp mô hình quên đi dữ liệu cũ để thích nghi với sự thay đổi của môi trường.

## 4. Ví dụ Chạy Độc Lập (Code Mẫu)

Dưới đây là ví dụ khởi tạo và chạy thuật toán LinUCB cho một arm đơn lẻ mà không cần thông qua `ClusterRouter`.

```python
import numpy as np
from coba.policies.linucb import LinUCBArmModel

# Khởi tạo mô hình cho một arm
model = LinUCBArmModel(
    arm="variant_A",
    n_features=3,
    alpha=1.0,  # Độ tò mò trung bình
    gamma=0.99  # Quên đi 1% dữ liệu cũ sau mỗi lần update để thích nghi tốt hơn
)

context = np.array([0.8, 0.2, 0.9])

# Lần đầu tiên, điểm sẽ khá cao vì hệ thống chưa biết gì (tò mò)
score = model.score(context)
print(f"Điểm ưu tiên (lần đầu): {score}")

# Quan sát kết quả và cập nhật mô hình
model.update(context, reward=1.0)

# Lần này độ tò mò giảm xuống, thuật toán dự đoán chính xác hơn
new_score = model.score(context)
print(f"Điểm ưu tiên (sau khi cập nhật): {new_score}")
```

## 5. Nguồn Tham Khảo (References)

Thuật toán LinUCB được giới thiệu và ứng dụng thành công lần đầu tiên tại Yahoo! để cá nhân hoá việc hiển thị bài viết tin tức.

> Chu, W., Li, L., Reyzin, L., & Schapire, R. (2011). *Contextual bandits with linear payoff functions*. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS).
>
> Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). *A contextual-bandit approach to personalized news article recommendation*. In Proceedings of the 19th international conference on World wide web (WWW).
