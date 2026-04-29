# Các Thuật Toán Không Theo Ngữ Cảnh (Context-Free)

## 1. Giải thích Trực quan (The Intuition)
Trong hệ thống `coba`, phần lớn các thuật toán đều là Contextual (Dựa vào ngữ cảnh). 

Nhưng đôi khi, bạn không có bất kỳ thông tin gì về khách hàng (Không biết họ ở đâu, thời tiết ra sao, giờ nào). Bài toán lúc này trở thành **A/B Testing** truyền thống, nhưng thông minh hơn. Bạn chỉ có 2 tay đòn (ví dụ: Nút đỏ và Nút xanh) và muốn tìm xem nút nào có tỷ lệ click cao hơn, mà không cần chia đều 50-50 ngân sách một cách lãng phí như A/B Test. 

Các giải thuật Context-Free ra đời để giải bài toán này. Chúng tìm kiếm nút tốt nhất càng nhanh càng tốt, dựa hoàn toàn vào dữ liệu lịch sử click.

## 2. UCB1 (Upper Confidence Bound 1)
**Đặc điểm:** Tất định (Deterministic). Phù hợp cho số lượng tay đòn ít.
**Cách hoạt động:** Nó tính Tỷ lệ chuyển đổi (Conversion Rate) trung bình của mỗi mức giá. Sau đó cộng thêm một khoảng tự tin (Confidence Interval). Khoảng tự tin này sẽ cực kỳ lớn nếu tay đòn đó ít được chọn, ép thuật toán phải chọn nó để kiểm chứng.

**Toán học:**
$$ Score_i = \hat{\mu}_i + \alpha \sqrt{\frac{\ln(N)}{n_i}} $$
Trong đó $N$ là tổng số lượt hiển thị toàn hệ thống, $n_i$ là số lượt hiển thị của mức giá $i$.

## 3. Thompson Sampling Cơ bản (Beta-Bernoulli)
**Đặc điểm:** Ngẫu nhiên (Stochastic). Hiệu năng thực tế rất xuất sắc.
**Cách hoạt động:** Mỗi tay đòn có 2 con số: Lượt thành công (`alpha`) và Lượt thất bại (`beta`). Khi cần quyết định, hệ thống tung xúc xắc dựa trên phân phối Beta(alpha, beta) cho từng mức giá. Mức giá nào có điểm tung xúc xắc cao nhất sẽ chiến thắng.

**Toán học:**
$$ Score_i \sim \text{Beta}(\alpha_i + \text{reward}, \beta_i + (1 - \text{reward})) $$

## 4. Ví dụ Chạy Độc Lập (Code Mẫu)

```python
import numpy as np
from coba.policies.thompson import ThompsonArmModel

# Khởi tạo mô hình Thompson Sampling không có ngữ cảnh
model = ThompsonArmModel(
    arm="price_100k",
    rng=np.random.default_rng(),
    alpha_prior=1.0, # Giả định ban đầu có 1 lượt thành công
    beta_prior=1.0   # Giả định ban đầu có 1 lượt thất bại
)

# Xin điểm ưu tiên (Không cần truyền context vector)
# Hệ thống sẽ sinh ngẫu nhiên 1 số nằm trong khoảng [0, 1] 
# (Tỷ lệ chuyển đổi kỳ vọng)
score = model.score(context=None)
print(f"Tỷ lệ chuyển đổi (mẫu ngẫu nhiên): {score * 100:.2f}%")

# Khách hàng MUA (reward = 1.0)
# Số alpha sẽ tăng lên, phân phối Beta dịch chuyển sang phải (tốt hơn)
model.update(context=None, reward=1.0)

new_score = model.score(context=None)
print(f"Tỷ lệ chuyển đổi sau khi học: {new_score * 100:.2f}%")
```

## 5. Nguồn Tham Khảo (References)

> Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). *Finite-time analysis of the multiarmed bandit problem*. Machine learning. (Chứng minh toán học nền tảng cho UCB1).
> 
> Thompson, W. R. (1933). *On the likelihood that one unknown probability exceeds another in view of the evidence of two samples*. Biometrika. (Paper gốc cách đây gần 1 thế kỷ đặt nền móng cho thuật toán Thompson).
