# Học Ngoại Tuyến: IPS & Doubly-Robust (Off-Policy Learning)

## 1. Giải thích Trực quan (The Intuition)

Hãy tưởng tượng bạn mới tiếp quản một nhà hàng và muốn đánh giá lại menu giá. Bạn có một tập dữ liệu 6 tháng từ ông chủ cũ, nhưng ông ta chỉ hay đề xuất 3 món A, B, C — còn món D gần như không ai biết đến.

Nếu bạn huấn luyện AI thuần tuý trên tập dữ liệu này, AI sẽ kế thừa sự thiên vị đó và **nghĩ rằng món D không ngon** chỉ vì chưa đủ dữ liệu. Đây là bẫy **Off-Policy Bias** kinh điển.

**Inverse Propensity Scoring (IPS)** là cách khử thiên vị đó: thay vì học đều từ mọi bản ghi, ta gán **trọng số ngược chiều** với xác suất mà ông chủ cũ đã chọn món đó. Món D hiếm được chọn → trọng số cao → AI học từ đó nhiều hơn để bù đắp.

> **Ứng dụng thực tế:** Khi khởi động hệ thống bandit mới từ log dữ liệu của một policy cũ (ví dụ: hệ thống cũ luôn ưu tiên chọn một arm nhất định trong hầu hết các tình huống), IPS giúp bootstrap mô hình mới mà không kế thừa sự thiên vị đó.

---

## 2. Inverse Propensity Scoring (IPS)

### Cơ chế

Với mỗi bản ghi log `(context, arm, reward, propensity)`:

$$w_i = \frac{1}{\pi_{\text{log}}(a_i | x_i)}$$

$$\tilde{r}_i = w_i \cdot r_i$$

Trong đó:
- $\pi_{\text{log}}(a_i | x_i)$: **Propensity** — xác suất mà thuật toán cũ (logging policy) đã chọn arm $a_i$ khi gặp context $x_i$.
- $w_i$: Trọng số IPS. Propensity càng thấp → trọng số càng cao.
- $\tilde{r}_i$: Reward đã được hiệu chỉnh, dùng để train mô hình mới.

### Clip để kiểm soát phương sai

Nếu propensity quá thấp (ví dụ: 0.001), trọng số $w_i$ sẽ bùng nổ lên 1000, khiến mô hình mất ổn định. `coba` dùng **clipping** để giới hạn:

```python
from coba.offpolicy import IPSConfig

config = IPSConfig(
    clip_min=1e-4,  # Propensity tối thiểu để tránh chia cho 0
    clip_max=10.0,  # Trọng số tối đa — ngăn outlier ảnh hưởng quá lớn
    use_dr=False    # Chỉ dùng IPS thuần
)
```

---

## 3. Doubly-Robust (DR) — IPS Phiên Bản Nâng Cao

### Vấn đề của IPS thuần

IPS thuần có phương sai rất cao khi:
- Propensity phân tán không đều
- Có nhiều arms ít được khám phá (rare arms)
- Tập dữ liệu nhỏ

### Cách DR giải quyết

**Doubly-Robust** kết hợp IPS với một mô hình dự đoán phần thưởng phụ trợ $\hat{r}(x, a)$:

$$\tilde{r}_i^{DR} = \hat{r}(x_i, a_i) + \frac{r_i - \hat{r}(x_i, a_i)}{\pi_{\text{log}}(a_i | x_i)}$$

**Giải thích trực quan:**
- Phần thứ nhất $\hat{r}(x_i, a_i)$: dùng mô hình để ước tính reward, **không cần propensity**.
- Phần thứ hai: hiệu chỉnh sai số của mô hình bằng IPS — phần hiệu chỉnh này nhỏ hơn nhiều so với IPS thuần, nên **phương sai thấp hơn**.

> **"Doubly-Robust"** nghĩa là ước lượng vẫn **không thiên lệch (unbiased)** nếu **ít nhất một trong hai** (mô hình phụ trợ HOẶC propensity) là chính xác.

### Kích hoạt DR trong COBA

```python
from coba.offpolicy import IPSConfig

config = IPSConfig(
    clip_min=1e-4,
    clip_max=10.0,
    use_dr=True  # Bật Doubly-Robust
)
```

---

## 4. Ví dụ Chạy Độc Lập (Code Mẫu)

### 4a. Khởi Động (Offline Bootstrap) từ Dữ Liệu Lịch Sử

```python
import numpy as np
from coba import ClusterBandit
from coba.offpolicy import DoublyRobustUpdater, IPSConfig
from coba.router import ClusterRouter
from coba.types import PolicyType

# 1. Khởi tạo router mới (chưa được train)
router = ClusterRouter(
    arms=["A", "B", "C", "D"],
    n_features=5,
    n_clusters=4,
    policy=PolicyType.LIN_TS,
)

# 2. Dữ liệu log từ hệ thống cũ — cực kỳ thiên vị về arm "A"
n_samples = 2000
contexts    = np.random.randn(n_samples, 5)
decisions   = np.random.choice(["A", "B", "C", "D"], size=n_samples, p=[0.70, 0.15, 0.10, 0.05])
rewards     = np.random.rand(n_samples)
propensities = np.array([0.70 if d == "A" else 0.15 if d == "B" else 0.10 if d == "C" else 0.05
                         for d in decisions])

# 3. Cấu hình IPS/DR
config = IPSConfig(clip_min=1e-4, clip_max=15.0, use_dr=False)

# 4. Gắn updater vào router và train offline
updater = DoublyRobustUpdater(router, config)
updater.fit_offline(
    contexts=contexts,
    decisions=decisions,
    rewards=rewards,
    propensities=propensities,
)

print(f"Router đã được khởi động (bootstrapped): {router.is_fitted}")
```

### 4b. Cập Nhật Online Hàng Ngày

```python
# Sau khi hệ thống đã live, mỗi ngày bạn nhận được log mới
new_contexts    = np.random.randn(500, 5)
new_decisions   = np.random.choice(["A", "B", "C", "D"], size=500)
new_rewards     = np.random.rand(500)
# Hệ thống mới đã ngẫu nhiên hơn → propensity gần đều
new_propensities = np.full(500, 0.25)

# update_from_logs KHÔNG chạy lại phân cụm KMeans
# — chỉ cập nhật các mô hình arm bên trong
updater.update_from_logs(
    contexts=new_contexts,
    decisions=new_decisions,
    rewards=new_rewards,
    propensities=new_propensities,
)
```

---

## 5. Khi Nào Dùng IPS vs DR?

| Tình huống | Khuyến nghị |
|---|---|
| Propensity phân tán đều, tập dữ liệu lớn | **IPS thuần** (đơn giản, nhanh) |
| Propensity skewed, nhiều rare arms | **Doubly-Robust** |
| Không có thông tin propensity | Dùng `propensities=None` — COBA giả định uniform $\frac{1}{K}$ |
| Muốn kiểm soát chặt variance | Tăng `clip_max` (giảm) hoặc dùng DR |

---

## 6. Nguồn Tham Khảo (References)

> Horvitz, D. G., & Thompson, D. J. (1952). *A generalization of sampling without replacement from a finite universe*. Journal of the American Statistical Association. (Nền tảng lý thuyết IPS gốc).
>
> Dudík, M., Langford, J., & Li, L. (2011). *Doubly robust policy evaluation and learning*. In Proceedings of the 28th International Conference on Machine Learning (ICML).
>
> Strehl, A., Langford, J., Li, L., & Kakade, S. (2010). *Learning from logged implicit exploration data*. In Advances in Neural Information Processing Systems (NeurIPS).
