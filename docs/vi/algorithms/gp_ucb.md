# GP-UCB (Gaussian Process Upper Confidence Bound)

## 1. Tổng quan

* **Loại**: Bayesian / Phi tham số
* **Class**: `GPUCBArmModel` (`policies/gp_ucb.py`)
* **Phù hợp**: Quyết định khối lượng thấp với quan sát tốn kém, bề mặt phần thưởng phi tuyến phức tạp, hoặc khi bạn có prior mạnh về độ mượt của phần thưởng.

## 2. Cách Hoạt Động

GP-UCB duy trì phân phối hậu nghiệm Gaussian Process trên hàm phần thưởng. Điểm số kết hợp trung bình hậu nghiệm (khai thác) và độ lệch chuẩn hậu nghiệm nhân `beta` (khám phá):

$$\text{score}(x) = \underbrace{\mu(x)}_{\text{khai thác}} + \underbrace{\beta \cdot \sigma(x)}_{\text{khám phá}}$$

Kernel **RBF (Radial Basis Function)**:

$$k(x, x') = \exp\!\left(-\frac{\|x - x'\|^2}{2 \ell^2}\right)$$

Trung bình và phương sai hậu nghiệm:

$$\mu(x) = k(x, X)^\top (K + \sigma_n^2 I)^{-1} y$$
$$\sigma^2(x) = k(x,x) - k(x, X)^\top (K + \sigma_n^2 I)^{-1} k(X, x)$$

Ma trận $(K + \sigma_n^2 I)$ được phân tích **Cholesky** tại mỗi lần gọi `score()` (cache, rebuild lazy).

### Độ phức tạp

| Thao tác | Chi phí |
|----------|---------|
| `update()` | O(1) — thêm quan sát |
| `score()` | O(n²) — Cholesky + triangular solve |
| Bộ nhớ | O(n²) — lưu toàn bộ ma trận kernel |

GP-UCB phù hợp nhất cho **khối lượng thấp đến trung bình**. Với throughput cao, nên dùng LinUCB hoặc LinTS.

## 3. Siêu tham số chính

* `gp_beta`: Hệ số khám phá UCB (mặc định `2.0`). Cao hơn → khám phá nhiều hơn.
* `gp_length_scale`: Bandwidth kernel RBF `ℓ` (mặc định `1.0`). Nhỏ hơn → phân biệt context gần nhau. Lớn hơn → tổng quát hóa mượt hơn.
* `gp_noise_var`: Phương sai nhiễu quan sát `σ_n²` (mặc định `0.1`). Tương tự L2 regularization.
* `gp_max_obs`: Số quan sát tối đa mỗi arm (mặc định `500`). Quan sát cũ nhất bị xóa (FIFO) khi vượt quá.

## 4. Ví dụ

```python
import numpy as np
from coba import ClusterBandit
from coba.types import PolicyType

bandit = ClusterBandit(
    arms=["A", "B", "C"],
    n_features=5,
    policy=PolicyType.GP_UCB,
    n_clusters=3,
    gp_beta=2.0,
    gp_length_scale=1.0,
    gp_noise_var=0.1,
    gp_max_obs=200,
)
```

## 5. Khi nào dùng GP-UCB

| Tình huống | Khuyến nghị |
|-----------|------------|
| < 1k quyết định/arm, phần thưởng phi tuyến | **GP-UCB** |
| Throughput cao (> 10k/s), phần thưởng tuyến tính | LinUCB hoặc LinTS |
| Phi tuyến, khối lượng lớn | BootstrappedTS |

## 6. Tài liệu tham khảo

Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). *Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design*. ICML 2010.
