# NeuralLinear (Mạng Nơ-ron + Thompson Sampling Tuyến tính)

## 1. Tổng quan

* **Loại**: Phi tuyến (MLP) + Bayesian tuyến tính (LinTS)
* **Class**: `NeuralLinearArmModel` (`policies/neural_linear.py`)
* **Phù hợp nhất**: Bề mặt phần thưởng phi tuyến khi bạn vẫn muốn khám phá Bayesian được hiệu chỉnh — mà không cần GPU.

## 2. Cách hoạt động

NeuralLinear là kiến trúc hai tầng: một MLP dùng chung trích xuất embedding phi tuyến, và mỗi arm chạy một LinTS head riêng trên các embedding đó.

### Tầng 1 — MLP Backbone dùng chung (`NeuralLinearBackbone`)

Một backbone được duy trì **cho mỗi cluster** (không phải mỗi arm). Đây là `MLPRegressor` của sklearn với kiến trúc:

$$\text{layers} = \text{hidden\_sizes} + (\text{embedding\_dim},)$$

Backbone duy trì một **replay buffer dùng chung** (FIFO, tối đa `10 000` entry) cho tất cả arm trong cluster, lưu tuple `(x, arm, reward, weight)`.

Mỗi `retrain_freq` updates tổng cộng, backbone refit lại MLP trên toàn bộ buffer. **Kích hoạt layer penultimate** (trước output regression head, sử dụng ReLU) được dùng làm embedding:

$$\phi(x) = \text{ReLU}(W_{L-1} \cdots \text{ReLU}(W_1 x + b_1) \cdots + b_{L-1}) \in \mathbb{R}^{d_{\text{emb}}}$$

### Tầng 2 — LinTS Head riêng từng arm

Mỗi arm duy trì một `LinTSArmModel` trên không gian embedding $\mathbb{R}^{d_{\text{emb}}}$. Sau mỗi lần backbone retrain, **toàn bộ LinTS head được rebuild từ đầu** bằng cách replay dữ liệu đã buffer qua embedding mới — các cập nhật tăng dần bị loại bỏ.

Giữa các lần retrain, quan sát mới cập nhật LinTS head tăng dần qua Sherman-Morrison.

### Cold Start

Cho đến khi backbone được fitted, `score()` trả về `float("inf")` để mỗi arm được khám phá ít nhất một lần trước khi học bắt đầu.

### Độ phức tạp

| Thao tác | Chi phí |
|----------|---------|
| `update()` | O(1) amortized (append buffer); O(n·d_emb²) mỗi lần retrain |
| `score()` | O(d_emb²) — LinTS sample trên embedding |
| Bộ nhớ | O(buffer\_maxlen × n\_features) cho buffer |

## 3. Các siêu tham số quan trọng

* `neural_embedding_dim`: Số chiều của layer penultimate MLP — đây là kích thước đầu vào LinTS (mặc định `16`). Lớn hơn → biểu diễn phong phú hơn nhưng cập nhật LinTS chậm hơn.
* `neural_hidden_sizes`: Tuple chiều rộng hidden layer (mặc định `(64, 32)`). Layer đầu ra `embedding_dim` được thêm tự động. Ví dụ: `(64, 32)` tạo ra layers 64 → 32 → 16 (với `embedding_dim=16`).
* `neural_retrain_freq`: Tổng số updates giữa các lần retrain backbone (mặc định `200`). Thấp hơn → backbone thích nghi nhanh hơn nhưng tốn CPU hơn. Cao hơn → serving nhanh hơn nhưng backbone bị lag.
* `v_sq`: Hệ số nhân phương sai posterior cho LinTS head (mặc định `1.0`). Cao hơn → nhiều khám phá hơn.
* `l2_lambda`: Điều chuẩn L2 cho cả MLP (sklearn `alpha`) và LinTS riêng từng arm (mặc định `1.0`).
* `gamma`: Hệ số chiết khấu cho phi tĩnh (mặc định `1.0`). Áp dụng cho LinTS head từng arm.

## 4. Ví dụ

```python
from coba import ClusterBandit
from coba.types import PolicyType

bandit = ClusterBandit(
    arms=["A", "B", "C"],
    n_features=20,
    policy=PolicyType.NEURAL_LINEAR,
    neural_embedding_dim=16,       # kích thước layer penultimate
    neural_hidden_sizes=(64, 32),  # layers thực tế: 64 → 32 → 16
    neural_retrain_freq=100,       # retrain backbone mỗi 100 updates
    v_sq=1.0,
    l2_lambda=1.0,
    n_clusters=3,
)

import numpy as np
context = np.random.randn(20)

# ~100 updates đầu tiên: backbone chưa fitted, score() = inf → thuần khám phá
bandit.fit(contexts, arms_chosen, rewards)
decision = bandit.decide(context)
print(decision.chosen_arm, decision.score)
```

## 5. Khi nào nên sử dụng

| Tình huống | Khuyến nghị |
|------------|-------------|
| Phần thưởng phi tuyến, cần khám phá Bayesian, không có GPU | **NeuralLinear** |
| Phần thưởng tuyến tính, throughput cao (> 10k/s) | LinUCB hoặc LinTS |
| Phi tuyến, linh hoạt sklearn, không yêu cầu Bayesian chặt | BootstrappedTS |
| Lượng dữ liệu rất thấp (< 1k obs/arm), phi tuyến phức tạp | GP-UCB |
| Lượng lớn + có GPU | Thư viện deep RL bên ngoài |

**Lưu ý**: NeuralLinear retrain backbone định kỳ — mỗi lần retrain là O(n) MLP fit trên toàn buffer. Với hệ thống throughput rất cao, đặt `neural_retrain_freq` cao (≥ 500) hoặc ưu tiên LinUCB/LinTS.

## 6. Tài liệu tham khảo

Riquelme, C., Tucker, G., & Snoek, J. (2018). *Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling*. ICLR 2018.
