# LinUCB-Hybrid (Giới hạn Tin cậy Tuyến tính Kết hợp)

## 1. Tổng quan

* **Loại**: Tất định / Lạc quan trước sự không chắc chắn (UCB)
* **Class**: `LinUCBHybridArmModel` (`policies/lin_ucb_hybrid.py`)
* **Phù hợp nhất**: Khi context chứa cả **đặc trưng dùng chung** (demographics người dùng, tín hiệu phiên) lẫn **đặc trưng riêng của từng arm** (embedding item, thuộc tính nội dung).

## 2. Cách hoạt động

Vector context đầy đủ `x` (dài `n_features = n_shared + n_arm`) được tách thành hai phần:

* `z = x[:n_shared]` — đặc trưng dùng chung cho tất cả arm (tuổi người dùng, thời điểm trong ngày)
* `x_arm = x[n_shared:]` — đặc trưng riêng của arm (embedding danh mục item)

**Một instance `SharedRidge`** được duy trì cho mỗi cluster và được cập nhật **mỗi lần** bất kỳ arm nào được kéo. Mỗi arm còn giữ một `RidgeRegression` riêng chỉ học từ dữ liệu của arm đó.

Điểm số kết hợp phần khai thác và phần thưởng khám phá UCB từ cả hai thành phần:

$$\text{score}(z, x_{\text{arm}}) = \underbrace{z^\top \hat{\beta}_{\text{shared}} + x_{\text{arm}}^\top \hat{\theta}_{\text{arm}}}_{\text{khai thác}} + \underbrace{\alpha \sqrt{z^\top A_0^{-1} z + x_{\text{arm}}^\top A_{\text{arm}}^{-1} x_{\text{arm}}}}_{\text{khám phá}}$$

**Đặc tính chính**: vì mọi lần kéo arm đều cập nhật `β_shared`, các đặc trưng dùng chung hội tụ nhanh hơn nhiều so với đặc trưng riêng — bạn nhận được học chuyển giao giữa các arm miễn phí trên chiều dùng chung.

> Đây là xấp xỉ của UCB hybrid đầy đủ từ Li et al. (dạng chính xác có thêm số hạng hiệp phương sai chéo). Xấp xỉ này giữ lại lợi ích thực tế chính trong khi duy trì độ phức tạp O(d²).

### Độ phức tạp

| Thao tác | Chi phí |
|----------|---------|
| `update()` | O(d_shared²) + O(d_arm²) — hai cập nhật Sherman-Morrison |
| `score()` | O(d_shared²) + O(d_arm²) — hai dạng toàn phương |
| Bộ nhớ | O(d_shared²) dùng chung + O(d_arm² × n_arms) riêng từng arm |

## 3. Các siêu tham số quan trọng

* `n_shared_features` (đặt trên `ClusterBandit`): số chiều context dùng chung — **`n_shared_features` phần tử đầu tiên** của mỗi vector context. Phải ≥ 0. Mặc định `0` (thuần per-arm, giống LinUCB tiêu chuẩn).
* `alpha`: độ rộng khám phá UCB (mặc định `1.0`). Cao hơn → khoảng tin cậy rộng hơn → nhiều khám phá hơn. Khoảng thông thường: 0.5–2.0.
* `l2_lambda`: điều chuẩn L2 cho ridge riêng từng arm (mặc định `1.0`).
* `gamma`: hệ số chiết khấu cho phi tĩnh (mặc định `1.0`). Đặt `< 1.0` (ví dụ `0.99`) để quên dần quan sát cũ.

## 4. Ví dụ

```python
import numpy as np
from coba import ClusterBandit
from coba.types import PolicyType

# Bố cục context: [tuổi_user, thời_gian_phiên, thời_điểm, id_thành_phố,  ← dùng chung (4 chiều)
#                  danh_mục_item, giá_item, độ_phổ_biến_item, ...]          ← riêng arm (6 chiều)

bandit = ClusterBandit(
    arms=["bài_A", "bài_B", "bài_C"],
    n_features=10,            # tổng độ dài context
    n_shared_features=4,      # 4 chiều đầu học chung giữa tất cả arm
    policy=PolicyType.LIN_UCB_HYBRID,
    alpha=1.0,
    l2_lambda=1.0,
    n_clusters=3,
)

context = np.random.randn(10)
decision = bandit.decide(context)
print(decision.chosen_arm, decision.score)
```

## 5. Khi nào nên sử dụng

| Tình huống | Khuyến nghị |
|------------|-------------|
| Context có chiều chung (user) + chiều riêng (item) rõ ràng | **LinUCB-Hybrid** |
| Toàn bộ context đặc trưng riêng, không kỳ vọng học chuyển giao | LinUCB |
| Cần khám phá ngẫu nhiên hoặc xử lý feedback trễ | LinTS |
| Bề mặt phần thưởng phi tuyến, lượng dữ liệu vừa | NeuralLinear hoặc BootstrappedTS |
| Lượng dữ liệu rất thấp, phi tuyến phức tạp | GP-UCB |

## 6. Tài liệu tham khảo

Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). *A Contextual-Bandit Approach to Personalized News Article Recommendation*. WWW 2010.
