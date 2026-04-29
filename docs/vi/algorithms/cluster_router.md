# Định Tuyến Cụm (Cluster Routing)

## 1. Giải thích Trực quan (The Intuition)

Không gian ngữ cảnh hiếm khi đồng nhất. Các vùng khác nhau trong feature space có thể có hành vi reward rất khác nhau. Một mô hình tuyến tính đơn lẻ buộc phải xấp xỉ tất cả các hành vi cùng lúc — dẫn đến dự đoán trung bình hoá kém hiệu quả.

**Cluster Router** giải quyết vấn đề này theo tư duy "chia để trị": thay vì một chuyên gia đa năng, nó chia không gian ngữ cảnh thành nhiều cụm và phân công mỗi cụm cho một Bandit chuyên biệt.

## 2. Cơ chế Hoạt động

COBA sử dụng **K-Means Clustering** trên vector ngữ cảnh để tự động nhận diện các vùng hành vi phân biệt:

1. **Phân cụm (Clustering):** Khi nhận được context $x$, hệ thống tìm cụm gần nhất trong $K$ cụm đã học.
2. **Định tuyến (Routing):** Chuyển request đến Bandit độc lập quản lý cụm đó.
3. **Tính Ổn định:** Sau khi `fit()` lần đầu, cấu trúc cụm được giữ cố định. `partial_fit()` chỉ cập nhật các mô hình arm bên trong.

## 3. Quản Lý Arms Động (Dynamic Arm Management)

`ClusterRouter` hỗ trợ thay đổi arms ngay khi hệ thống đang chạy:

* **Warm Start:** Copy trọng số từ một arm đã train sang arm mới, tránh cold start.
* **Cập Nhật Nguyên Tử:** Thêm/xóa arm tự động đồng bộ xuống tất cả các cụm.

## 4. Ví dụ Cụ Thể (Code Mẫu)

```python
import numpy as np
from coba.router import ClusterRouter
from coba.types import PolicyType

# 1. Khởi tạo router
router = ClusterRouter(
    arms=["A", "B", "C"],
    n_clusters=5,
    policy=PolicyType.LIN_UCB,
    n_features=4,
    use_minibatch=True,  # khuyên dùng cho hệ thống online
    scale_contexts=True  # chuẩn hoá features trước khi phân cụm
)

# 2. Huấn luyện Offline
n_samples = 1000
contexts  = np.random.randn(n_samples, 4)
decisions = np.random.choice(["A", "B", "C"], size=n_samples)
rewards   = np.random.rand(n_samples)

router.fit(contexts, decisions, rewards)
print(f"Đã huấn luyện: {router.is_fitted}")

# 3. Dự đoán Online
ctx = np.array([2.5, 0.5, 1.0, -0.2])
chosen = router.predict(ctx)
print(f"Arm được chọn: {chosen}")

# 4. Cập nhật Online
router.update(ctx, chosen, reward=1.0)

# 5. Thêm arm mới với warm start từ "B"
router.add_arm(arm="D", warm_start_from="B")

scores = router.score_all(ctx)
print(f"Điểm cho tất cả arms: {scores}")
```

## 5. Nguồn Tham Khảo

Ý tưởng kết hợp K-Means clustering với Bandit độc lập được lấy cảm hứng từ kiến trúc của thư viện MabWiser.

> Sự kết hợp giữa K-Means (Unsupervised Learning) và MAB (Reinforcement Learning) mang lại sự cân bằng giữa sức mạnh phi tuyến tính và tốc độ inference.
