# Định Tuyến Cụm (Cluster Routing)

## 1. Giải thích Trực quan (The Intuition)
Trong định giá động (dynamic pricing) cho gọi xe hoặc giao hàng, hành vi của khách hàng và thị trường không bao giờ là một đường thẳng đơn giản. Ví dụ: Khu vực trung tâm thương mại đông đúc vào giờ tan tầm có độ nhạy cảm về giá (price elasticity) hoàn toàn trái ngược với khu vực ngoại ô vắng vẻ vào lúc nửa đêm.

Nếu bạn cố gắng dùng một mô hình tuyến tính đơn lẻ (như LinUCB hay LinTS) để "ôm" hết toàn bộ dữ liệu này, mô hình sẽ bị quá tải, dự đoán trung bình và hoạt động kém hiệu quả.

**Cluster Router** giải quyết vấn đề này theo tư duy: *"Chia để trị"*. 
Thay vì dùng 1 chuyên gia đa năng, nó chia thị trường ra làm nhiều khu vực/chế độ (Clusters) và phân công mỗi khu vực cho 1 chuyên gia Bandit riêng biệt chăm sóc.

## 2. Cơ chế Hoạt động

COBA sử dụng thuật toán **K-Means Clustering** trên các vector ngữ cảnh (context vectors) để tự động nhận diện các "chế độ" của thị trường:

1. **Phân cụm (Clustering):** Khi nhận được một request với context $x$, hệ thống đo khoảng cách từ $x$ tới tâm của $K$ cụm. Nó xếp request này vào cụm gần nhất (ví dụ: Cụm "Giờ cao điểm mưa").
2. **Định tuyến (Routing):** Khi đã xác định được cụm, `ClusterRouter` sẽ "chuyển phát" request đó đến bộ giải thuật Bandit độc lập được quản lý bởi riêng cụm đó.
3. **Tính Ổn định:** COBA sử dụng `MiniBatchKMeans`. Trong môi trường Online Learning, tâm của các cụm sẽ dần dịch chuyển linh hoạt theo sự biến đổi của thị trường mà không cần phải train lại toàn bộ từ đầu.

## 3. Quản Lý Nhánh Động (Dynamic Arm Management)

Trong thực tế kinh doanh, bạn không thể giữ cố định các mức giá mãi mãi. Hôm nay bạn chạy các mức `[1.0, 1.2, 1.5]`, nhưng ngày mai bạn muốn thử nghiệm thêm mức `1.3`.

`ClusterRouter` hỗ trợ tính năng thay đổi "tay đòn" (arms) ngay trong lúc hệ thống đang chạy (Runtime):

* **Khởi Động Ấm (Warm Start):** Nếu bạn thêm mức giá `1.3` như một nhánh mới tinh, nó sẽ bắt đầu từ con số 0 (Cold Start) và hoạt động rất ngẫu nhiên, dễ làm hỏng trải nghiệm user. `ClusterRouter` cho phép bạn "copy" toàn bộ trọng số não bộ của mức giá `1.2` sang cho `1.3` làm nền tảng ban đầu.
* **Cập Nhật Nguyên Tử:** Việc thêm/xoá tay đòn sẽ tự động được đồng bộ mượt mà xuống hàng chục cụm Bandit bên dưới mà không làm sập hệ thống.

## 4. Ví dụ Cụ Thể (Code Mẫu)

Dưới đây là ví dụ từ lúc khởi tạo đến lúc thêm một mức giá mới trên môi trường production.

```python
import numpy as np
from coba.routers.cluster_router import ClusterRouter
from coba.types import PolicyType

# 1. Khởi tạo router với 5 chế độ thị trường (clusters)
router = ClusterRouter(
    arms=[1.0, 1.2, 1.5],
    n_clusters=5,
    policy=PolicyType.LIN_UCB,
    n_features=4,
    use_minibatch=True, # Khuyên dùng cho hệ thống online
    scale_contexts=True # Chuẩn hoá (Standardize) features trước khi phân cụm
)

# 2. Huấn luyện Offline (Batch Fit) từ historical logs
n_samples = 1000
contexts = np.random.randn(n_samples, 4)
decisions = np.random.choice([1.0, 1.2, 1.5], size=n_samples)
rewards = np.random.rand(n_samples)

router.fit(contexts, decisions, rewards)
print(f"Đã huấn luyện: {router.is_fitted}")

# 3. Dự đoán Online (Khi có request mới)
new_context = np.array([2.5, 0.5, 1.0, -0.2])

# Bước này ngầm định: Tìm cụm gần nhất -> Đưa context cho LinUCB của cụm đó -> Lấy ra tay đòn
chosen_arm = router.predict(new_context)
print(f"Mức giá được chọn: {chosen_arm}")

# 4. Nhận phản hồi và Cập nhật (Online Update)
# Khách hàng chấp nhận mức giá (reward = 1.0)
router.update(new_context, chosen_arm, reward=1.0)

# 5. Quản Lý Nhánh Động (Warm Start)
# Thêm mức giá mới 1.3x.
# Copy trọng số của nhánh 1.2x để nhánh 1.3x không bị học lại từ đầu (tránh ngẫu nhiên quá lố).
router.add_arm(arm=1.3, warm_start_from=1.2)

# Hệ thống tự động nhận diện và tính điểm cho nhánh 1.3 ở request tiếp theo
scores = router.score_all(new_context)
print(f"Điểm cho tất cả các nhánh: {scores}")
```

## 5. Nguồn Tham Khảo (References)

Ý tưởng chia thị trường bằng phân cụm (Clustering) kết hợp với Bandit độc lập được truyền cảm hứng từ kiến trúc của thư viện MabWiser và các hệ thống Production thực tế.

> Sự kết hợp giữa K-Means (Unsupervised Learning) và MAB (Reinforcement Learning) là một biến thể của mô hình Contextual Bandits, mang lại sự cân bằng hoàn hảo giữa sức mạnh dự đoán phi tuyến tính và tốc độ tính toán siêu tốc ở tầng Inference.
