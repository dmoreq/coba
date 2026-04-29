# COBA: Động Cơ Contextual Bandit Tổng Quát

**COBA** (COntextual BANdit engine) là một thư viện Contextual Bandit linh hoạt, hiệu năng cao, được thiết kế chuyên biệt cho các bài toán định giá động (dynamic pricing) và ra quyết định theo thời gian thực (real-time decision making).

## Triết Lý Thiết Kế Lõi

1. **Độc Lập Domain (Domain Agnostic)**: COBA xử lý hoàn toàn bằng ma trận `numpy` và toán học số thực (floating-point). Nó hoàn toàn không "biết" về cấu trúc nghiệp vụ của bạn (như H3 index, khu vực địa lý, người dùng, v.v.). Điều này giúp phân tách rạch ròi giữa thuật toán và nghiệp vụ, cho phép COBA có thể được cắm vào bất kỳ bài toán nào.
2. **Định Tuyến Cụm (KMeans Cluster Routing)**: Thay vì đào tạo một mô hình Bandit khổng lồ, phức tạp cho toàn bộ không gian ngữ cảnh, COBA sử dụng cơ chế `ClusterRouter`. Nó nhóm các ngữ cảnh đầu vào thành $K$ "chế độ thị trường" (market regimes) sử dụng KMeans (ví dụ: "Giờ cao điểm khu trung tâm", "Buổi tối ngoại ô") và duy trì các mô hình Bandit độc lập cho mỗi chế độ. Điều này giúp hệ thống học nhanh hơn và dự đoán chính xác hơn cho từng vùng dữ liệu cục bộ.
3. **Hiệu Năng Cao Cấp (High Performance)**: Được tối ưu hoàn toàn trên nền `numpy` với công thức cập nhật online Sherman-Morrison cho Linear Regression. Điều này giúp việc cập nhật mô hình có độ phức tạp chỉ $O(d^2)$ thay vì phải tính toán nghịch đảo ma trận $O(d^3)$, hoàn toàn không gây nghẽn cổ chai trong môi trường xử lý luồng (streaming environment).

## Tính Năng Nổi Bật
- **Hỗ Trợ Đa Thuật Toán (Multi-Policies)**: Bao gồm cả thuật toán theo ngữ cảnh (LinUCB, LinTS) và không theo ngữ cảnh (UCB1, Thompson Sampling).
- **Hỗ Trợ Offline Evaluation**: Đánh giá chính xác hiệu năng của policy mới dựa trên historical logs với các kỹ thuật Rejection Sampling, Doubly Robust, và NCIS.
- **Bootstrapping từ Dữ Liệu Cũ (Off-policy learning)**: Tính năng tự động hiệu chỉnh 편향 (bias) khi học từ dữ liệu logs cũ bằng Inverse Propensity Scoring (IPS).
- **Cơ chế Thêm/Bớt Cánh Tay Động (Dynamic Arm Management)**: Cho phép thêm bớt mức giá trực tiếp khi hệ thống đang chạy (online) với cơ chế warm-start.

## Kiến Trúc Hệ Thống

Thư viện được chia thành các module lõi:

- `coba.cluster_bandit`: Facade chính `ClusterBandit`. Đây là entrypoint duy nhất để module nghiệp vụ (consumer) giao tiếp với thuật toán.
- `coba.policies`: Chứa mã nguồn các thuật toán lõi (`LinUCB`, `LinTS`, `UCB1`, `Thompson Sampling`).
- `coba.routers`: Logic định tuyến, biến đổi vector ngữ cảnh nhiều chiều thành id của cluster tương ứng bằng `MiniBatchKMeans`.
- `coba.evaluation`: Các phương pháp đánh giá Offline (Rejection Sampling, Doubly Robust, NCIS).
- `coba.offpolicy`: Các tiện ích Inverse Propensity Scoring (IPS).

## Hướng Dẫn Nhanh (Quick Start)

```python
import numpy as np
from coba.cluster_bandit import ClusterBandit
from coba.types import PolicyType

# 1. Khởi tạo hệ thống Bandit
bandit = ClusterBandit(
    arms=[1.0, 1.1, 1.2, 1.5],
    n_features=7,
    policy=PolicyType.LIN_UCB,
    n_clusters=5
)

# 2. Bootstrapping từ Historical Logs (Học Offline)
bandit.fit_from_logs(
    contexts=np.random.randn(1000, 7),
    decisions=np.random.choice([1.0, 1.1, 1.2, 1.5], 1000),
    rewards=np.random.rand(1000),
    propensities=np.full(1000, 0.25) # Xác suất mà policy cũ chọn action này
)

# 3. Phục vụ trong Production (Chạy Online)
context_vector = np.array([10.5, 2.0, 50, 10, 8, 2, 5.25])
decision = bandit.decide(context_vector)
print(f"Mức giá được chọn: {decision.chosen_arm}")

# 4. Nhận phản hồi (Feedback) và cập nhật
bandit.update(
    context=context_vector,
    arm=decision.chosen_arm,
    reward=0.85
)
```

## Tích hợp vào Domain của bạn
Để sử dụng COBA trong các microservice (như FastAPI), bạn nên xây dựng một **Domain Facade** nhằm dịch các Pydantic schema (vd: `H3PricingContext`) thành vector `numpy` thuần túy mà COBA yêu cầu. (Xem ví dụ tham khảo trong module `bandit_by_location/models/bandit.py`).
