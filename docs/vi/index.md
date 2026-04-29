# COBA: Động Cơ Contextual Bandit Tổng Quát

**COBA** (COntextual BANdit engine) là một thư viện Contextual Bandit linh hoạt, hiệu năng cao, dành cho bài toán ra quyết định theo thời gian thực (real-time decision making).

## Triết Lý Thiết Kế Lõi

1. **Độc Lập Domain (Domain Agnostic)**: COBA xử lý hoàn toàn bằng ma trận `numpy` và toán học số thực (floating-point). Nó hoàn toàn không "biết" về cấu trúc nghiệp vụ của bạn. Điều này giúp phân tách rạch ròi giữa thuật toán và nghiệp vụ, cho phép COBA có thể được tích hợp vào bất kỳ bài toán nào.
2. **Định Tuyến Cụm (KMeans Cluster Routing)**: Thay vì đào tạo một mô hình Bandit khổng lồ cho toàn bộ không gian ngữ cảnh, COBA sử dụng `ClusterRouter`. Nó nhóm các ngữ cảnh đầu vào thành $K$ cụm hành vi phân biệt bằng KMeans và duy trì các mô hình Bandit độc lập cho mỗi cụm. Điều này giúp hệ thống học nhanh hơn và dự đoán chính xác hơn cho từng vùng dữ liệu cục bộ.
3. **Hiệu Năng Cao Cấp (High Performance)**: Được tối ưu hoàn toàn trên nền `numpy` với công thức cập nhật online Sherman-Morrison cho Linear Regression. Điều này giúp việc cập nhật mô hình có độ phức tạp chỉ $O(d^2)$ thay vì phải tính toán nghịch đảo ma trận $O(d^3)$.

## Tính Năng Nổi Bật

- **Hỗ Trợ Đa Thuật Toán (Multi-Policies)**: Bao gồm cả thuật toán theo ngữ cảnh (LinUCB, LinTS, Logistic) và không theo ngữ cảnh (UCB1, Thompson Sampling).
- **Hỗ Trợ Offline Evaluation**: Đánh giá hiệu năng của policy mới dựa trên historical logs với các kỹ thuật Rejection Sampling, Doubly Robust, và NCIS.
- **Bootstrapping từ Dữ Liệu Cũ (Off-policy learning)**: Tự động hiệu chỉnh bias khi học từ dữ liệu logs cũ bằng Inverse Propensity Scoring (IPS).
- **Cơ chế Thêm/Bớt Arms Động (Dynamic Arm Management)**: Thêm bớt arms trực tiếp khi hệ thống đang chạy với cơ chế warm-start.

## Kiến Trúc Hệ Thống

- `coba.bandit`: Facade chính `ClusterBandit` — entrypoint duy nhất cho người dùng thư viện.
- `coba.policies`: Các thuật toán lõi (`LinUCB`, `LinTS`, `UCB1`, `Thompson Sampling`, v.v.).
- `coba.router`: Logic định tuyến KMeans.
- `coba.evaluation`: Các phương pháp đánh giá offline.
- `coba.offpolicy`: Các tiện ích Inverse Propensity Scoring (IPS).

## Hướng Dẫn Nhanh (Quick Start)

```python
import numpy as np
from coba import ClusterBandit
from coba.types import PolicyType

# 1. Khởi tạo hệ thống Bandit
bandit = ClusterBandit(
    arms=["A", "B", "C", "D"],
    n_features=5,
    policy=PolicyType.LIN_UCB,
    n_clusters=3
)

# 2. Bootstrapping từ Historical Logs (Học Offline)
bandit.fit_from_logs(
    contexts=np.random.randn(1000, 5),
    decisions=np.random.choice(["A", "B", "C", "D"], 1000),
    rewards=np.random.rand(1000),
    propensities=np.full(1000, 0.25)
)

# 3. Ra quyết định trực tuyến (Online)
ctx = np.array([0.5, -1.2, 0.3, 2.1, -0.8])
decision = bandit.decide(ctx)
print(f"Arm được chọn: {decision.chosen_arm}")

# 4. Nhận phản hồi và cập nhật
bandit.update(context=ctx, arm=decision.chosen_arm, reward=0.85)
```

## Tích hợp vào Domain của bạn

Xây dựng một **Domain Facade** để dịch các đối tượng domain của bạn thành vector `numpy` thuần túy mà COBA yêu cầu. COBA không cần biết ý nghĩa của các feature hay tên của arms — ánh xạ đó hoàn toàn nằm ở tầng ứng dụng của bạn.
