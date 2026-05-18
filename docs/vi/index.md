# COBA: Động Cơ Contextual Bandit Tổng Quát

**COBA** (COntextual BANdit engine) là một thư viện Contextual Bandit linh hoạt, hiệu năng cao, dành cho bài toán ra quyết định theo thời gian thực (real-time decision making).

## Triết Lý Thiết Kế Lõi

1. **Độc Lập Domain (Domain Agnostic)**: COBA xử lý hoàn toàn bằng ma trận `numpy` và toán học số thực (floating-point). Nó hoàn toàn không "biết" về cấu trúc nghiệp vụ của bạn. Điều này giúp phân tách rạch ròi giữa thuật toán và nghiệp vụ, cho phép COBA có thể được tích hợp vào bất kỳ bài toán nào.
2. **Định Tuyến Cụm (KMeans Cluster Routing)**: Thay vì đào tạo một mô hình Bandit khổng lồ cho toàn bộ không gian ngữ cảnh, COBA sử dụng `ClusterRouter`. Nó nhóm các ngữ cảnh đầu vào thành $K$ cụm hành vi phân biệt bằng KMeans và duy trì các mô hình Bandit độc lập cho mỗi cụm. Điều này giúp hệ thống học nhanh hơn và dự đoán chính xác hơn cho từng vùng dữ liệu cục bộ.
3. **Hiệu Năng Cao Cấp (High Performance)**: Được tối ưu hoàn toàn trên nền `numpy` với công thức cập nhật online Sherman-Morrison cho Linear Regression. Điều này giúp việc cập nhật mô hình có độ phức tạp chỉ $O(d^2)$ thay vì phải tính toán nghịch đảo ma trận $O(d^3)$.

## Tính Năng Nổi Bật

- **Hỗ Trợ 17 Thuật Toán (Multi-Policies)**: Bao gồm thuật toán không bối cảnh (epsilon-greedy, UCB1, Thompson, Softmax), bối cảnh tuyến tính (LinUCB, LinTS, Logistic, LinUCB-Hybrid), bối cảnh phi tuyến (Neural Linear, Random Forest, GP-UCB, Cluster Routing), và hành động liên tục (CATS). Plus các biến thể drift-aware và sliding-window.
- **Hỗ Trợ Offline Evaluation**: Đánh giá hiệu năng của policy mới dựa trên historical logs với các kỹ thuật Rejection Sampling, Doubly Robust, và NCIS.
- **Bootstrapping từ Dữ Liệu Cũ (Off-policy learning)**: Tự động hiệu chỉnh bias khi học từ dữ liệu logs cũ bằng Inverse Propensity Scoring (IPS).
- **Cơ chế Thêm/Bớt Arms Động (Dynamic Arm Management)**: Thêm bớt arms trực tiếp khi hệ thống đang chạy với cơ chế warm-start.
- **Phát Hiện Drift & Thích Ứng**: Phát hiện tự động sự dịch chuyển phân phối phần thưởng thông qua Page-Hinkley test và thích ứng nhanh chóng.
- **Model Persistence**: Lưu và tải các bandit được huấn luyện thông qua joblib để triển khai sản xuất.
- **Các Hoạt Động Batch**: Vectorized `decide_batch()` và `update_batch()` cho hệ thống thông lượng cao.

## Kiến Trúc Hệ Thống

- `coba.bandit`: Facade chính `ClusterBandit` — entrypoint duy nhất cho người dùng thư viện.
- `coba.policies`: Các thuật toán lõi (`LinUCB`, `LinTS`, `UCB1`, `Thompson Sampling`, v.v.).
- `coba.router`: Logic định tuyến KMeans.
- `coba.evaluation`: Các phương pháp đánh giá offline.
- `coba.offpolicy`: Các tiện ích Inverse Propensity Scoring (IPS).

## Các Mẫu Nâng Cao

### Tối ưu Đa Mục Tiêu (Multi-Objective via Scalarization)

Để tối ưu hóa nhiều chỉ số cùng lúc, tính composite reward trước khi gọi `update()`:

```python
w_primary   = 0.7
w_secondary = 0.3

normalized_primary   = raw_primary / max_primary
normalized_secondary = raw_secondary / max_secondary

composite_reward = (w_primary * normalized_primary) + (w_secondary * normalized_secondary)
bandit.update(context=ctx, arm=chosen_arm, reward=composite_reward)
```

Cách tiếp cận này giữ engine lõi chạy với tốc độ tối đa, trong khi Business Logic hoàn toàn linh hoạt ở tầng ứng dụng.

### Tích hợp vào Domain của bạn

Xây dựng một **Domain Facade** để dịch các đối tượng domain của bạn thành vector `numpy` thuần túy mà COBA yêu cầu. COBA không cần biết ý nghĩa của các feature hay tên của arms — ánh xạ đó hoàn toàn nằm ở tầng ứng dụng của bạn.

> Để chạy ví dụ nhanh (Quick Start), xem [README](../../README.md).
