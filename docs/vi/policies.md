# Thư viện Thuật Toán (Policies Reference)

Hệ thống `coba` cung cấp một tập hợp các thuật toán Multi-Armed Bandit (MAB) từ cơ bản đến nâng cao. Mỗi thuật toán được thiết kế để giải quyết những đặc thù riêng biệt của bài toán Dynamic Pricing (Định giá Động).

Để giúp bạn hiểu rõ bản chất cốt lõi của từng giải thuật, chúng tôi đã tách chi tiết giải thích (The Intuition), toán học cơ sở, tài liệu tham khảo khoa học, và **Code Mẫu (Chạy Độc Lập)** vào từng bài viết riêng biệt dưới đây.

## Các Thuật Toán Theo Ngữ Cảnh (Contextual Bandits)

Đây là nhóm thuật toán chủ lực. Chúng phân tích thông tin môi trường hiện tại (ví dụ: thời tiết, lượng tài xế rảnh, giờ cao điểm) để dự đoán mức giá tối ưu nhất.

1. **[LinUCB (Linear Upper Confidence Bound)](algorithms/linucb.md)**
   Thuật toán tất định (Deterministic) sử dụng "Khoảng đệm an toàn" để kích thích sự tò mò trong môi trường mới.

2. **[LinTS (Linear Thompson Sampling)](algorithms/lints.md)**
   Thuật toán ngẫu nhiên (Stochastic) sử dụng phân phối Bayesian. Rất phù hợp cho hệ thống có độ trễ phản hồi (delayed feedback) và batch-updates.

3. **[Logistic Bandits (Laplace Approximation)](algorithms/logistic.md)**
   Được thiết kế riêng biệt để xử lý dữ liệu Nhị phân (Binary). Lựa chọn số 1 nếu mục tiêu của bạn là tối ưu **Tỷ lệ chuyển đổi (Conversion Rate)**.

4. **[Meta-Heuristics (Tree-based & Scikit-learn Wrappers)](algorithms/sklearn_meta.md)**
   Bọc các mô hình học máy cực mạnh như **LightGBM** hoặc **Random Forest** vào các cơ chế Bandit (Bootstrapped, Epsilon Greedy) để giải quyết các mối quan hệ phi tuyến tính phức tạp.

---

## Các Thuật Toán Cơ Bản (Context-Free)

Nhóm này không quan tâm đến bối cảnh khách hàng, chỉ dựa thuần tuý vào dữ liệu click lịch sử. Chúng là sự nâng cấp hoàn hảo thay thế cho phương pháp **A/B Testing truyền thống**.

5. **[UCB1 & Thompson Sampling](algorithms/context_free.md)**
   Giải pháp thông minh, tự động dồn traffic (lượt hiển thị) vào mức giá hoặc tay đòn có tỷ lệ chuyển đổi cao nhất một cách nhanh chóng.

---

## Chủ Đề Nâng Cao

Để đưa hệ thống Bandit từ môi trường Lab ra thực tế (Production), bạn sẽ cần giải quyết 3 bài toán lớn:

1. **[Định Tuyến Cụm (Cluster Routing)](algorithms/cluster_router.md)**
   Giải quyết bài toán phi tuyến tính (non-linear) và thị trường không ổn định bằng cách tự động chia khách hàng thành các cụm nhỏ, mỗi cụm quản lý một Bandit độc lập. Đồng thời hỗ trợ Thêm/Sửa/Xoá mức giá (arms) ngay lập tức lúc hệ thống đang chạy.

2. **[Học Ngoại Tuyến (Off-Policy Learning)](algorithms/offpolicy_ips.md)**
   Khởi động mô hình Bandit mới từ dữ liệu lịch sử bị thiên vị do hệ thống cũ sinh ra. COBA sử dụng **Inverse Propensity Scoring (IPS)** và **Doubly-Robust (DR)** để khử nhiễu.

3. **Tối ưu Đa Mục Tiêu (Multi-Objective Contextual Bandits)**
   Nếu bạn cần tối ưu hóa nhiều chỉ số kinh doanh cùng lúc (ví dụ: vừa muốn tăng Doanh thu/GMV, vừa muốn giữ Tỷ lệ chuyển đổi cao), cách tốt nhất và nhanh nhất trong `coba` là **Scalarization (Tích hợp Reward)**.

Thay vì viết một policy đa mục tiêu phức tạp làm chậm hệ thống, bạn chỉ cần định nghĩa một hàm tính Reward tổng hợp trước khi gọi hàm `update()`:

```python
# Trọng số do Business quyết định
w_gmv = 0.7
w_cr = 0.3

# Chuẩn hoá (Normalize) các chỉ số
normalized_gmv = actual_gmv / max_expected_gmv
is_converted = 1.0 if user_booked else 0.0

# Gom lại thành 1 reward duy nhất
composite_reward = (w_gmv * normalized_gmv) + (w_cr * is_converted)

# Cập nhật thuật toán
policy.update(context, arm_chosen, reward=composite_reward)
```
Cách tiếp cận này giúp giữ engine lõi chạy với tốc độ tối đa, trong khi tầng Business Logic bên ngoài hoàn toàn linh hoạt.
