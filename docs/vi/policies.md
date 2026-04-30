# Thư viện Thuật Toán (Policies Reference)

Hệ thống `coba` cung cấp một tập hợp các thuật toán Multi-Armed Bandit (MAB) từ cơ bản đến nâng cao.

Để giúp bạn hiểu rõ bản chất cốt lõi của từng giải thuật, chi tiết giải thích (The Intuition), toán học cơ sở, tài liệu tham khảo khoa học, và code mẫu được tách vào từng bài viết riêng biệt dưới đây.

## Các Thuật Toán Theo Ngữ Cảnh (Contextual Bandits)

Đây là nhóm thuật toán chủ lực. Chúng phân tích feature vector ở mỗi bước quyết định để dự đoán arm tối ưu nhất.

1. **[LinUCB (Linear Upper Confidence Bound)](algorithms/linucb.md)**
   Thuật toán tất định (Deterministic) sử dụng "khoảng tin cậy" để khuyến khích khám phá trong các vùng ngữ cảnh chưa quen.

2. **[LinTS (Linear Thompson Sampling)](algorithms/lin_ts.md)**
   Thuật toán ngẫu nhiên (Stochastic) sử dụng phân phối Bayesian. Rất phù hợp cho hệ thống có độ trễ phản hồi (delayed feedback) và batch-updates.

3. **[Logistic Bandits (Laplace Approximation)](algorithms/logistic.md)**
   Được thiết kế riêng để xử lý dữ liệu nhị phân. Lựa chọn tốt nhất khi reward là kết quả 0/1 (ví dụ: chuyển đổi, click).

4. **[Meta-Heuristics (Scikit-learn Wrappers)](algorithms/sklearn_meta.md)**
   Bọc các mô hình học máy như LightGBM hoặc Random Forest vào cơ chế Bandit (Bootstrapped, Epsilon-Greedy) để xử lý các mối quan hệ phi tuyến tính phức tạp.

---

## Các Thuật Toán Cơ Bản (Context-Free)

Nhóm này không sử dụng feature vector, chỉ dựa vào thống kê tổng hợp của từng arm. Đây là sự nâng cấp thông minh thay thế cho A/B testing truyền thống.

5. **[UCB1 & Thompson Sampling](algorithms/context_free.md)**
   Tự động dồn traffic vào arm hiệu quả nhất khi bằng chứng tích lũy.

---

## Chủ Đề Nâng Cao

1. **[Định Tuyến Cụm (Cluster Routing)](algorithms/cluster_router.md)**
   Xử lý không gian ngữ cảnh phi tuyến bằng cách tự động phân chia thành các cụm, mỗi cụm quản lý một Bandit độc lập. Hỗ trợ thêm/xóa arms lúc runtime với warm-start.

2. **[Học Ngoại Tuyến (Off-Policy Learning)](algorithms/offpolicy_ips.md)**
   Khởi động mô hình Bandit mới từ dữ liệu lịch sử bị thiên vị. COBA sử dụng IPS và Doubly-Robust để khử nhiễu.

3. **Tối ưu Đa Mục Tiêu (Multi-Objective via Scalarization)** — xem [Kiến Trúc & Mẫu](index.md#tối-ưu-đa-mục-tiêu-multi-objective-via-scalarization).
