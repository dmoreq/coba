# Tham Chiếu Policies (Thuật Toán Bandits)

Tất cả 17 loại policy được COBA hỗ trợ, được ánh xạ đến các nhóm thuật toán tương ứng.

---

## Các Policy Không Dùng Ngữ Cảnh (Context-Free)

Các policy này bỏ qua context và tối ưu hóa chỉ dựa trên thống kê phần thưởng của arm.

### 1. Epsilon-Greedy
**Bài Học:** Bài 0 (Explore vs Exploit)
**Loại:** Tất định
**Trực Quan:** Chọn arm tốt nhất với xác suất `1-ε`, khám phá ngẫu nhiên với xác suất `ε`
**Khi Sử Dụng:** Chiến lược cơ sở đơn giản nhất, tỷ lệ khám phá dự đoán được
**Tham Chiếu:** [docs/algorithms/context_free.md](./algorithms/context_free.md)

### 2. UCB1 (Upper Confidence Bound)
**Bài Học:** Bài 1 (UCB1 Landing Page Testing)
**Loại:** Tất định
**Trực Quan:** Thêm bonus khám phá mà thu nhỏ khi arm được kéo nhiều hơn
**Khi Sử Dụng:** Khám phá mịn giảm dần, cục bộ Regret Optimal
**Công Thức:** `score = mean_reward + alpha * sqrt(ln(N) / n_pulls)`
**Tham Chiếu:** [docs/algorithms/context_free.md](./algorithms/context_free.md)

### 3. Thompson Sampling
**Bài Học:** Bài 2 (Thompson Sampling Email Subject Lines)
**Loại:** Stochastic (Bayesian)
**Trực Quan:** Lấy mẫu từ posterior Beta cho mỗi arm, chọn arm có mẫu cao nhất
**Khi Sử Dụng:** Kết quả nhị phân (chuyển đổi), hiệu suất thực nghiệm tốt
**Ràng Buộc:** Phần thưởng phải trong [0, 1]
**Tham Chiếu:** [docs/algorithms/context_free.md](./algorithms/context_free.md)

### 4. Softmax
**Bài Học:** Bài 11 (Softmax Playlist Generation)
**Loại:** Stochastic
**Trực Quan:** Lấy mẫu arm với xác suất tỉ lệ với `exp(score / tau)`; nhiệt độ `tau` kiểm soát khám phá
**Khi Sử Dụng:** Khám phá ngẫu nhiên mịn, có thể điều chỉnh thông qua nhiệt độ
**Tham Chiếu:** [docs/algorithms/softmax.md](./algorithms/softmax.md)

---

## Các Policy Bối Cảnh Tuyến Tính

Những policy này học một mô hình tuyến tính cho mỗi arm, sử dụng các tính năng ngữ cảnh.

### 5. LinUCB (Linear Upper Confidence Bound)
**Bài Học:** Bài 3 (LinUCB Product Recommendation)
**Loại:** Tất định
**Trực Quan:** Fit ridge regression cho mỗi arm; thêm bonus khám phá tỉ lệ với độ không chắc chắn
**Khi Sử Dụng:** Bandits bối cảnh, bề mặt phần thưởng tuyến tính, đặc trưng có thể giải thích
**Công Thức:** `score = x^T beta + alpha * sqrt(x^T A_inv x)`
**Tham Chiếu:** [docs/algorithms/linucb.md](./algorithms/linucb.md)

### 6. LinTS (Linear Thompson Sampling)
**Bài Học:** Bài 4 (LinTS Loan Offer Personalisation)
**Loại:** Stochastic (Bayesian)
**Trực Quan:** Duy trì posterior Bayesian trên hệ số arm, lấy mẫu và chọn tốt nhất
**Khi Sử Dụng:** Bối cảnh, phản hồi trễ, cập nhật batch
**Tham Chiếu:** [docs/algorithms/lin_ts.md](./algorithms/lin_ts.md)

### 7. Logistic Bandits (UCB)
**Bài Học:** Bài 5 (Logistic Bandits for Ad CTR)
**Loại:** Tất định
**Trực Quan:** Mô hình kết quả nhị phân (0/1) với hồi quy logistic cho mỗi arm; khám phá UCB
**Khi Sử Dụng:** Tỷ lệ click-through, chuyển đổi, phần thưởng nhị phân
**Ràng Buộc:** Phần thưởng phải là 0 hoặc 1
**Tham Chiếu:** [docs/algorithms/logistic.md](./algorithms/logistic.md)

### 8. Logistic Thompson Sampling (LogisticTS)
**Bài Học:** Không có bài học chuyên dụng (có sẵn policy)
**Loại:** Stochastic
**Trực Quan:** Hồi quy logistic Bayesian; lấy mẫu và chọn arm tốt nhất
**Khi Sử Dụng:** Kết quả nhị phân với định lượng độ không chắc chắn Bayesian
**Ràng Buộc:** Phần thưởng phải là 0 hoặc 1

---

## Các Policy Bối Cảnh Phi Tuyến

### 9. LinUCB-Hybrid
**Bài Học:** Bài 7 (LinUCB-Hybrid News Personalisation)
**Loại:** Tất định
**Trực Quan:** Chia tách bối cảnh thành tính năng chung (học chung) + tính năng riêng arm
**Khi Sử Dụng:** Tính năng chung được chia sẻ quy mô trên các arm
**Tham Chiếu:** [docs/algorithms/lin_ucb_hybrid.md](./algorithms/lin_ucb_hybrid.md)

### 10. Cluster Routing
**Bài Học:** Bài 6 (KMeans Cluster Routing Music Demo)
**Loại:** Tất định (phân cụm)
**Trực Quan:** Phân chia không gian bối cảnh qua K-means; duy trì LinUCB độc lập mỗi cụm
**Khi Sử Dụng:** Môi trường phi tuyến/không dừa, các phân đoạn người dùng không đồng nhất
**Tham Chiếu:** [docs/algorithms/cluster_router.md](./algorithms/cluster_router.md)

### 11. Neural Linear
**Bài Học:** Bài 8 (NeuralLinear Video Recommendation)
**Loại:** Hybrid (Deep + Linear)
**Trực Quan:** Xương sống MLP chung trích xuất nhúng phi tuyến; các đầu LinTS riêng arm học phía trên
**Khi Sử Dụng:** Bề mặt phần thưởng phi tuyến phức tạp, không muốn full deep RL overhead
**Tham Chiếu:** [docs/algorithms/neural_linear.md](./algorithms/neural_linear.md)

### 12. Random Forest Meta-Learner (UCB)
**Bài Học:** Bài 9 (Random Forest Dynamic Pricing)
**Loại:** Ensemble (dựa trên cây)
**Trực Quan:** Fit Random Forest cho mỗi arm; sử dụng bất đồng ý cây làm bonus khám phá
**Khi Sử Dụng:** Phần thưởng phi tuyến, mạnh mẽ với ngoại lệ, tương tác tính năng
**Tham Chiếu:** [docs/algorithms/sklearn_meta.md](./algorithms/sklearn_meta.md)

### 13. Random Forest Thompson Sampling
**Bài Học:** Không có bài học chuyên dụng (có sẵn policy)
**Loại:** Ensemble (Stochastic)
**Trực Quan:** Cây bootstrap cho mỗi arm; lấy mẫu từ dự đoán ensemble
**Khi Sử Dụng:** Kết quả nhị phân với phi tuyến dựa trên cây

### 14. Gaussian Process UCB
**Bài Học:** Bài 10 (GP-UCB Clinical Trial)
**Loại:** Probabilistic (Gaussian Process)
**Trực Quan:** Duy trì posterior GP đầy đủ cho mỗi arm; UCB tận dụng phương sai posterior làm độ không chắc chắn
**Khi Sử Dụng:** Các quyết định khối lượng thấp, bề mặt phi tuyến phức tạp, cần định lượng độ không chắc chắn
**Trade-off:** O(n²) inference — không phù hợp cho thông lượng cao
**Tham Chiếu:** [docs/algorithms/gp_ucb.md](./algorithms/gp_ucb.md)

---

## Các Policy Thích Ứng Theo Thời Gian & Đặc Biệt

### 15. Sliding-Window LinUCB (LinUCB-SW)
**Bài Học:** Bài 12 (Sliding-Window LinUCB Flash Sale)
**Loại:** Tất định (có cửa sổ)
**Trực Quan:** LinUCB với cửa sổ trượt của các quan sát gần đây; dữ liệu cũ bị loại bỏ
**Khi Sử Dụng:** Phân phối phần thưởng thay đổi nhanh (flash sale, shift theo mùa)
**Tham Chiếu:** [docs/algorithms/linucb_sw.md](./algorithms/linucb_sw.md)

### 16. Drift Detection (Page-Hinkley + LinUCB)
**Bài Học:** Bài 13 (PageHinkley Drift Detection)
**Loại:** Thích Ứng
**Trực Quan:** Theo dõi sự dịch chuyển phân phối qua Page-Hinkley test; reset arm khi phát hiện drift
**Khi Sử Dụng:** Môi trường không dừa, phát hiện và thích ứng với drift khái niệm
**Tham Chiếu:** [docs/advanced_features.md](./advanced_features.md#6-pagehinkleydetector--reward-drift-detection)

### 17. CATS (Continuous Action Tree Search)
**Bài Học:** Bài 15 (CATS Real-Time Bidding)
**Loại:** Cây dựa trên Hành Động Liên Tục
**Trực Quan:** Cây nhị phân phân chia không gian hành động liên tục; Thompson Sampling mỗi leaf
**Khi Sử Dụng:** Đấu giá thời gian thực, giá đấu giá liên tục, điều chỉnh tham số
**Arms:** Không có (phạm vi liên tục thay thế)
**Tham Chiếu:** [docs/algorithms/cats.md](./algorithms/cats.md)

---

## Các Policy Bootstrapped (Nâng Cao)

### Bootstrapped UCB
**Bài Học:** Không có bài học chuyên dụng (có sẵn policy)
**Loại:** Ensemble (Bootstrapped)
**Trực Quan:** Huấn luyện nhiều mô hình qua bootstrap sampling; chọn arm có trung bình ensemble cao nhất + độ không chắc chắn
**Khi Sử Dụng:** Định lượng độ không chắc chắn mạnh mẽ, phi tuyến thông qua sklearn base estimators

### Bootstrapped Thompson Sampling
**Bài Học:** Không có bài học chuyên dụng (có sẵn policy)
**Loại:** Ensemble (Stochastic)
**Trực Quan:** Ensemble bootstrap; lấy mẫu một mô hình, chọn arm tốt nhất mỗi mẫu
**Khi Sử Dụng:** Kết quả nhị phân/thưa với đa dạng ensemble

---

## Học Ngoài Tuyến

### Inverse Propensity Scoring (IPS)
**Bài Học:** Bài 14 (Offline Evaluation IPS/DR/NCIS)
**Loại:** Phương pháp Đánh Giá
**Trực Quan:** Cân nhân lại phần thưởng lịch sử bằng `π(a|x) / p(a|x)` để sửa độ thiên vị logging policy
**Khi Sử Dụng:** Đánh giá policy trên dữ liệu log, bootstrap từ lịch sử có độ thiên vị
**Tham Chiếu:** [docs/algorithms/offpolicy_ips.md](./algorithms/offpolicy_ips.md)

### Doubly Robust (DR)
**Bài Học:** Bài 14 (Offline Evaluation IPS/DR/NCIS)
**Loại:** Phương pháp Đánh Giá (Hybrid)
**Trực Quan:** Kết hợp mô hình phần thưởng (phương pháp trực tiếp) + IPS; không thiên vị nếu cái nào là chính xác
**Khi Sử Dụng:** Đánh giá phương sai thấp hơn IPS một mình, cần mô hình phần thưởng
**Tham Chiếu:** [docs/evaluation.md](./evaluation.md)

### Normalized Capped Importance Sampling (NCIS)
**Bài Học:** Bài 14 (Offline Evaluation IPS/DR/NCIS)
**Loại:** Phương pháp Đánh Giá
**Trực Quan:** IPS với trọng số có giới hạn + chuẩn hóa để ngăn chặn các vụ nổ phương sai
**Khi Sử Dụng:** Sự khác biệt xác suất cực trị (logging policy gần như tất định)
**Tham Chiếu:** [docs/evaluation.md](./evaluation.md)

---

## Hướng Dẫn Chọn Policy

| Tình Huống | Các Policy Được Khuyến Nghị | Tại Sao |
|-----------|---|---|
| **Không bối cảnh** | UCB1, Thompson, Epsilon-Greedy | Optimal không bối cảnh |
| **Bối cảnh tuyến tính** | LinUCB, LinTS, Logistic | Nhanh, có thể giải thích, bị ràng buộc regret |
| **Bối cảnh phi tuyến** | Neural Linear, Cluster Routing, Forest | Bắt được tương tác, phi dừa |
| **Kết quả nhị phân** | Logistic, Thompson, Forest | Phù hợp với phần thưởng 0/1 |
| **Phần thưởng trôi** | LinUCB-SW, Drift Detection | Thích ứng với sự dịch chuyển phân phối |
| **Hành động liên tục** | CATS | Phân chia không gian hành động cây |
| **Đánh giá ngoại tuyến** | IPS, DR, NCIS | Khử độ thiên vị dữ liệu log |
| **Quyết định khối lượng thấp** | GP-UCB | Độ không chắc chắn đầy đủ, O(n²) có thể chấp nhận |
| **Thông lượng cao** | LinUCB, Forest, Neural Linear | O(d²) hoặc O(1) mỗi quyết định |

---

## Độ Phức Tạp & Hiệu Năng

| Policy | Độ Phức Tạp Thời Gian | Bộ Nhớ | Phù Hợp Nhất |
|--------|---|---|---|
| **Epsilon-Greedy** | O(1) | O(n_arms) | Đường cơ sở |
| **UCB1** | O(1) | O(n_arms) | Không bối cảnh |
| **Thompson** | O(1) | O(n_arms) | Bayesian đường cơ sở |
| **LinUCB** | O(d²) | O(d² × n_arms) | Bối cảnh tuyến tính |
| **LinTS** | O(d²) | O(d² × n_arms) | Bayesian bối cảnh |
| **Logistic** | O(d) | O(d × n_arms) | Kết quả nhị phân |
| **Neural Linear** | O(hidden²) | O(embedding_dim × hidden × d) | Phi tuyến + Bayesian |
| **Random Forest** | O(trees × depth) | O(trees × depth × features) | Phi tuyến ensemble |
| **GP-UCB** | O(n²) | O(n²) | Đắt tiền, khối lượng thấp |
| **Cluster Routing** | O(d² × K) | O(d² × K × n_arms) | Phi dừa, phân chia |

---

## Chương Trình Học Được Ánh Xạ Đến Các Policy

| Chỉ Mục | Bài Học | Policy | Độ Khó |
|--------|--------|---------|---------|
| 0 | Explore vs Exploit | epsilon_greedy | Beginner |
| 1 | UCB1 Landing Page | ucb1 | Beginner |
| 2 | Thompson Sampling Email | thompson | Beginner |
| 3 | LinUCB Product Rec | linucb | Intermediate |
| 4 | LinTS Loan Offer | lints | Intermediate |
| 5 | Logistic Bandits Ad CTR | logistic_ucb | Intermediate |
| 6 | Cluster Routing Music | linucb (clustered) | Intermediate |
| 7 | LinUCB-Hybrid News | linucb_hybrid | Advanced |
| 8 | NeuralLinear Video | neural_linear | Advanced |
| 9 | Random Forest Pricing | random_forest_ucb | Advanced |
| 10 | GP-UCB Clinical Trial | gp_ucb | Advanced |
| 11 | Softmax Playlist | softmax | Intermediate |
| 12 | Sliding-Window Flash Sale | linucb_sw | Advanced |
| 13 | PageHinkley Drift Detection | linucb (drift-aware) | Advanced |
| 14 | Offline Eval IPS/DR/NCIS | linucb | Advanced |
| 15 | CATS Real-Time Bidding | cats | Advanced |
| 16 | Production Constraints | linucb | Advanced |

---

## Câu Hỏi?

Xem các tài liệu thuật toán riêng biệt (trong thư mục `/algorithms/`) để biết chi tiết toán học, ví dụ sử dụng và hướng dẫn điều chỉnh siêu tham số.
