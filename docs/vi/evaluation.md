# Đánh Giá Ngoại Tuyến (Offline Policy Evaluation)

Khi bạn muốn kiểm thử một thuật toán Reinforcement Learning mới hoặc một policy mới (ví dụ thay đổi LinUCB thành LinTS), bạn cần đánh giá hiệu năng của nó trên dữ liệu lịch sử (historical logs) trước khi tung ra A/B test. Tuy nhiên, vì các log này được sinh ra bởi một "logging policy" (policy cũ) khác hoàn toàn so với policy bạn đang test, bạn không thể đơn giản là tính trung bình cộng phần thưởng. Bạn phải xử lý độ lệch (bias) bằng các phương pháp Đánh Giá Ngoại Tuyến (Offline Policy Evaluation - OPE).

COBA cung cấp sẵn 3 phương pháp luận đánh giá tiêu chuẩn:

## 1. Rejection Sampling (Li et al., 2010)
* **Cách hoạt động**: Lặp qua từng bản ghi trong dữ liệu lịch sử và "hỏi" policy mới xem nó sẽ chọn mức giá nào. Nếu policy mới chọn **giống hệt** với mức giá đã chọn trong lịch sử, ta chấp nhận bản ghi này và giữ lại phần thưởng. Ngược lại, nếu khác nhau, ta loại bỏ hoàn toàn bản ghi đó.
* **Ưu điểm**: Cực kỳ đơn giản, trực quan và hoàn toàn không bị thiên lệch (unbiased) **nếu và chỉ nếu** dữ liệu lịch sử được thu thập bằng một policy ngẫu nhiên đồng đều (uniform random).
* **Nhược điểm**: Lãng phí rất nhiều dữ liệu (có thể vứt bỏ đến >80% log). Gần như không thể sử dụng nếu policy thu thập dữ liệu (logging policy) bị lệch hướng hoặc thiên vị mức giá nào đó.

## 2. Doubly Robust - DR (Dudík et al., 2011)
* **Cách hoạt động**: Là sự kết hợp sức mạnh giữa mô hình dự đoán phần thưởng (Direct Method) và kỹ thuật tính trọng số nghịch đảo (Inverse Propensity Scoring - IPS).
  $$ \hat{V}_{DR} = \frac{1}{N} \sum_{i=1}^N \left( \hat{r}(x_i, \pi(x_i)) + \frac{\mathbb{I}(a_i = \pi(x_i))}{p_i} (r_i - \hat{r}(x_i, a_i)) \right) $$
* **Ưu điểm**: Được gọi là "Đúng Kép" (Doubly Robust) vì công cụ ước lượng này sẽ không bị thiên lệch (unbiased) nếu *một trong hai* yếu tố sau chính xác: Mô hình dự đoán phần thưởng $\hat{r}$ đúng, *hoặc* xác suất propensity $p_i$ đúng. Nó giúp giảm thiểu đáng kể phương sai (variance) so với thuật toán IPS thuần túy.
* **Nhược điểm**: Yêu cầu hệ thống phải lưu trữ xác suất (propensities) mà logging policy đã sử dụng để chọn mức giá, đồng thời đòi hỏi phải có một mô hình ước lượng trước phần thưởng của mỗi ngữ cảnh.

## 3. Normalized Capped Importance Sampling - NCIS (Gilotte et al., 2018)
* **Cách hoạt động**: Tương tự như IPS chuẩn (nhân phần thưởng với hệ số $\frac{\pi(a|x)}{p(a|x)}$). Tuy nhiên, để ngăn chặn tình trạng phương sai bùng nổ (variance explosions) khi xác suất $p(a|x)$ quá nhỏ tiệm cận 0, NCIS sẽ "cắt bỏ" (cap) trọng số tại một giá trị cực đại nhất định, sau đó chuẩn hóa (normalize) toàn bộ hệ số trọng số sao cho tổng của chúng bằng 1.
* **Ưu điểm**: Phương sai thấp hơn nhiều so với IPS chuẩn, rất lý tưởng để sử dụng khi xác suất propensities cực kỳ nhỏ hoặc phân bố mất cân bằng.
* **Nhược điểm**: Do tính năng "cắt bớt" trọng số, phương pháp này sẽ chấp nhận đưa vào một sai số nhỏ (slight bias) để đổi lấy sự ổn định.

## Ví Dụ Minh Hoạ (Step-by-Step)

Dưới đây là ví dụ từng bước cách sử dụng module `evaluation` của COBA để kiểm thử một `ClusterRouter` đã được huấn luyện (fitted) bằng dữ liệu lịch sử.

```python
import numpy as np
from coba.evaluation.metrics import rejection_sampling_eval, doubly_robust_eval, ncis_eval
from coba.routers.cluster_router import ClusterRouter

# 1. Chuẩn bị dữ liệu lịch sử (historical data)
n_samples = 1000
n_features = 5
contexts = np.random.randn(n_samples, n_features) # Vectors ngữ cảnh giả lập
decisions = np.random.choice(["arm_A", "arm_B"], size=n_samples) # Các quyết định (actions) đã lưu log
rewards = np.random.rand(n_samples) # Phần thưởng thực tế trong log
propensities = np.full(n_samples, 0.5) # Giả định logging policy chọn ngẫu nhiên đều 50/50

# 2. Khởi tạo và huấn luyện router
# (Giả định `router` đã được huấn luyện trên tập dữ liệu train)
# router = ClusterRouter(...)
# router.fit(X_train, y_train_actions, y_train_rewards)

# 3. Đánh giá bằng Rejection Sampling
# Lý tưởng khi log được thu thập hoàn toàn ngẫu nhiên đều (uniform random).
rej_result = rejection_sampling_eval(
    router=router, 
    contexts=contexts, 
    decisions=decisions, 
    rewards=rewards
)
print(rej_result)
# Output: EvalResult(method='rejection_sampling', estimated_reward=0.5100, n_used=502/1000 [50.2%])

# 4. Đánh giá bằng Doubly Robust (DR)
# Yêu cầu phải có ước lượng phần thưởng (reward estimates) cho các hành động đã chọn trong log.
# Ở đây ta giả lập các ước lượng bằng mảng random để minh hoạ.
reward_estimates = np.random.rand(n_samples) 

dr_result = doubly_robust_eval(
    router=router,
    contexts=contexts,
    decisions=decisions,
    rewards=rewards,
    propensities=propensities,
    reward_estimates=reward_estimates
)
print(dr_result)
# Output: EvalResult(method='doubly_robust', estimated_reward=0.4950, n_used=1000/1000 [100.0%])

# 5. Đánh giá bằng NCIS (Normalized Capped Importance Sampling)
# Cần tính toán policy_scores (điểm/xác suất mà router của ta dự đoán cho các quyết định trong log)
policy_scores = np.array([
    router.score_all(ctx).get(dec, 0.0) 
    for ctx, dec in zip(contexts, decisions)
])

ncis_result = ncis_eval(
    policy_scores=policy_scores,
    logging_scores=propensities, # propensities của logging policy
    rewards=rewards
)
print(ncis_result)
# Output: EvalResult(method='ncis', estimated_reward=0.4980, n_used=1000/1000 [100.0%])
```
