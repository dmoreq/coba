# CATS (Continuous Action Tree Search)

## Tổng Quan

**Loại:** Tìm Kiếm Cây Hành Động Liên Tục
**Bài Học:** Bài Học 15 (CATS Real-Time Bidding)
**Loại Policy:** `cats`
**Phù Hợp Nhất:** Không gian hành động liên tục (ví dụ: lệnh đấu giá, giá cả, tham số)

---

## Cách Thức Hoạt Động

Thay vì các arm rời rạc, CATS phân chia không gian hành động liên tục `[a_min, a_max]` thành một cây nhị phân.

**Cấu Trúc Leaf:**
- Mỗi leaf là một phạm vi liên tục: `[lo, hi]`
- Thompson Sampling duy trì một posterior Beta cho mỗi leaf
- Tại thời điểm quyết định, lấy mẫu từ posterior mỗi leaf, chọn leaf có mẫu cao nhất
- Trả lại điểm giữa của leaf dưới dạng hành động liên tục

**Ví Dụ Cây (độ sâu=3, phạm vi=[0, 5]):**
```
                    [0, 5]
                   /      \
              [0, 2.5]    [2.5, 5]
              /    \        /    \
        [0,1.25] [1.25,2.5] [2.5,3.75] [3.75,5]
        ...
```

Sau khi quan sát phần thưởng cho hành động 1.2, thuật toán cập nhật posterior Beta cho leaf chứa 1.2.

---

## Siêu Tham Số Chính

**`a_min` / `a_max`**
- **Giới Hạn Phạm Vi Hành Động**
- **Ví Dụ:** `a_min=0.01` (lệnh đấu giá tối thiểu), `a_max=10.0` (lệnh đấu giá tối đa)

**`cats_depth`**
- **Độ Sâu Cây (Chiều Cao)**
- **Mặc Định:** 6
- **Phạm Vi:** 1–12
- **Hiệu Ứng:**
  - Độ sâu 1: Chỉ 2 leaf (nửa trên/dưới)
  - Độ sâu 6: 64 leaf (phân chia chi tiết)
  - Độ sâu cao hơn = độ phân giải tốt hơn, nhưng học chậm hơn mỗi leaf

---

## Ví Dụ Sử Dụng

```python
from coba.continuous.bandit import ContinuousBandit
from coba.config import BanditConfig

# Đấu giá thời gian thực: tối ưu hóa giá lệnh đấu giá trong phạm vi [0.01, 10.0]
config = BanditConfig(
    policy=PolicyType.CATS,
    cats_a_min=0.01,
    cats_a_max=10.0,
    cats_depth=6  # 64 leaf
)

bandit = ContinuousBandit(
    a_min=0.01,
    a_max=10.0,
    n_features=4,  # Các tính năng ngữ cảnh người dùng
    config=config
)

# Quyết định lệnh đấu giá
context = np.array([user_value, time_of_day, competition, pacing])
decision = bandit.decide(context)
chosen_bid = decision.chosen_action  # Float trong [0.01, 10.0]

# Quan sát kết quả (tỷ lệ thắng, doanh thu, v.v.)
bandit.update(context=context, arm=chosen_bid, reward=0.95)
```

---

## Discrete vs Continuous Bandits

| Khía Cạnh | Rời Rạc (LinUCB) | Liên Tục (CATS) |
|-----------|---|---|
| **Hành Động** | `["A", "B", "C"]` | `0.0–10.0` (vô hạn) |
| **Học Tập** | Mô hình tuyến tính mỗi arm | Thompson lấy mẫu mỗi leaf |
| **Thời Gian Quyết Định** | O(n_arms × d) | O(tree_depth × d) |
| **Trường Hợp Sử Dụng** | Lựa chọn nội dung | Tối ưu hóa giá/lệnh đấu giá |

---

## Ngữ Cảnh Bài Học

**Bài Học 15: CATS Real-Time Bidding**

Người dùng tối ưu hóa giá lệnh đấu giá trong môi trường RTB mô phỏng:
- Ngữ Cảnh: Giá trị người dùng, thời gian ngày, cạnh tranh, tốc độ
- Hành Động: Giá lệnh đấu giá (liên tục, $0–$10)
- Phần Thưởng: Xác suất thắng cuộc
- Mục Tiêu: Tìm điểm ngọt ngào (lệnh đấu giá cao → luôn thắng → lãng phí tiền; lệnh đấu giá thấp → không bao giờ thắng)

**Các Điều Khiển Tương Tác:**
- Điều chỉnh `cats_depth` để xem phân chia chi tiết vs thô
- Quan sát phạm vi lệnh đấu giá nào mà cây khám phá
- Trực quan hóa điểm leaf theo thời gian thực (hiển thị trong `leafScores` endpoint)

---

## Chi Tiết Thuật Toán

### Xây Dựng Cây
Cây là một **cây nhị phân hoàn chỉnh** với `2^depth` leaf.

Node `i` ở độ sâu `d` bao phủ phạm vi:
$$[\text{a\_min} + i \cdot \text{width}, \text{a\_min} + (i+1) \cdot \text{width}]$$

trong đó width = `(a_max - a_min) / 2^depth`

### Lựa Chọn Leaf
Tại mỗi bước quyết định:
1. Với mỗi leaf, lấy mẫu từ posterior Beta
2. Chọn leaf có giá trị mẫu cao nhất
3. Trả lại điểm giữa dưới dạng hành động liên tục

### Học Tập
Sau khi quan sát reward `r` cho hành động `a`:
1. Tìm leaf chứa `a`
2. Cập nhật posterior Beta: `alpha += reward`, `beta += (1 - reward)`
3. Posterior của các leaf khác không thay đổi

---

## Khi Nào Sử Dụng

| Tình Huống | Khuyến Nghị |
|-----------|-----------|
| **Arms rời rạc** | ❌ Dùng LinUCB thay thế |
| **Tối ưu hóa liên tục** | ✅ Hoàn hảo |
| **Định giá thời gian thực** | ✅ Lý tưởng |
| **Kiểm tra A/B với giá trị** | ✅ Lựa chọn tốt |
| **Điều chỉnh tham số** | ✅ Thích hợp |
| **Hàng nghìn hành động** | ✅ Hiệu quả hơn các lựa chọn thay thế |

---

## So Sánh với Các Lựa Chọn Thay Thế

| Phương Pháp | Cách | Độ Phức Tạp |
|-----------|------|-----------|
| **Tìm Kiếm Lưới + LinUCB** | Rời rạc thành K bucket, dùng K-arm LinUCB | O(d² × K) |
| **Gaussian Process** | Posterior GP trên hàm liên tục | O(n²) |
| **Dựa Vào Gradient** | Bandit với ước lượng gradient (khó) | O(d) mỗi lần lặp |
| **CATS** | Cây nhị phân + Thompson mỗi leaf | O(depth × d) |

CATS **đơn giản hơn GP** (Thompson vs EM), **nhanh hơn tìm kiếm lưới** (độ sâu logarit), và **tránh không ổn định gradient**.

---

## Leaf Scores Endpoint

Backend `/api/sessions/{id}/leaf-scores` trả lại:

```json
{
  "leaves": [
    {
      "index": 0,
      "lo": 0.01,
      "hi": 5.0,
      "midpoint": 2.505,
      "ucb": 1.234
    },
    ...
  ],
  "active_leaf": 5,
  "sampled_action": 3.256
}
```

**Trực Quan Hóa:** Bài học hiển thị điểm UCB leaf dưới dạng các thanh (cao hơn = hứa hẹn hơn), tô sáng leaf hoạt động, và đánh dấu điểm hành động được chọn.

---

## Ghi Chú Triển Khai Frontend

Thành phần bài học hiển thị:
- **Điểm UCB Leaf** dưới dạng thanh (cao hơn = hứa hẹn hơn)
- **Tô sáng Leaf hoạt động** (được lấy mẫu ở bước này)
- **Điểm hành động** có chiều dài phạm vi được đánh dấu

---

## Tham Khảo

- Kannan et al., *Bandits with Delayed, Aggregated Anonymous Feedback* (ICML 2018)
- Xem thêm: [Advanced Features](./advanced_features.md) cho các ràng buộc sản xuất (min_pull_rates, abstention)
