# Sliding-Window LinUCB (LinUCB Cửa Sổ Trượt)

## Tổng Quan

**Loại:** Tất định, Thích Ứng Theo Thời Gian
**Bài Học:** Bài Học 12 (Sliding-Window LinUCB Flash Sale)
**Loại Policy:** `linucb_sw`
**Phù Hợp Nhất:** Môi trường phi dừa nơi dữ liệu gần đây quan trọng hơn

---

## Cách Thức Hoạt Động

Sliding-Window LinUCB là LinUCB chỉ giữ lại `W` quan sát gần đây nhất cho mỗi arm. Dữ liệu cũ bị loại bỏ.

**LinUCB Tiêu Chuẩn:**
$$\text{score}(x) = x^\top \hat{\beta} + \alpha \sqrt{x^\top A^{-1} x}$$

**Sliding-Window LinUCB:**
- Duy trì hồi quy ridge cho mỗi arm chỉ sử dụng `W` quan sát cuối cùng
- Khi cửa sổ đầy, loại bỏ quan sát cũ nhất và thêm cái mới
- Cùng hàm điểm như LinUCB, nhưng trên dữ liệu theo cửa sổ

---

## Siêu Tham Số Chính

**`window_size` (hoặc `linucb_sw_window`)**
- **Mặc định:** 200
- **Phạm vi:** > 0 (thường 100–1000)
- **Hiệu ứng:**
  - Cửa sổ nhỏ (50): Thích ứng nhanh với drift, nhưng ồn ào hơn
  - Cửa sổ lớn (1000): Ổn định nhưng chậm thích ứng
  - Nguyên tắc: ~3-5× số lần cập nhật dự kiến mỗi arm

---

## Ví Dụ Sử Dụng

```python
from coba.policies.linucb import SlidingWindowLinUCBArmModel

model = SlidingWindowLinUCBArmModel(
    arm="variant_A",
    n_features=5,
    window_size=200,  # Giữ 200 quan sát cuối cùng
    alpha=1.0,
    l2_lambda=1.0
)

# 200 quan sát đầu tiên điền cửa sổ
for i in range(200):
    context = np.random.randn(5)
    model.update(context, reward=np.random.rand())

# Bắt đầu từ quan sát 201, dữ liệu cũ bị loại bỏ
model.update(context, reward=0.9)  # Thay thế quan sát cũ nhất (200 bước trước)
```

---

## Ví Dụ Thích Ứng Drift: Flash Sale

**Trường Hợp Sử Dụng: Flash Sale**

Nhu cầu sản phẩm thay đổi trong suốt flash sale 24 giờ:
- **6 AM:** Nhu cầu thấp, chiết khấu cao hiệu quả (reward = chiết khấu * nhu cầu)
- **12 PM:** Nhu cầu đỉnh, chiết khấu ít hiệu quả
- **6 PM:** Suy giảm, chiết khấu tích cực lại hiệu quả

LinUCB tiêu chuẩn học một mô hình toàn cục trung bình theo thời gian. Sliding-Window LinUCB chỉ xem các mẫu gần đây → thích ứng nhanh hơn.

```
Thời Gian  | Nhu Cầu | Chiết Khấu Tốt | Điểm LinUCB | Điểm SW-LinUCB
-----------|--------|----------------|------------|----------------
06:00      | 10%    | 50%            | (cũ)       | 50% ← dữ liệu tươi
12:00      | 100%   | 5%             | (cũ)       | 5%  ← đỉnh gần đây
18:00      | 30%    | 40%            | (cũ)       | 40% ← thích ứng
```

---

## Khi Nào Sử Dụng

| Tình Huống | Khuyến Nghị |
|-----------|-----------|
| **Phần thưởng dừa** | ❌ Không cần; dùng LinUCB tiêu chuẩn |
| **Phần thưởng trôi chậm** | ⚠️ Kích thước cửa sổ vừa |
| **Drift khái niệm nhanh** | ✅ Thiết yếu; cửa sổ nhỏ |
| **Mẫu theo mùa** | ✅ Lựa chọn tốt |
| **Các cuộc đấu giá thời gian thực, định giá** | ✅ Lý tưởng |

---

## So Sánh với Drift Detection

| Phương Pháp | Cơ Chế | Khi Nào Reset |
|-----------|-------|-------------|
| **Sliding-Window** | Loại bỏ dữ liệu cũ liên tục | Luôn (mỗi bước) |
| **Drift Detection** | Theo dõi sự dịch chuyển, reset khi báo động | Chỉ khi phát hiện thay đổi |
| **Adaptive γ** | Chiết khấu quan sát cũ | gamma < 1.0 |

- **Sliding-Window:** Đơn giản, không cần điều chỉnh phát hiện
- **Drift Detection:** Hiệu quả (không reset trừ khi cần), nhưng yêu cầu điều chỉnh ngưỡng
- **Adaptive γ:** Mờ dần-ra (không cắt cứng)

---

## Ngữ Cảnh Bài Học

**Bài Học 12: Sliding-Window LinUCB Flash Sale**

Người dùng kiểm soát kích thước cửa sổ và quan sát cách thích ứng drift hoạt động:
- **Cửa sổ nhỏ (50):** Theo dõi sự thay đổi nhu cầu chặt chẽ nhưng ồn ào hơn
- **Cửa sổ lớn (500):** Trơn hơn nhưng chậm phía sau drift
- **Cửa sổ tối ưu (200):** Cân bằng giữa tính phản ứng và ổn định

So sánh với chạy LinUCB tiêu chuẩn để xem sự khác biệt trong thích ứng thời gian thực.

---

## Ghi Chú Triển Khai

- **Hiệu Suất:** Loại bỏ quan sát cũ khỏi hồi quy ridge yêu cầu cập nhật ma trận nghịch đảo (công thức Sherman-Morrison ngược). O(d²) mỗi lần loại bỏ.
- **Ổn Định Số:** Kích thước cửa sổ nên >> d (số chiều tính năng) để tránh suy giảm hạng.
- **Bộ Nhớ:** O(window_size × d) thay vì O(total_observations × d).

---

## Tham Khảo

- Jadbabaie et al., *Online Optimization Under Time-Varying Distributions* (2016)
- Besbes et al., *Stochastic Optimization under Time-Varying Distributions* (2015)
- Xem thêm: [Phát Hiện Drift](./advanced_features.md#6-pagehinkleydetector--reward-drift-detection)
