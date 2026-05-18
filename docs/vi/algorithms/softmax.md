# Softmax (Khám Phá Có Kiểm Soát Nhiệt Độ)

## Tổng Quan

**Loại:** Stochastic / Khám phá-Khai Thác qua Điều Chỉnh Nhiệt Độ
**Bài Học:** Bài Học 11 (Softmax Playlist Generation)
**Loại Policy:** `softmax`
**Phù Hợp Nhất:** Các tình huống yêu cầu khám phá ngẫu nhiên mịn với cường độ có thể điều chỉnh

---

## Cách Thức Hoạt Động

Softmax chuyển đổi điểm số arm thành phân phối xác suất qua hàm softmax, sau đó lấy mẫu một arm.

$$P(\text{arm} = a) = \frac{\exp(\text{score}_a / \tau)}{\sum_{a'} \exp(\text{score}_{a'} / \tau)}$$

Tham số nhiệt độ `τ` kiểm soát cường độ khám phá:
- **`τ` → 0** (lạnh): Tất định (luôn chọn tốt nhất)
- **`τ` lớn** (nóng): Phân phối đều (khám phá thuần túy)
- **`τ` = 1** (vừa phải): Cân bằng khám phá-khai thác

---

## Siêu Tham Số Chính

**`tau`** (Nhiệt Độ)
- **Mặc định:** 1.0
- **Phạm vi:** > 0
- **Hiệu ứng:** Kiểm soát độ sắc nét của khám phá
  - Nhỏ (0.1): Khai thác nhiều, khám phá hiếm
  - Vừa (1.0): Cân bằng
  - Lớn (10.0): Khám phá thường xuyên, giống phân phối đều

---

## Ví Dụ Sử Dụng

```python
from coba.policies.softmax import SoftmaxArmModel

model = SoftmaxArmModel(
    arm="option_A",
    rng=np.random.default_rng(42),
    tau=1.0  # temperature
)

# Trước cập nhật, tất cả arm có điểm = 0 → phân phối đều
score = model.score(context=None)  # Trả về một mẫu từ phân phối

# Sau khi quan sát phần thưởng, điểm số phân kỳ
model.update(context=None, reward=1.0)  # Phần thưởng cao tăng điểm
score_after = model.score(context=None)  # Lấy mẫu giờ ưu tiên arm này hơn
```

---

## Khi Nào Sử Dụng

| Tình Huống | Khuyến Nghị |
|-----------|-----------|
| **Khám phá mịn** | ✅ Lựa chọn tốt |
| **Phần thưởng nhị phân/thưa** | ✅ Hoạt động tốt |
| **Cần ngẫu nhiên có thể tái tạo** | ✅ Xác định thông qua RNG |
| **Muốn xác suất có thể diễn giải** | ✅ Softmax rất trực quan |
| **Đánh giá ngoại tuyến** | ⚠️ Yêu cầu ghi nhật ký xác suất |

---

## So Sánh với Các Policy Khác Không Dùng Ngữ Cảnh

| Policy | Cơ Chế | Khám Phá |
|--------|-------|----------|
| **UCB1** | Bonus tự tin | Bonus lạc quan |
| **Thompson** | Lấy mẫu từ posterior Beta | Không chắc chắn Bayesian |
| **Softmax** | Chia tỉ lệ nhiệt độ | Khám phá ngẫu nhiên mịn |
| **Epsilon-Greedy** | Xác suất epsilon | Chuyển đổi cứng |

Softmax mềm hơn epsilon-greedy (xác suất mịn vs chuyển đổi cứng) và đơn giản hơn Thompson để điều chỉnh (chỉ một tham số).

---

## Ngữ Cảnh Bài Học

**Bài Học 11: Softmax Playlist Generation**

Người dùng kiểm soát `tau` để xem cách nhiệt độ ảnh hưởng đến việc tạo danh sách phát:
- Temp thấp: Thuật toán xu hướng đề xuất những bài hát yêu thích tương tự
- Temp cao: Thuật toán đề xuất các bài hát đa dạng, bất ngờ
- Hoàn hảo để dạy tradeoff khám phá-khai thác trong một lĩnh vực dễ hiểu (âm nhạc)

---

## Tham Khảo

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed), §2.2 Softmax Action Selection
- Kuleshov & Precup, *Algorithms for Multi-Armed Bandit Problems* (2014)
