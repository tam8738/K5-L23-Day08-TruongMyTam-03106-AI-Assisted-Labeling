# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 509; đối tượng sau duyệt: 302; ngưỡng accepted: IoU >= 0.9
- Đã **loại 26 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 247 | 247/509 = 48.5% | 509 box máy |
| edited (sửa) | 45 | 45/509 = 8.8% | 509 box máy |
| deleted (xoá) | 99 | 99/509 = 19.4% | 509 box máy |
| added (vẽ thêm) | 9 | 9/302 = 3.0% | 302 đối tượng sau duyệt |

Trong số deleted + added, **3 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 6.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 122 | 22 | 42 | 1 |
| bicycle | 2 | 2 | 3 | 1 |
| motorbike | 77 | 11 | 22 | 3 |
| car | 32 | 4 | 8 | 0 |
| bus | 3 | 0 | 0 | 2 |
| truck | 1 | 0 | 2 | 1 |
| three_wheeler | 10 | 6 | 22 | 1 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 46 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 40 | 23 | 6 | 63.5% (KTC 95%: 51–74%) | 87.0% (KTC 95%: 74–94%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 10/12 = 83.3% (KTC 95%: 55–95%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - wrong_class: 3/3
  - loose: 2/3
  - extra: 3/3
  - missing: 2/3
  - lọt: img_013.jpg (loose), img_003.jpg (missing)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
