# Label contract — Phố châu Á — BỔ SUNG v1.1 — 03106 - Truong My Tam

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 và thêm quy tắc mới rút ra từ lỗi của máy / lỗi khi duyệt. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | Phân loại theo hình thái và bàn đạp: có bàn đạp đạp chân rõ ràng và khung xe đạp mảnh → `bicycle`; có sàn để chân/vỏ yếm như xe tay ga hoặc không có bàn đạp → `motorbike`. | `img_006.jpg` | `CAL-02` |
| 2 | Khối xe máy đỗ/chạy dày đặc: từng xe hay `ignore_region`? | Nếu từng xe nhìn rõ bánh/thân (cao ≥ 20 px, tách được cá thể) như hàng xe đỗ gần ở `img_006.jpg`, `img_014.jpg` → gán từng `motorbike`; nếu cụm ≥ 5 xe ở xa/nhòe sáng không tách được biên từng xe như `img_024.jpg` → gộp vào `ignore_region`. | `img_024.jpg` | `CAL-02` |
| 3 | Hàng nhô ra trên xe đạp / ba gác: box và mask xe có gồm hàng không? | Giỏ/thùng gắn cố định thuộc xe (`img_006.jpg`); hàng hóa rời nhô ra quá ~20% chiều dài xe hoặc người ngồi trên thùng xe (`img_003.jpg`) **không** gộp vào mask xe, và người phải gán box `person` riêng. | `img_003.jpg` | `CAL-03` |
| 4 | Người ngồi trong xe ba bánh có mui che: ngưỡng nhìn thấy ≥ 20% áp dụng thế nào? | Thay vì ước lượng diện tích mơ hồ, quy ước: nhìn thấy rõ **đầu + vai** hoặc **nửa thân trên** qua khung hở (`img_003.jpg`, `img_013.jpg`) thì gán `person`; nếu chỉ thấy bóng mờ hoặc một phần tay/chân nhỏ trong mui tối thì không gán. | `img_013.jpg` | `CAL-03` |
| 5 | Kính chắn gió / cửa sổ xe và người lái trong mask SAM: tính vào đâu? | Kính chắn gió/khung kính thuộc mask phương tiện; nhưng **người lái và hành khách ngồi lộ ra ở khoang hở** (`img_013.jpg`) không thuộc mask `three_wheeler` — khi dùng SAM phải đặt điểm âm (`--neg`) lên người lái để tách khỏi mask xe. | `img_013.jpg` | `CAL-03` |
| 6 | Box máy chồng nhau quá nhiều trong ảnh đêm (`img_024.jpg`): gộp hay giữ? | Xóa các box máy trùng lặp/nhòe sáng trên cùng một phương tiện, chỉ giữ 1 box ôm sát phần nhìn thấy; với vùng đèn xe nhòe dày đặc ở xa thì bao bằng `ignore_region`. | `img_024.jpg` | `CAL-02` |
| 7 | Có đặt mức `conf` tối thiểu để tự động chấp nhận box máy không cần sửa không? | **Không.** Mọi box máy dù `conf` cao (> 0.85) vẫn phải kiểm tra mắt vì YOLO26 có thể gán sai lớp (`tuk-tuk` → `car`/`truck`) hoặc giữ box lỏng ở hậu cảnh (`img_013.jpg`). | `img_004.jpg` | `CAL-01` |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_004.jpg` | Dãy tuk-tuk đỗ dọc phố và có người đứng/ngồi cạnh xe (`CAL-01`) | YOLO26 gán tuk-tuk thành `car` (conf 0.34–0.37); SAM nhận box `car` nên tạo mask mang nhãn sai | Đổi toàn bộ tuk-tuk sang `three_wheeler`, giữ box `person` riêng cho người lái/người đứng cạnh | Mục 2 & 4.3: Không dùng trực tiếp lớp `car`/`truck` của detector COCO cho xe 3 bánh; luôn đối chiếu lớp `three_wheeler` |
| `img_024.jpg` | Ảnh giao thông ban đêm nhìn từ trên cao, có minibus (`#216 bus` đánh dấu `can_review`) và nhiều cụm xe máy nhòe sáng (`CAL-02`) | YOLO26 tạo nhiều box nhỏ rời rạc và bỏ sót/nhầm lẫn ở các đám xe máy nhòe đèn | Gán `bus` cho xe khách/minibus (`#216`, `#203`), bao các cụm xe máy nhỏ/nhòe ở góc xa bằng `ignore_region` (`#192`–`#199`) | Mục 2 (`bus`, `ignore_region`) & 4.1.6: Minibus chở khách gán `bus`; cụm phương tiện nhòe chuyển động ban đêm không tách được cá thể gán `ignore_region` |
| `img_003.jpg` | Xe ba gác chở nông sản ở tiền cảnh có trẻ em và người lớn trên/sau xe; ở nền xa bên trái có phương tiện nhỏ bị khuất (`CAL-03`, `CAL-04`) | YOLOE bắt được `three_wheeler` chính nhưng phần nền xa bên trái bị lọt bẫy `missing` khi người duyệt chỉ tập trung vào xe lớn | Gán 1 box `three_wheeler` cho xe ba gác chính, tách từng box `person` cho người trên xe và người đi đường; bổ sung bước quét nền xa trước khi duyệt vật lớn | Mục 4.1.4 & Quy trình review: Quét ảnh theo thứ tự hậu cảnh → trung cảnh → tiền cảnh trước khi bật nhãn máy để tránh sót vật nhỏ ở xa |
| `img_013.jpg` | Tuk-tuk chiếm gần nửa ảnh ở tiền cảnh, phía xa bên phải có xe màu trắng bị gán box lỏng/sai lớp (`#76`, `CAL-04`) | Máy tạo box ở hậu cảnh (`img_013.jpg`) nhưng biên dưới/trái bị dư tạo thành bẫy `loose` bị lọt khi duyệt | Khi kiểm ảnh có vật thể lớn chiếm khung hình, bắt buộc phóng to kiểm tra các box phụ ở hậu cảnh để siết sát 4 cạnh (sai số ≤ 2 px) | Mục 4.1.1: Kiểm tra độ ôm sát 4 cạnh cho cả các box phương tiện phụ ở nền xa, không chỉ sửa vật chính ở trung tâm |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt 7 câu hỏi mở mục 6; bổ sung quy tắc phân biệt `three_wheeler` với `car`/`truck`, quy tắc `ignore_region` cho ảnh đêm dày đặc, quy tắc tách người lái khỏi mask SAM của xe, và quy tắc quét hậu cảnh trước tiền cảnh | Rút ra từ `CAL-01` (`img_004.jpg`), `CAL-02` (`img_024.jpg`), `CAL-03` (`img_003.jpg`), `CAL-04` (`img_013.jpg`) |
