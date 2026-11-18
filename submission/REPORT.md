# Báo cáo Day 08 — 03106 - Trương Mỹ Tâm

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / car / bus / truck | YOLO26m (`yolo26m.pt`) | `0.30` (mặc định `--conf 0.3`) | Từ `r01` (`conf=0.25`, 215 box/8 ảnh) và `r02` (`conf=0.50`, 139 box/8 ảnh): nâng lên 0.5 làm mất nhiều `person` ở xa (131 → 87 box). Khi quét `threshold_sweep.py` trên 4 ảnh gold (`submission/qc/threshold_sweep.md`), `person` (tối đa recall 89% trên 19 vật gold) và `car` (tối đa recall 88% trên 8 vật gold) không đạt ngưỡng recall 90% ở `conf <= 0.5`, còn `bicycle` (1 vật), `bus` (0 vật), `truck` (1 vật) có `< 5` vật gold nên giữ nguyên ngưỡng mặc định `0.30`. |
| motorbike | YOLO26m (`yolo26m.pt`) | `0.35` | Theo `submission/qc/threshold_sweep.md` (nguồn từ `r05`), tại ngưỡng `0.35` lớp `motorbike` đạt recall `90.0%` (`9/10` vật gold) và precision `81.8%` (`9/11` box dự đoán), cân bằng tốt giữa việc giữ xe máy thật và lọc bớt box rác ở nền xa. |
| three_wheeler | YOLOE-26s (`yoloe-26s-seg.pt`, prompt: `'tuk-tuk'`, `'cyclo'`, `'cargo tricycle'`) | `0.15` | Ở `r01`–`r02`, YOLO26 hoàn toàn không có lớp `three_wheeler` trong danh mục COCO. Ở `r03` (`yoloe-s`, 8 box/8 ảnh pilot) và `r06` (125 box/36 ảnh), YOLOE bắt được tuk-tuk/xích lô/ba gác; quét ngưỡng trên gold (`threshold_sweep.md`) tại `0.15` đạt recall `100.0%` (`7/7` vật gold) và precision `77.8%` (`7/9` box). |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
  ```bash
  python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "motorbike=0.35,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
  ```
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: Trên Google Colab T4, tôi đã xin quyền truy cập gated repo `facebook/sam3`, cấu hình `HF_TOKEN` trong Colab Secrets và chạy tải xong toàn bộ **SAM 3 (`6.90 GB`, 12 files trong 46s)**. Tuy nhiên, khi chạy tiếp các ô code bên dưới trong notebook Colab (Ô 3 và Ô 7), do không để ý các dòng lệnh mẫu mặc định vẫn đang truyền `--segmenter sam --sam sam2.1_b.pt` (chứ không tự động chuyển sang `--model sam3`), nên thực tế toàn bộ bước tạo mask ở Part 2 (`r04`, `1.74s/ảnh` cho `img_004.jpg`, `img_013.jpg`) và Part 5 (`sam2_points.py`) được chạy bằng **SAM 2.1 base (`sam2.1_b.pt`)**. Kết quả `sam2.1_b.pt` vẫn đáp ứng tốt yêu cầu bài lab với tốc độ nhanh và hỗ trợ click điểm dương/âm (`--pos` / `--neg`).
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào?
  - Lỗi **lệch taxonomy / sai bản chất lớp** không thể sửa bằng đổi ngưỡng: YOLO26 chỉ có 80 lớp COCO nên luôn gọi tuk-tuk/ba gác là `car` hoặc `truck` (ví dụ trên `img_004.jpg` gọi tuk-tuk là `car 0.34`, `car 0.37`; trên `img_013.jpg` gọi tuk-tuk là `car 0.42`), còn YOLOE đôi khi gán nhầm thùng xe bán tải thành `three_wheeler` (`img_002.jpg`, conf `0.37`).
  - **Cách sửa:** Phải kết hợp model open-vocabulary (`YOLOE` với prompt `'tuk-tuk'`, `'cyclo'`, `'cargo tricycle'`) ưu tiên đứng trước `YOLO26` khi chạy `merge_prelabels.py` (`--iou 0.6`) để đè lên box `car`/`truck` trùng vị trí, sau đó người duyệt kiểm tra lại trong CVAT để đổi lớp hoặc xóa box dư.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  - **Điểm cao mà sai:** Trên `img_011.jpg` (`r01`/`r02`), phần rìa trái chiếc xe ba bánh/xe chở hàng bị YOLO26 gán `truck` với `conf = 0.77`; trên `img_013.jpg` (`r04`), chiếc tuk-tuk lớn bị YOLO26 gán `car` với `conf = 0.42` và chiếc xe van phía xa bị gán `car 0.72` nhưng box ăn lệch xuống xe máy phía trước.
  - **Điểm thấp mà đúng:** Trên `img_002.jpg` (`r03`), chiếc xe ba bánh đẩy hàng bên trái có `conf = 0.40` (và ở tập gold có những chiếc `three_wheeler` chỉ đạt `conf = 0.15`–`0.25`) nhưng là vật thật; nhờ hạ ngưỡng `three_wheeler=0.15` nên giữ được `7/7` vật trên tập gold.
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan:
  - Tập gold chỉ có **4 ảnh** (`img_001.jpg`, `img_012.jpg`, `img_019.jpg`, `img_036.jpg`) với tổng cộng **46 đối tượng tính điểm** (sau khi trừ `ignore_region`). Việc vừa dùng 4 ảnh này để chọn ngưỡng (`threshold_sweep.py`) vừa dùng chính 4 ảnh đó để báo cáo precision/recall (`review_stats.py`) là đánh giá **in-sample**, nên con số recall `87.0%` (`40/46`) là lạc quan hơn thực tế. Ngoài ra các lớp `bicycle` (`1` vật), `bus` (`0` vật), `truck` (`1` vật) có mẫu quá nhỏ (`< 5` vật) nên không thể ước lượng ngưỡng riêng đáng tin cậy.

## 2. AI error profile (từ `qc/review_stats.md`)

> Nguồn: `submission/qc/review_stats.md` và `submission/qc/review_stats.json` (tính trên 24 ảnh batch, ngưỡng `accept_iou >= 0.9`; đã loại **26 đối tượng nằm quanh các bẫy nhân tạo** để không làm méo chất lượng thật của máy).

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---:|---:|---|
| accepted | 247 | 247/509 = 48.5% | 509 box máy |
| edited | 45 | 45/509 = 8.8% | 509 box máy |
| deleted | 99 | 99/509 = 19.4% | 509 box máy |
| added | 9 | 9/302 = 3.0% | 302 đối tượng sau duyệt |

*(Lưu ý: phần còn lại `509 - (247 + 45 + 99) = 118` box máy nằm trong các vùng `ignore_region` do người duyệt vẽ nên script không tính là `deleted`).*

- Trong số `deleted + added`, có **3 trường hợp là đổi lớp** (box máy đặt đúng vị trí nhưng sai lớp,IoU ≥ 0.5), nên số vật máy **bỏ sót thật** trên lô batch xấp xỉ `added - đổi lớp = 9 - 3 = 6` đối tượng (`6/302 = 2.0%`).
- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)?
  - **`three_wheeler`** là lớp máy tệ nhất về tỉ lệ nhiễu: chỉ có `10/38 = 26.3%` box máy được `accepted` (10 accepted, 6 edited, **22 deleted**, 1 added). Nguyên nhân là để bắt hết tuk-tuk/xích lô/ba gác, ngưỡng `three_wheeler` được hạ xuống `0.15`, kéo theo nhiều box false positive của YOLOE trên xe tải nhỏ, xe đẩy hoặc mái hiên (ví dụ `img_002.jpg`, `img_004.jpg`, `img_013.jpg`), cộng với việc khung mái tuk-tuk khiến box máy thường bị lỏng phải chỉnh lại (`edited = 6`).
  - Xét về số lượng tuyệt đối bị xóa, **`person`** có `42` box `deleted` và `22` box `edited` (trên `186` box máy ngoài vùng ignore) do ở các ảnh chợ/phố đông (`img_005.jpg`, `img_018.jpg`, `img_021.jpg`), máy bắt cả người rất nhỏ hoặc bị che khuất > 80%.
- So với gold: precision / recall của **máy** (nguồn `submission/qc/review_stats.md`):
  - **Precision của máy:** `40/63 = 63.5%` (KTC 95%: `51%–74%`, mẫu số: `63` box máy dự đoán gồm `40 TP + 23 FP`).
  - **Recall của máy:** `40/46 = 87.0%` (KTC 95%: `74%–94%`, mẫu số: `46` đối tượng gold gồm `40 TP + 6 FN`).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: **`10/12 = 83.3%`** (KTC 95%: `55%–95%`, đạt ngưỡng yêu cầu `>= 80%` trong `taxonomy.json`). Chi tiết theo loại (`submission/qc/review_stats.md` & `traps_key.json`):
  - `wrong_class`: bắt **`3/3`** (`img_017.jpg`, `img_021.jpg`, `img_027.jpg` — đều là bẫy đổi `person` thành `bicycle`).
  - `extra`: bắt **`3/3`** (`img_029.jpg`, `img_031.jpg`, `img_034.jpg` — xóa hết các box giả `car`, `person`, `truck`).
  - `loose`: bắt **`2/3`** (sửa được trên `img_006.jpg` và `img_024.jpg`; **lọt 1 bẫy** trên `img_013.jpg`).
  - `missing`: bắt **`2/3`** (vẽ bù được trên `img_010.jpg` và `img_028.jpg`; **lọt 1 bẫy** trên `img_003.jpg`).
- Loại nào lọt và vì sao (bạn đã bỏ bước nào của checklist)?
  1. **`img_003.jpg` (`missing` — thiếu 1 box `three_wheeler` ở `[74.35, 176.3, 45.63, 50.28]` nền xa bên trái):** Khi mở ảnh, chiếc xe ba gác chở đầy chuối/nông sản ở tiền cảnh (`#13`) cùng nhiều người xung quanh thu hút toàn bộ sự chú ý. Tôi đã làm chưa kỹ **Bước 1 (Ẩn nhãn máy & quét toàn ảnh từ hậu cảnh ra tiền cảnh)** nên bỏ sót chiếc xe nhỏ ở phía xa bên trái đường.
  2. **`img_013.jpg` (`loose` — box `#76` ở `[799.8, 134.39, 144.01, 113.31]` góc trên bên phải bị nới lỏng biên trái/dưới):** Chiếc tuk-tuk màu xanh-đỏ chiếm gần trọn khung hình nên khi duyệt tới **Bước 4 (Siết box lỏng)**, tôi chỉ tập trung kiểm kỹ box tuk-tuk và 4 người ở trung tâm mà nhìn lướt qua chiếc xe màu trắng ở góc xa bên phải.
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt?
  - Trong quá trình làm thực tế, ở lần xuất đầu tiên khi tôi giữ `accepted` tới `366/509 = 71.9%`, tỉ lệ bắt bẫy chỉ đạt `7/12 = 58.3%` — đó là dấu hiệu điển hình của **thiên kiến tự động hóa (automation bias)**: thấy box máy có vẻ hợp lý là để nguyên và lướt nhanh.
  - Sau khi rà soát lại nghiêm ngặt theo checklist 4 bước (gom các cụm nhỏ `< 20 px` vào `ignore_region`, siết lại các box lệch để đạt `IoU >= 0.9`, xóa box thừa và tìm vật sót), tỉ lệ `accepted` giảm xuống `247/509 = 48.5%` (với `45/509` box được `edited` và `99/509` box `deleted`), đồng thời tỉ lệ bắt bẫy tăng lên **`10/12 = 83.3%`**. Điều này chứng minh: `accepted` thấp hơn nhưng bắt bẫy cao hơn phản ánh việc người duyệt **thực sự kiểm tra và chỉnh sửa hình học/ngữ nghĩa** thay vì bấm chấp nhận thụ động theo máy.

## 4. Máy có đáng dùng không?

- Thời gian (từ `submission/time_log.csv`):
  - **Gán tay từ đầu (`gold_manual`):** `55.7 phút / 4 ảnh = 13.93 phút/ảnh` (tổng 55 đối tượng trên 4 ảnh gold, bao gồm cả thời gian đọc kỹ guideline và chỉnh lại các box `< 20 px`).
  - **Duyệt có pre-label (`batch_review`):** `89.6 phút / 24 ảnh = 3.73 phút/ảnh` (tổng 345 đối tượng trên 24 ảnh batch).
  - **Thiên lệch cần tính đến:** Khi sang Part 4 tôi đã quen công cụ CVAT, thuộc hệ thống 8 lớp và quy tắc `< 20 px` nhờ làm 4 ảnh gold trước đó, nên tốc độ nhanh hơn một phần nhờ hiệu ứng học tập (learning effect), không hoàn toàn 100% do công của AI. Tuy nhiên mức giảm từ `13.93 phút/ảnh` xuống `3.73 phút/ảnh` (nhanh hơn **3.7 lần**, tiết kiệm ~73% thời gian mỗi ảnh) vẫn rất rõ rệt.
- Kết luận: **Có đáng dùng, nhưng chỉ dùng theo công thức kết hợp (hybrid) và bắt buộc có checklist kiểm định:**
  - Dùng **YOLO26m** cho các lớp phương tiện/người phổ thông (`person`, `motorbike`, `car`), nơi máy giữ được phần lớn box đúng (`motorbike` accepted `77/110 = 70.0%` ngoài ignore, `car` accepted `32/44 = 72.7%`).
  - Dùng **YOLOE-26s** ở ngưỡng thấp (`0.15`) như một bộ gợi ý vùng ứng viên (high-recall proposal) cho `three_wheeler`, nhưng người duyệt phải xác định trước rằng sẽ phải xóa khoảng `22/38 = 57.9%` box thừa.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. **Quy tắc 1 — Tắt hiển thị nhãn máy 5–10 giây đầu và đếm nhẩm vật ở hậu cảnh trước:** Khi mở mỗi ảnh trong CVAT, bấm phím `H` (hoặc biểu tượng con mắt) ẩn toàn bộ box `source=file`, quét mắt từ vùng xa/rìa ảnh vào trung tâm và đếm số `three_wheeler`/`truck`/`bus` trước khi bật nhãn lại. *(Cách kiểm tra: tỉ lệ bắt bẫy `missing` phải đạt `>= 2/3` và không bỏ sót phương tiện lớn ở rìa ảnh).*
  2. **Quy tắc 2 — Kiểm tra bắt buộc 100% box thuộc lớp nhạy cảm (`three_wheeler`, `bicycle`, `truck`):** Không bao giờ chấp nhận ngay box của 3 lớp này; phải phóng to kiểm tra xem `bicycle` có phải là `person` bị gán nhầm không và `three_wheeler` có phải là đuôi xe tải/ô tô không. *(Cách kiểm tra: bắt được `3/3 = 100%` bẫy `wrong_class` và `3/3 = 100%` bẫy `extra` trong `review_stats.md`).*
  3. **Quy tắc 3 — Chuẩn hóa cụm nhỏ `< 20 px` bằng `ignore_region` và siết 4 cạnh box ở rìa ảnh:** Với 6 ảnh mật độ cao (`img_005`, `img_018`, `img_021`, `img_024`, `img_028`, `img_029`), bao ngay cụm xe/người ở xa `< 20 px` bằng `ignore_region` trước, sau đó kiểm tra từng box còn lại sao cho biên lệch `<= 2 px` và `validate_labels.py --all` đạt **0 cảnh báo** box nhỏ. *(Cách kiểm tra: `validate_labels.py --all` báo 0 cảnh báo trên cả 4 ảnh gold và 24 ảnh batch).*

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---:|---|---|---|
| `img_013.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 click (`2 pos`: `400,480`, `600,110`; `1 neg`: `720,300`) | **Không gồm thân người lái chính** (nhờ điểm âm `720,300` đặt vào lưng áo xanh của tài xế), nhưng mask (`317,255 px = 45.4%` ảnh, 1 polygon) vẫn ăn nhẹ vào phần quần/chân người lái, một phần người ngồi sau và bánh trước xe máy dựng sát mũi tuk-tuk. | Khi chỉ dùng box tự động của YOLO26 (`r04`), mask gộp cả người ngồi và mang nhãn sai (`car`). Khi dùng `sam2_points.py`, phần khung hở của tuk-tuk làm mask dễ tràn sang người ngồi bên trong và xe máy kế bên nếu thiếu điểm âm. | `tuktuk_img013.jpg` |
| `img_002.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 click (`2 pos`: `480,580`, `520,415`; `1 neg`: `610,490` + box gợi ý `360,390,715,660`) | **Không gồm thân người lái áo xanh** (nhờ điểm âm `610,490` đặt vào lưng tài xế), nhưng mask (`51,240 px = 7.3%` ảnh, 1 polygon) vẫn ăn vào khoảng trống khoang hành khách phía sau nơi có người đi bộ đứng ở nền chợ. | Khi chỉ click điểm dương mà không có khung gợi ý (`--box`), cấu trúc rỗng giữa mui và thân tuk-tuk làm SAM không hội tụ được mask; khi thêm box + điểm âm thì tách được tài xế nhưng phần nền nhìn xuyên qua khung xe vẫn bị tính vào mask. | `tuktuk_img002.jpg` |
| `img_004.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 click (`2 pos`: `350,470`, `350,280`; `1 neg`: `465,360`) | **Không gồm người đứng cạnh xe** (`person 0.84` bên phải nhờ điểm âm `465,360`), mask (`44,588 px = 6.4%` ảnh, 1 polygon) bao trọn đầu xe, kính chắn gió và mui chiếc tuk-tuk màu xanh lá. | Vùng bóng tối dưới gầm bánh trước và phần giáp ranh với chiếc tuk-tuk phía sau làm biên mask hơi ăn nhẹ xuống mặt gạch dưới gầm. | `tuktuk_img004.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu?
  - **Quy tắc giúp ích rõ:** Mục 4.2.2 (không vẽ lỗ thủng nếu lỗ `< 5%` diện tích vật, lỗ lớn nhìn xuyên qua khung xe thì bỏ ra), Mục 4.2.3 (bóng đổ không thuộc vật, kính chắn gió thuộc vật) và Mục 4.2.4 (hai vật chạm nhau không chồng polygon, vùng chạm thuộc vật phía trước) giúp quyết định rõ cần đặt click dương trên mui + thân xe và click âm trên người lái / xe bên cạnh.
  - **Quy tắc còn thiếu trong v1.0 (đã bổ sung vào v1.1):** Guideline v1.0 chưa nói rõ khi tạo mask cho `three_wheeler` có khoang hở (tuk-tuk, xích lô, ba gác) thì phần người lái/hành khách hoặc hàng hóa chất cao nhìn thấy qua khung hở xử lý thế nào khi click SAM. Trong `guideline_v1.1_calibrated.md`, tôi đã chốt: kính xe thuộc mask xe nhưng người ngồi/hàng hóa rời nhô > 20% không thuộc mask xe — bắt buộc đặt điểm âm (`neg`) lên người lái.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: Trong bài này tôi thực hiện SAM bằng offline script (`scripts/prelabel.py --segmenter sam` ở Part 2 và `scripts/sam2_points.py` ở Part 5 trên Colab T4) thay vì bật Nuclio serverless trong CVAT Docker để tiết kiệm RAM cho máy local Windows. So với chạy bằng box tự động ở Part 2, cơ chế truyền điểm click dương/âm (`sam2_points.py`, mô phỏng trực tiếp SAM Interactor của CVAT) kiểm soát biên vật thể có khoang rỗng tốt hơn hẳn vì chủ động loại trừ được người lái.

## 6. Calibration (Part 6)

- Số quyết định: **4 quyết định** trong `submission/qc/calibration_decisions.csv` (`CAL-01` đến `CAL-04`, bao gồm đầy đủ các nguồn `loi_may`, `can_review`, `cau_hoi_mo`, `bay_lot`); quan trọng nhất là **`CAL-01`** (xử lý xung đột nhãn `three_wheeler` vs `car`/`truck` giữa YOLOE và YOLO26 trên `img_004.jpg`) và **`CAL-03`** (quy tắc tách box/mask giữa xe ba bánh, hàng hóa nhô ra và người ngồi trên xe ở `img_003.jpg`, `img_013.jpg`).
- Quy tắc mới đã vào `submission/guidelines/guideline_v1.1_calibrated.md`:
  1. Chốt toàn bộ 7 câu hỏi mở của v1.0 (phân biệt xe đạp điện vs xe máy điện theo bàn đạp; ngưỡng gom cụm `ignore_region` cho ảnh đêm nhòe sáng `img_024.jpg`; giới hạn hàng nhô 20%; quy tắc "thấy rõ đầu + vai" thay cho ước lượng 20% diện tích mơ hồ; đặt click âm lên người lái khi tạo mask SAM cho xe; không tự động duyệt theo ngưỡng `conf`).
  2. Bổ sung 4 ca khó mới có ảnh minh họa cụ thể (`img_004.jpg`, `img_024.jpg`, `img_003.jpg`, `img_013.jpg`) và cập nhật bảng lịch sử phiên bản `v1.1`.
- Nguyên nhân gốc: Kết hợp cả **3 yếu tố**:
  - **Máy (`loi_may`):** Giới hạn cố hữu của detector closed-set (YOLO26 không biết `three_wheeler`) và nhiễu false positive ở ngưỡng thấp của open-vocabulary (YOLOE).
  - **Guideline v1.0 mở (`cau_hoi_mo`, `can_review`):** Cố ý để mở 7 câu hỏi về biên giới hàng hóa, ảnh đêm dày đặc và lỗ hở trên mask xe ba bánh.
  - **Người duyệt (`bay_lot`):** Thiên kiến tập trung vào vật thể lớn ở tiền cảnh mà lướt nhanh các vật nhỏ/box phụ ở hậu cảnh (`img_003.jpg`, `img_013.jpg`).

## 7. QC và quyết định (Part 6)

- Mẫu (`submission/qc/qc_sample.csv` và `submission/qc/qc_report_auto.md`): **6 ảnh** trên tổng số 24 ảnh batch (`1` ảnh ca khó có tick `can_review` là `img_024.jpg` + `5` ảnh ngẫu nhiên với `seed = 42` là `img_003.jpg`, `img_006.jpg`, `img_013.jpg`, `img_014.jpg`, `img_033.jpg`), tổng cộng **84 đối tượng** (không tính `ignore_region`). Người kiểm: `03106` (**hạn chế:** người QC cũng chính là người duyệt nên có thể lặp lại điểm mù của chính mình).
- Lỗi theo **đối tượng**: **`4/84 = 4.8%`** · theo **ảnh**: **`4/6 = 66.7%`** (ngưỡng Release yêu cầu ảnh lỗi `<= 2%`).
- Top lỗi (từ `submission/qc/qc_review.csv` & `qc_report_auto.md`) + một ảnh ví dụ mỗi loại:
  1. **`sai_vi_tri_kich_thuoc` (`2/4` lỗi, chiếm `50.0%` số lỗi):** Ví dụ `img_013.jpg` (box `#76` ở nền phải bị lỏng biên trái/dưới — chính là bẫy `loose` bị lọt) và `img_014.jpg` (box `#79 motorbike` ở góc phải dưới mở rộng dư sang phần mặt đường trống bên trái).
  2. **`bo_sot` (`1/4` lỗi):** Ví dụ `img_003.jpg` (bỏ sót 1 chiếc `three_wheeler` nhỏ ở nền xa bên trái — khớp với bẫy `missing` bị lọt).
  3. **`thua` (`1/4` lỗi):** Ví dụ `img_006.jpg` (box `#50 motorbike` bao trùm lên cả cụm xe máy đỗ sát nhau trong khi xe đầu dãy đã có box `#45 motorbike`).
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`):
  - Với `4/6` ảnh có lỗi, cận trên khoảng tin cậy 95% (Wilson) của tỉ lệ ảnh lỗi thực sự lên tới **`90%`**. Ngay cả trong trường hợp lý tưởng có `0/6` ảnh lỗi thì với mẫu chỉ `6` ảnh, cận trên 95% Wilson vẫn là ~`39%`, hoàn toàn **không thể chứng minh thống kê** rằng tỉ lệ lỗi thật đạt ngưỡng `<= 2%`. Muốn kiểm định ngưỡng `2%` một cách chặt chẽ cần cỡ mẫu lớn hơn nhiều (hàng trăm ảnh).
- Script kiểm được / không kiểm được:
  - **Script kiểm được (`validate_labels.py`, `review_stats.py`, `check_submission.py`):** Kiểm tra cú pháp JSON COCO, tên lớp hợp lệ theo `taxonomy.json`, tọa độ box không tràn khỏi ảnh, chiều cao box `< 20 px`, đủ số lượng ảnh (4 gold, 24 batch), khớp các bẫy nhân tạo đã biết (`traps_key.json`) và tính IoU giữa pre-label với nhãn cuối.
  - **Script KHÔNG kiểm được:** Không biết box có thực sự ôm sát biên vật thể ngoài đời hay không (trừ các box bẫy đã có tọa độ gốc), không phát hiện được lỗi **cả máy và người cùng bỏ sót** trên ảnh batch (vì không có nhãn gold cho 24 ảnh batch), và không tự biết một box lớn bao trùm 3 chiếc xe máy (`#50` trên `img_006.jpg`) là đúng hay thừa nếu không có mắt người kiểm lại ở bước QC.

**Quyết định: Rework**, ngày `2026-11-18`. Lý do (đi theo cây quyết định Hold → Rework → Release):
1. **Xét Hold:** Guideline đã được chuẩn hóa và chốt đầy đủ các câu hỏi mở thành bản `v1.1` (`guideline_v1.1_calibrated.md`), dữ liệu 24 ảnh không bị hỏng/mờ không dùng được (`guideline_gap = 0`, `data_problem = 0`) → **Không Hold**.
2. **Xét Rework vs Release:** Dù tỉ lệ bắt bẫy trên cả lô đạt `10/12 = 83.3%` (`>= 80%`), nhưng khi kiểm tra mẫu QC 6 ảnh thì tỉ lệ ảnh có lỗi là **`4/6 = 66.7% > 2%`** (và tỉ lệ lỗi theo đối tượng là `4/84 = 4.8%`), vẫn còn 2 bẫy lọt (`img_003.jpg` missing, `img_013.jpg` loose) và lỗi box lỏng/chồng ở `img_006.jpg`, `img_014.jpg` → **Không đủ điều kiện Release, bắt buộc chọn Rework**.

Nếu Rework / Hold, việc cần làm tiếp:
- Mở lại CVAT trên Task `batch-review` để sửa 4 ảnh phát hiện lỗi trong mẫu QC (`img_003.jpg`: thêm box `three_wheeler` ở nền xa bên trái; `img_006.jpg`: xóa/thu gọn box `#50 motorbike` bị trùng; `img_013.jpg`: siết lại box `#76` ở nền phải; `img_014.jpg`: thu gọn biên trái của box `#79 motorbike`).
- Áp dụng quy tắc hậu cảnh của `guideline v1.1` để rà lại toàn bộ các box phương tiện phụ ở rìa ảnh trên 20 ảnh batch còn lại, xuất thành `submission/labels/qc_fixed.json`, chạy lại `build_release.py` và lấy mẫu QC vòng 2 trước khi phát hành.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không):
  - Có dùng trợ lý AI (OpenCode / Claude) để: hướng dẫn xử lý lỗi môi trường Windows Device Guard chặn `pip.exe` và DLL của `numpy` trên Python 3.13/3.14; hướng dẫn quy trình chạy các script kiểm tra (`validate_labels.py`, `threshold_sweep.py`, `merge_prelabels.py`, `inject_traps.py`, `review_stats.py`, `build_release.py`, `qc_sample.py`, `qc_report.py`) và tổng hợp/định dạng bảng số liệu báo cáo.
  - **Cách kiểm tra:** Mọi nhãn trong `gold_manual.json` và `batch_reviewed.json` đều do tôi trực tiếp vẽ/duyệt trên CVAT; mọi con số trong báo cáo được đối chiếu trực tiếp với đầu ra thật của script trong `submission/qc/threshold_sweep.md`, `submission/qc/review_stats.md`, `submission/qc/qc_report_auto.md` và `work/runs.csv`.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)?
  - **Không.** Tôi hoàn thành gán tay 4 ảnh gold trong CVAT, chạy `validate_labels.py --all`, ghi `time_log.csv` (`55.7` phút) và commit + push Part 1 (`commit 27d2874`) trước khi chạy bất kỳ model pre-label nào trên Colab.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt:
  - **Máy local:** Windows 10/11 (RAM ~24 GB, không có GPU NVIDIA rời), chạy CVAT bằng Docker local (`http://localhost:8080`) và chạy các script đo/kiểm tra trên môi trường Python local.
  - **Chạy model AI:** Dùng Google Colab với GPU **NVIDIA Tesla T4** (`torch 2.14.1+cu130`, `ultralytics 8.4.215`) rồi tải `colab_stage1.zip` và `colab_outputs.zip` về giải nén vào gốc repo.
  - **Vướng mắc cài đặt:** Máy Windows bật chính sách Application Control / Device Guard nên chặn `C:\Python314\Scripts\pip.exe` và chặn nạp DLL `_umath_linalg` của `numpy` trên bản Python cài mới; đã khắc phục bằng cách chuyển sang môi trường Python hợp lệ có sẵn trên máy và gọi qua `python -m pip`.
- Dùng cache `data/cache/` thay vì tự chạy?
  - **Không.** Toàn bộ kết quả YOLO26, YOLOE và SAM đều được chạy trực tiếp trên Google Colab T4 GPU (`work/runs.csv`: `r01` đến `r06`).
- Lỗi / bug script đã gặp (nếu có):
  - Không có bug trong mã nguồn script của repo; chỉ gặp vấn đề khác biệt cú pháp giữa PowerShell và Command Prompt (CMD) trên Windows khi xuống dòng bằng dấu `` ` `` hoặc dùng lệnh `Test-Path` / `source` trong CMD.
- Một điều bạn làm khác lần sau:
  - Ngay từ vòng duyệt đầu tiên trên CVAT, tôi sẽ thực hiện nghiêm ngặt việc **ẩn toàn bộ nhãn máy 10 giây đầu để quét hậu cảnh trước tiền cảnh**, thay vì để mắt bị hút ngay vào vật thể lớn ở giữa ảnh (như chiếc xe ba gác ở `img_003.jpg` hay chiếc tuk-tuk ở `img_013.jpg`) dẫn đến phải duyệt lại nhiều vòng mới nâng được tỉ lệ bắt bẫy từ `7/12` lên `10/12`. Ngoài ra, trên Colab sau khi đã mất công xin quyền và tải `facebook/sam3` (`6.90 GB`), tôi sẽ kiểm tra kỹ tham số `--model` ở các ô code tiếp theo để chạy thử trực tiếp `--model sam3` thay vì để nguyên lệnh mặc định `--sam sam2.1_b.pt`.
