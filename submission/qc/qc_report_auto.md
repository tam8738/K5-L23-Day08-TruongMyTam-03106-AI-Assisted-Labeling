# Báo cáo QC tự động (nháp)

> **Gợi ý** do script tính; bạn xác nhận. Script chỉ đếm theo phiếu kiểm bằng mắt người.

## Mẫu
- 6 ảnh (1 ca khó + 5 ngẫu nhiên), 84 đối tượng.
- taxonomy v1.0

## Số đo
| Chỉ số | Giá trị | Mẫu số |
|---|---|---|
| Lỗi **theo đối tượng** | 4/84 = 4.8% | 84 đối tượng trong mẫu |
| **Ảnh có lỗi** (dùng cho ngưỡng) | 4/6 = 66.7% | 6 ảnh trong mẫu |
| Máy: accepted / edited / deleted | 247/509 = 48.5% / 45/509 = 8.8% / 99/509 = 19.4% | 509 box máy |
| Máy: added | 9/302 = 3.0% | 302 đối tượng sau duyệt |
| Bắt lỗi bẫy | 10/12 = 83.3% | 12 bẫy |

## Top 3 loại lỗi
| Loại lỗi | Số lỗi | Ảnh ví dụ |
|---|---|---|
| sai_vi_tri_kich_thuoc | 2 | img_013.jpg, img_014.jpg |
| bo_sot | 1 | img_003.jpg |
| thua | 1 | img_006.jpg |

## Giới hạn thống kê
- 4 ảnh lỗi trong 6 ảnh mẫu: cận trên 95% (Wilson) của tỉ lệ ảnh lỗi thật là **90%**. Ngưỡng 2% không thể được **chứng minh** bằng mẫu nhỏ như vậy: hãy nói rõ giới hạn này trong báo cáo.

## Gợi ý quyết định
**REWORK**

- ảnh có lỗi 66.7% > 2%

Cây quyết định: Hold (guideline chưa rõ / dữ liệu hỏng) -> Rework (> ngưỡng, lỗi lặp, bắt bẫy / nhất quán thấp) -> Release.
