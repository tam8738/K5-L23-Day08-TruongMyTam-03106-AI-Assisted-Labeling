# Thông tin học viên

- Họ và tên: Trương Mỹ Tâm
- MSHV: 03106
- Lớp (L2 / L3 / khác): L23
- Tài khoản GitHub: tam8738
- Repo bài nộp (URL): https://github.com/tam8738/K5-L23-Day08-TruongMyTam-03106-AI-Assisted-Labeling
- Commit chốt bài (hash): 0a232b9 (part1: 27d2874, part2-3: dcca28b, part4: af2aead, part5: 0a232b9)
- Máy: Windows 10/11 (CPU local, không có GPU rời) + Google Colab GPU CUDA (NVIDIA Tesla T4)
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 247/509 = 48.5% / 45/509 = 8.8% / 99/509 = 19.4% (mẫu số: 509 box máy); added: 9/302 = 3.0% (mẫu số: 302 đối tượng sau duyệt)
- So với gold: precision / recall của **máy** 40/63 = 63.5% / 40/46 = 87.0% (mẫu số: 63 box máy dự đoán và 46 đối tượng gold)
- Bắt bẫy: 10/12 = 83.3%
- Thời gian: gán tay 13.93 phút/ảnh (55.7 phút / 4 ảnh); duyệt có pre-label 3.73 phút/ảnh (89.6 phút / 24 ảnh)
- Lỗi QC: 66.7% theo ảnh (4/6 ảnh), 4.8% theo đối tượng (4/84 đối tượng)
- **Quyết định:** Rework — lý do một dòng: Tỉ lệ ảnh có lỗi trong mẫu QC là 4/6 = 66.7% vượt ngưỡng 2% (còn lỗi box lỏng/thừa và sót vật nhỏ ở hậu cảnh) nên cần sửa lại trước khi Release.
