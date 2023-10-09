# extract-from-id-card-and-verify-face
Trích xuất thông tin trên thẻ căn cước công dân và xác thực khuôn mặt

Ta sẽ trích 3 trường thông tin quan trọng trên thẻ CCCD: Mã ID, họ tên và ngày sinh. Để có thể trích xuất thông tin trên thẻ CCCD, ta thực hiện sử dụng một thẻ CCCD làm ảnh mẫu, và xoay tất cả các ảnh CCCD cần đọc theo ảnh mẫu đó, dựa trên việc ghép các keypoint sử dụng SIFT và BFMatcher. Sau khi xoay ảnh, ta cắt vùng ảnh chứa trường thông tin theo tọa độ có sẵn và sử dụng OCR để đọc.

Kết quả đạt được trên 314 ảnh thẻ CCCD tự thu thập (loại bỏ các trường hợp ảnh xấu):
- Số thẻ CCCD đọc đúng cả 3 trường thông tin: 235/255 - 92.1%
- Số trường thông tin ID đọc đúng: 262/262 - 100%
- Số trường thông tin họ tên đọc đúng: 286/310 - 92.3%
- Số trường thông tin ngày sinh đọc đúng: 288/295 - 97.6%

Đi sâu hơn về trường thông tin ID, ta tự xây dựng 2 cách đọc: sử dụng CNN và sử dụng CRNN.
Đối với CNN, ta phân tách từng chữ số và xây model phân loại các chữ số. Kết quả sẽ được ghép từ 12 ảnh chữ số.

Kết quả đạt được trên 262 ảnh ID:
- Detect chữ số: 262/262 ảnh được detect đầy đủ 12 chữ số.
- Độ chính xác 261/262 - 99.6%
- Tốc độ chạy: 1.69s/ảnh.

Đối với CRNN, ta sinh ảnh để train và xây dựng model CRNN để đọc.

Kết quả đạt được trên 262 ảnh ID:
- Độ chính xác 255/262 - 97.3%
- Tốc độ chạy: 0.02s/ảnh.

Xác thực khuôn mặt sẽ thực hiện các bước sau:
- Phát hiện khuôn mặt bằng MTCNN.
- Loại bỏ các trường hợp không đủ điều kiện (ảnh nghiêng đầu, ảnh nhiều hơn 1 khuôn mặt, ảnh đeo khẩu trang).
- Encoding khuôn mặt bằng FaceNet.
- So sánh 2 vector embedding khuôn mặt: ảnh chụp chân dung và ảnh mặt trên CCCD.

Kết quả đạt được:
- Số đối tượng: 6 người
- Tổng số ảnh chụp (không tính CCCD): 70
- Số trường hợp phân loại sai ảnh đeo khẩu trang: 0/6 ảnh đeo khẩu trang, 2/64 ảnh không đeo khẩu trang.
- Số trường hợp xác định nghiêng mặt: 150
- Số trường hợp xác định không nghiêng mặt (đủ điều kiện xác thực): 270
- Số trường hợp False Rejection: 3/45
- Số trường hợp False Acceptance: 0/255
- Tốc độ trung bình mỗi lần thử: 2.25s
