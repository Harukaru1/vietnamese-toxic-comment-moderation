# Báo cáo dự án: Kiểm duyệt bình luận tiếng Việt

Tài liệu này tóm tắt dự án đã làm gì, vì sao làm như vậy, kết quả ra sao và rút ra được điều gì.
Số liệu chi tiết nằm trong `reports/`; phần lý thuyết nằm trong [`tai_lieu_hoc.md`](tai_lieu_hoc.md).

## 1. Bài toán

Phân loại một bình luận tiếng Việt trên mạng xã hội vào ba lớp:

| Lớp | Ý nghĩa (định nghĩa hiện tại, guideline v3.2) |
|---|---|
| clean | Không xúc phạm. Chê, góp ý, tranh luận đều là clean. Từ tục chỉ để nhấn mạnh ("hay vl") cũng là clean. |
| offensive | Mắng một người mà không miệt thị họ vì họ là ai ("m ngu"), hoặc chửi thề xả bực. |
| hate | Miệt thị vì vùng miền, quốc gia, giới tính, ngoại hình, nhóm fan...; đe doạ, trù ẻo. |

Mục đích sử dụng: xếp thứ tự ưu tiên cho người kiểm duyệt, không tự động xoá bình luận.

## 2. Điểm xuất phát và những gì đã thay đổi

Dự án bắt đầu từ một bản thử nghiệm trong luận văn tốt nghiệp. Bảng dưới đây so sánh cách làm cũ và mới.

| Vấn đề | Cách làm ban đầu | Cách làm trong dự án này | Lý do |
|---|---|---|---|
| Nhãn dữ liệu | Hai mô hình gán nhãn tự động; người chỉ xem lại chỗ hai mô hình bất đồng | Viết guideline, kiểm tra chéo bằng nhãn mù của người, đo Cohen's κ | Lỗi mà cả hai mô hình cùng mắc thì không ai xem lại |
| Chia train/test | Chia ngẫu nhiên | Bỏ câu trùng giữa train và test | 4,7% tập test trùng với train, làm điểm cao giả khoảng 0,03 |
| Tiền xử lý | Có bỏ stopword | Không bỏ stopword | Từ phủ định ("không", "chẳng") đổi nghĩa cả câu |
| Đánh giá | Một tập test cùng nguồn với train | Thêm hai tập mới thu năm 2026 | Điểm cùng nguồn không nói lên mô hình chạy thế nào trên dữ liệu mới |
| So sánh mô hình | So hai con số | Khoảng tin cậy bootstrap cho mọi phép so sánh | Chênh 0,01–0,03 thường chỉ là nhiễu |
| Triển khai | Không có | Hugging Face Hub, Streamlit, FastAPI + Docker, CI | Mô hình phải dùng được |

## 3. Các bước đã làm

### 3.1. Mô hình cơ sở
- TF-IDF + Logistic Regression / Linear SVM, và PhoBERT fine-tune.
- Trên tập test cùng nguồn, hai hướng ngang nhau (F1-macro khoảng 0,80). PhoBERT chỉ hơn khi gặp dữ liệu từ nguồn khác.

### 3.2. Làm sạch dữ liệu
- **Rò rỉ dữ liệu:** 178 câu trong test cũng có trong train. Sau khi bỏ, F1 giảm từ 0,820 xuống 0,793. Đây mới là con số thật.
- **Nhãn sai:** dùng cleanlab đánh dấu 5,5% nhãn nghi sai; xem tay 120 câu thì khoảng một phần ba sai rõ. Bỏ các câu này khỏi train.
- **Thêm dữ liệu:** bộ UIT-ViHSD (cùng ba lớp).

### 3.3. Tập test 2026 thứ nhất và guideline v3
- 1.000 bình luận mới (700 YouTube, 300 diễn đàn VOZ). Mô hình v2 chỉ đạt 0,59 trên tập này, so với 0,80 cùng nguồn.
- Phân tích lỗi cho thấy lớp offensive yếu vì nhãn không nhất quán: các câu có "vl" được gán 52% clean, 25% offensive, 23% hate.
- Guideline v3 chốt quy tắc: từ tục chỉ để nhấn mạnh là clean. Sửa 772 nhãn. Kết quả: 0,594 → 0,679.

### 3.4. Hai lần huấn luyện lại không có kết quả (v4, v5)
- **v4:** thêm 821 câu chọn bằng active learning. Không khác v3 (chênh −0,006, khoảng tin cậy [−0,068; +0,051]).
- **v5:** sửa thêm 163 nhãn và sửa một lỗi văn bản (từ "đéo" bị thay bằng "bad" trong khoảng 1.700 câu). Tốt hơn trên bộ câu thử viết tay, nhưng báo nhầm nhiều hơn trên dữ liệu thật. Không triển khai.
- Bài học: tập test thứ nhất chỉ có 48 câu hate, không đủ để phân biệt các chênh lệch nhỏ.

### 3.5. Tập test thứ hai và đổi định nghĩa
- Thu 12.000 bình luận YouTube mới, chọn 2.000 câu để gán nhãn (một phần ngẫu nhiên, một phần là câu mô hình báo độc hại hoặc có từ gợi ý).
- Tác giả gán mù 100 câu. Độ đồng thuận thấp (κ 0,41), chủ yếu ở một dạng: mắng một người cụ thể ("m ngu"). Guideline cũ xếp là hate, tác giả xếp là offensive.
- Tra cứu cho thấy cách của tác giả khớp với định nghĩa của VLSP 2019 và OLID. Đổi guideline thành v3.2 và gán lại. Độ đồng thuận sau khi đổi: κ 0,70 (ba lớp), 0,84 (độc hại / không).
- Trên tập này, v2 đến v5 ngang nhau và đều báo nhầm nhiều: cứ 10 câu báo độc hại thì khoảng 7 câu nhầm.

### 3.6. v6: cách làm nhỏ nhưng có hiệu quả
- Nguyên nhân báo nhầm: dữ liệu train cũ gán "độc hại" rộng hơn cách gán hiện tại (ví dụ lời chỉ trích nặng với kẻ phạm tội).
- Thay vì huấn luyện lại trên 52.000 câu cũ, lấy v3 và huấn luyện tiếp 3 vòng chỉ trên 1.739 câu gán theo chuẩn mới.
- Nửa còn lại của tập test thứ hai (968 câu) được giữ riêng, không bao giờ dùng để huấn luyện.

## 4. Kết quả chính

Trên 968 câu giữ riêng, quy đổi về tỉ lệ bình luận thực tế (khoảng 5% là độc hại):

| Mô hình | Precision độc hại | Recall độc hại | F1 độc hại | F1 offensive | F1 hate |
|---|---:|---:|---:|---:|---:|
| v3 | 0,28 | 0,45 | 0,35 | 0,26 | 0,17 |
| v5 | 0,28 | 0,42 | 0,34 | 0,27 | 0,17 |
| **v6 (đang triển khai)** | **0,45** | **0,50** | **0,47** | **0,51** | **0,28** |

- Chênh lệch F1 giữa v6 và v3: +0,125, khoảng tin cậy 95% từ +0,045 đến +0,223.
- Với ngưỡng 0,8 trên xác suất độc hại, v6 đạt precision 0,62 và recall 0,36. App dùng ngưỡng này cho hàng "đề xuất ẩn".
- Đánh đổi: trên tập 2026 thứ nhất, F1 độc hại của v6 là 0,58 so với 0,63 của v3.

## 5. Sản phẩm

- **Mô hình:** Hugging Face Hub, mỗi phiên bản một nhánh (v2 đến v6).
- **App Streamlit:** kiểm tra một bình luận, một đoạn văn (tô màu từng câu), nhiều bình luận, và cả phần bình luận của một video YouTube.
- **Góp ý ẩn danh:** mỗi lượt 👍/👎 được lưu vào một dataset riêng tư để dùng cho các đợt gán nhãn sau. Không lưu tên hay IP.
- **API:** FastAPI + Docker. Kiểm thử tự động bằng GitHub Actions.

## 6. Hạn chế

- Precision 0,45 nghĩa là mô hình chỉ nên dùng để gợi ý cho người kiểm duyệt.
- Mô hình đọc từng bình luận riêng lẻ, không có ngữ cảnh.
- Nhãn do một LLM và một người gán; độ đồng thuận được đo nhưng không hoàn hảo.
- Phân biệt hate với offensive còn yếu: mới có 78 câu hate theo định nghĩa hiện tại.
- Hai lần huấn luyện v6 với cùng dữ liệu cho precision 0,45 và 0,53: kết quả còn dao động giữa các lần chạy.

## 7. Bài học

1. **Kiểm tra dữ liệu trước khi chỉnh mô hình.** Rò rỉ, nhãn không nhất quán và lỗi văn bản ảnh hưởng nhiều hơn mọi thay đổi về mô hình.
2. **Điểm trên tập test cùng nguồn dễ gây ảo tưởng.** 0,80 cùng nguồn chỉ còn 0,59 trên dữ liệu mới.
3. **Định nghĩa nhãn là một phần của mô hình.** Mỗi lần kiểm tra chéo đều dẫn đến một lần sửa guideline, và lần sửa cuối tạo ra cải thiện lớn nhất.
4. **Tập test nhỏ không phân biệt được cải thiện nhỏ.** Phải có khoảng tin cậy trước khi kết luận.
5. **Kết quả âm tính cũng đáng báo cáo.** v4 và v5 không được triển khai, và chính chúng chỉ ra rằng vấn đề nằm ở nhãn cũ.
6. **Ít dữ liệu đúng chuẩn có thể hơn nhiều dữ liệu lệch chuẩn.** 1.739 câu gán đúng định nghĩa giúp nhiều hơn việc sửa lẻ tẻ trong 52.000 câu cũ.

## 8. Hướng phát triển

- Gán thêm ví dụ hate theo guideline v3.2. Phép thử với một nửa dữ liệu cho thấy lớp hate vẫn tăng khi có thêm dữ liệu, còn phát hiện độc hại nói chung thì đã chững.
- Huấn luyện vài lần và lấy trung bình dự đoán để giảm dao động.
- Mô hình có ngữ cảnh (bình luận gốc + câu trả lời).
- Nhận diện nói lái ("mẹo mày bé").
