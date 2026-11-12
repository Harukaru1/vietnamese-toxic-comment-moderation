# Tài liệu học: lý thuyết, thuật toán và lý do lựa chọn

Mỗi mục trả lời bốn câu: nó là gì, hoạt động thế nào, vì sao dự án dùng nó, và xem ở file nào.
Kết quả tổng thể nằm trong [`bao_cao_du_an.md`](bao_cao_du_an.md).

## 1. Tiền xử lý văn bản

**Là gì.** Đưa bình luận về một dạng thống nhất trước khi đưa vào mô hình.

**Các bước** (`src/toxic_moderation/preprocess.py`, hàm `clean_text`):
1. Chuẩn hoá Unicode NFC: cùng một chữ có dấu có thể được mã hoá theo hai cách; NFC đưa về một cách.
2. Viết thường, thay URL và @tên bằng khoảng trắng hoặc một ký hiệu chung.
3. Bỏ emoji.
4. Rút gọn ký tự lặp: "vlllll" thành "vl".
5. Tách từ bằng pyvi: "học sinh" thành "học_sinh". PhoBERT được huấn luyện trên văn bản đã tách từ nên bước này bắt buộc.

**Vì sao.**
- Không bỏ stopword: từ phủ định như "không", "chẳng" đổi nghĩa cả câu.
- Không dùng từ điển teencode: thử nghiệm cho thấy không tăng điểm, vì n-gram ký tự và subword đã bao phủ phần lớn biến thể chính tả.
- Bình luận chỉ có emoji thành chuỗi rỗng sau khi làm sạch. Mô hình sẽ gán đại một nhãn cho chuỗi rỗng, nên hàm `has_text` chặn trước và trả về clean.

## 2. TF-IDF và mô hình tuyến tính

**TF-IDF là gì.** Biến mỗi văn bản thành một vector số. Mỗi chiều ứng với một từ hoặc cụm ký tự.
- TF (term frequency): từ đó xuất hiện bao nhiêu lần trong văn bản.
- IDF (inverse document frequency): từ càng hiếm trong toàn bộ dữ liệu thì trọng số càng cao.
- Từ xuất hiện ở mọi nơi ("là", "và") có trọng số thấp; từ đặc trưng ("ngu") có trọng số cao.

**N-gram.** Dự án dùng song song hai loại đặc trưng: từ đơn và cặp từ liền nhau, cùng với các chuỗi 2–5 ký tự liên tiếp. Chuỗi ký tự chịu được lỗi chính tả và viết tắt ("nguuu", "ngoo").

**Logistic Regression và Linear SVM.** Cả hai học một trọng số cho mỗi chiều rồi cộng lại để ra điểm cho từng lớp.
- Logistic Regression tối ưu xác suất dự đoán đúng.
- Linear SVM tối ưu khoảng cách giữa các lớp (margin).

**Vì sao dùng.** Làm mô hình cơ sở: nhanh, rẻ, dễ giải thích. Nếu mô hình lớn không hơn mô hình cơ sở thì không đáng dùng. Trong dự án, TF-IDF + SVM ngang PhoBERT trên tập test cùng nguồn (khoảng 0,80), và chỉ thua trên dữ liệu nguồn khác.

**File.** `src/toxic_moderation/features.py`, `scripts/train_baseline.py`.

## 3. PhoBERT và fine-tuning

**Transformer / BERT là gì.** Một mạng nơ-ron đọc cả câu cùng lúc; cơ chế attention cho phép mỗi từ "nhìn" các từ khác để hiểu nghĩa theo ngữ cảnh. BERT được huấn luyện trước trên lượng lớn văn bản bằng cách che một số từ và đoán lại.

**PhoBERT.** BERT (kiến trúc RoBERTa) huấn luyện trước trên khoảng 20 GB văn bản tiếng Việt đã tách từ. Dự án dùng `vinai/phobert-base-v2`.

**Fine-tuning.** Thêm một lớp phân loại 3 đầu ra lên trên PhoBERT, rồi huấn luyện tiếp toàn bộ mô hình trên dữ liệu có nhãn với learning rate nhỏ.

**Các tham số và lý do** (`scripts/train_phobert.py`):

| Tham số | Giá trị | Lý do |
|---|---|---|
| Learning rate | 2e-5 (v1–v5), 1e-5 (v6) | Nhỏ để không phá kiến thức đã học trước; v6 nhỏ hơn nữa vì chỉ tinh chỉnh |
| Độ dài tối đa | 160 token (128 cho v6) | Bình luận thường ngắn; dài hơn chỉ tốn tính toán |
| Số vòng | tối đa 6 (v1–v5), 3 (v6) | Chọn vòng có F1-macro cao nhất trên tập validation |
| Chọn mô hình | theo F1-macro trên validation | Không bao giờ chọn theo tập test, nếu không điểm test mất ý nghĩa |

**Class weights.** Khi lớp mất cân bằng, có thể nhân lỗi của lớp hiếm với một trọng số lớn hơn. Ở v1, cách này làm mô hình báo offensive quá tay (precision 0,52); bỏ đi thì lên 0,62. Từ v2 đến v5 không dùng; v6 dùng lại với căn bậc hai của trọng số (nhẹ hơn) vì dữ liệu mới có tới 86% clean.

## 4. Các chỉ số đánh giá

- **Precision:** trong các câu mô hình báo độc hại, bao nhiêu phần trăm độc hại thật. Thấp nghĩa là báo nhầm nhiều.
- **Recall:** trong các câu độc hại thật, mô hình bắt được bao nhiêu phần trăm. Thấp nghĩa là bỏ sót nhiều.
- **F1:** trung bình điều hoà của precision và recall; chỉ cao khi cả hai cùng cao.
- **F1-macro:** tính F1 cho từng lớp rồi lấy trung bình, mỗi lớp có trọng số bằng nhau.

**Vì sao không dùng accuracy.** Khoảng 95% bình luận là clean. Một mô hình luôn đoán clean đạt accuracy 95% nhưng vô dụng.

**Vì sao chỉ số chính là độc hại / không độc hại.** Hai người gán nhãn đồng ý với nhau nhiều nhất ở mức này (κ 0,84), còn ranh giới hate / offensive thì kém hơn (κ 0,70). Chỉ số không nên phụ thuộc vào ranh giới mà chính con người còn bất đồng.

**File.** `src/toxic_moderation/evaluate.py`, `scripts/evaluate_model.py`, `scripts/eval_heldout.py`.

## 5. Rò rỉ dữ liệu (data leakage)

**Là gì.** Thông tin của tập test lọt vào tập train. Ở đây là các câu giống hệt nhau nằm ở cả hai tập, phần lớn là bình luận spam được đăng nhiều lần.

**Hậu quả.** Mô hình "nhớ" câu đó nên điểm test cao giả. Bỏ 178 câu trùng làm F1 giảm từ 0,820 xuống 0,793.

**Cách xử lý.** So khớp sau khi làm sạch văn bản (để "Ngu VL!!!" và "ngu vl" được coi là một), rồi bỏ câu trùng khỏi train. `scripts/make_clean_splits.py`.

## 6. Confident learning (cleanlab)

**Là gì.** Phương pháp tìm nhãn có khả năng sai trong dữ liệu.

**Cách hoạt động.**
1. Chia dữ liệu thành 5 phần. Huấn luyện trên 4 phần, dự đoán xác suất cho phần còn lại. Lặp lại để mọi câu đều có dự đoán từ một mô hình chưa từng thấy nó (out-of-fold).
2. Nếu mô hình rất chắc một câu thuộc lớp A trong khi nhãn ghi lớp B, câu đó bị đánh dấu nghi sai.

**Vì sao dùng và dùng thế nào.** Dữ liệu ban đầu do mô hình gán nhãn tự động nên chắc chắn có nhiễu. Cleanlab đánh dấu 5,5%; xem tay 120 câu thì một phần ba sai rõ. Các câu nghi sai chỉ bị bỏ khỏi train, không bỏ khỏi test, vì bỏ khỏi test sẽ làm bài kiểm tra dễ đi.

**File.** `scripts/find_label_issues.py`.

## 7. Guideline gán nhãn và Cohen's κ

**Guideline.** Văn bản quy định cách gán nhãn, kèm ví dụ cho các trường hợp khó. `docs/annotation_guidelines.md`.

**Cohen's κ là gì.** Đo mức đồng thuận giữa hai người gán nhãn, đã trừ đi phần trùng do ngẫu nhiên.
- κ = (tỉ lệ trùng thực tế − tỉ lệ trùng ngẫu nhiên) / (1 − tỉ lệ trùng ngẫu nhiên).
- κ = 1 là trùng hoàn toàn, κ = 0 là không hơn đoán mò. Thường coi trên 0,6 là khá, trên 0,8 là tốt.

**Vì sao không dùng tỉ lệ trùng.** Khi 85% câu là clean, hai người gán bừa clean cho mọi câu cũng trùng nhau 85%.

**Quy trình trong dự án.** Một LLM gán toàn bộ. Tác giả gán một mẫu mà không nhìn nhãn của LLM (gán mù). Tính κ, bàn các câu lệch, sửa guideline, gán lại. Bốn lần làm như vậy đều dẫn đến một lần sửa guideline (v3, v3.1, hai quy tắc bổ sung, v3.2).

**File.** `scripts/agreement.py`.

## 8. Bootstrap và khoảng tin cậy

**Vấn đề.** Mô hình A đạt 0,647, mô hình B đạt 0,641. A có thật sự hơn B không, hay chỉ do may rủi của tập test?

**Bootstrap.** Lấy mẫu lại tập test có hoàn lại (một câu có thể được chọn nhiều lần) để tạo 2.000 tập test "giả" cùng kích thước. Tính chỉ số trên mỗi tập. Khoảng chứa 95% các giá trị ở giữa là khoảng tin cậy 95%.

**Bootstrap ghép cặp.** Khi so hai mô hình, dùng cùng một mẫu lấy lại cho cả hai rồi tính hiệu. Nếu khoảng tin cậy của hiệu chứa số 0, chưa thể nói mô hình nào hơn.

**Trong dự án.**
- v4 so với v3: [−0,068; +0,051], chứa 0, nên không triển khai v4.
- v6 so với v3: [+0,045; +0,223], không chứa 0, nên cải thiện là thật.

**File.** `scripts/bootstrap_compare.py`, `scripts/eval_heldout.py`.

## 9. Active learning

**Là gì.** Thay vì gán nhãn ngẫu nhiên, chọn những câu có ích nhất cho mô hình để gán.

**Các cách chọn đã dùng** (`scripts/build_active_pool.py`):
- Câu mô hình không chắc nhất (xác suất cao nhất còn thấp).
- Câu có từ gợi ý độc hại mà mô hình báo clean (nghi bỏ sót).
- Câu mô hình báo độc hại (nghi báo nhầm).
- Một phần ngẫu nhiên, để dữ liệu không lệch hẳn về trường hợp khó.

**Kết quả.** v4 không tốt hơn. Vài trăm câu mới không thắng được hàng chục nghìn câu cũ gán theo quy tắc khác. Active learning chỉ có ích khi phần dữ liệu còn lại nhất quán với nó.

## 10. Lấy mẫu phân tầng và trọng số

**Vấn đề.** Chỉ khoảng 5% bình luận là độc hại. Lấy ngẫu nhiên 2.000 câu thì chỉ có khoảng 100 câu độc hại, quá ít để đo.

**Cách làm.** Chia kho 12.000 câu thành các nhóm: mô hình báo độc hại, có từ gợi ý, còn lại. Lấy nhiều hơn ở hai nhóm đầu. Tập test nhờ đó có 249 câu độc hại.

**Hệ quả và cách sửa.** Tập test không còn giống thực tế. Để ước lượng đúng, mỗi câu được gán trọng số bằng (số câu của nhóm trong kho) chia (số câu của nhóm được gán nhãn). Câu thuộc nhóm bị lấy ít sẽ "đại diện" cho nhiều câu hơn. Đây là trọng số nghịch đảo xác suất được chọn (inverse-probability weighting).

**File.** `scripts/build_testset_pool.py`, `scripts/tune_threshold.py` (hàm `cell` và `prf`).

## 11. Tập giữ riêng (held-out)

**Nguyên tắc.** Dữ liệu dùng để đo không được dùng để huấn luyện hay để chọn bất kỳ tham số nào.

**Trong dự án.** Tập test thứ hai được chia đôi cố định (`labels/testset2_split.csv`). Một nửa trở thành dữ liệu huấn luyện cho v6; nửa kia (968 câu) chỉ dùng để đo. Ngưỡng 0,8 được chọn trên tập validation riêng, rồi mới báo cáo trên nửa giữ riêng.

## 12. Tinh chỉnh tiếp một mô hình đã có (v6)

**Vấn đề.** 52.000 câu train cũ gán theo định nghĩa cũ. Sửa vài trăm nhãn trong đó (v4, v5) không thay đổi được mô hình.

**Cách làm.** Lấy v3 (đã học tiếng Việt mạng xã hội và khái niệm chung về độc hại) rồi huấn luyện tiếp 3 vòng, learning rate 1e-5, chỉ trên 1.739 câu gán theo định nghĩa mới.

**Vì sao hiệu quả.** Mô hình không phải học lại từ đầu, chỉ phải dịch chuyển ranh giới quyết định. Dữ liệu mới tuy ít nhưng không bị dữ liệu cũ lấn át.

**Rủi ro.** Mô hình có thể quên bớt kiến thức cũ (catastrophic forgetting). Vì vậy phải đo lại trên tập test cũ: F1 độc hại giảm từ 0,63 xuống 0,58, chấp nhận được so với phần tăng trên dữ liệu mới.

**Phép thử đường cong học.** Huấn luyện với một nửa dữ liệu mới: F1 độc hại gần như không đổi (0,49 so với 0,47–0,51), còn F1 hate thấp hơn (0,23 so với 0,28). Kết luận: thêm dữ liệu chủ yếu giúp lớp hate.

**File.** `notebooks/train_phobert_v6_kaggle.ipynb`, `reports/testset2_heldout.json`.

## 13. Ngưỡng quyết định

**Là gì.** Mô hình cho ra xác suất. Mặc định chọn lớp có xác suất cao nhất, nhưng có thể chỉ báo độc hại khi xác suất vượt một ngưỡng.

**Đánh đổi.** Ngưỡng cao: ít báo nhầm hơn, bỏ sót nhiều hơn.

| Ngưỡng trên P(độc hại) của v6 | Precision | Recall |
|---|---:|---:|
| 0,5 | 0,44 | 0,55 |
| 0,8 | 0,62 | 0,36 |

**Trong app.** Từ 0,8 trở lên: "đề xuất ẩn". Từ 0,4 đến 0,8: "cần người xem lại". Dưới 0,4: để nguyên. Hai hàng đầu chiếm khoảng 7% bình luận và chứa 59% số câu độc hại.

**Lưu ý.** Với v3, không ngưỡng nào giúp được: mô hình sai một cách tự tin. Ngưỡng chỉ có ích khi xác suất của mô hình phản ánh đúng độ chắc chắn.

## 14. Xử lý đoạn văn

**Vấn đề.** Mô hình học trên bình luận ngắn. Một đoạn dài có một câu xúc phạm lẫn trong nhiều câu bình thường dễ bị bỏ sót.

**Cách làm** (`src/toxic_moderation/segment.py`). Tách đoạn thành câu, phân loại từng câu. Nhãn của đoạn là nhãn nặng nhất trong các câu mà mô hình chắc từ 60% trở lên; câu kém chắc hơn chỉ bật cờ "cần xem lại".

## 15. Triển khai

| Thành phần | Dùng để | File |
|---|---|---|
| Hugging Face Hub | Lưu mô hình, mỗi phiên bản một nhánh; đổi phiên bản bằng một dòng cấu hình | `src/toxic_moderation/predictor.py` |
| Streamlit | Giao diện demo | `streamlit_app/streamlit_app.py` |
| FastAPI + Docker | Dịch vụ REST cho hệ thống khác gọi | `app/main.py`, `Dockerfile` |
| CommitScheduler | Ghi góp ý ẩn danh vào dataset riêng tư, đẩy lên định kỳ | `src/toxic_moderation/feedback_store.py` |
| YouTube Data API | Lấy bình luận công khai, không lấy tên người viết | `src/toxic_moderation/youtube.py` |
| pytest + GitHub Actions | Tự chạy kiểm thử mỗi lần đẩy code | `tests/`, `.github/workflows/ci.yml` |

**Vì sao tách nhãn khỏi dữ liệu.** Nội dung bình luận không được công khai (giấy phép và quyền riêng tư). Các file trong `relabels/` và `labels/` chỉ chứa chỉ số dòng hoặc id cùng với nhãn, nên có thể đưa lên GitHub để người khác kiểm tra mà không lộ dữ liệu.

## 16. Câu hỏi phỏng vấn có thể gặp

- **Vì sao F1 chỉ 0,47?** Vì đo trên dữ liệu mới, tỉ lệ độc hại thật khoảng 5%, và theo định nghĩa chặt. Trên tập test cùng nguồn con số là 0,80, nhưng con số đó không phản ánh thực tế.
- **Làm sao biết v6 tốt hơn thật?** Bootstrap ghép cặp trên 968 câu chưa từng dùng để huấn luyện: khoảng tin cậy của chênh lệch không chứa 0.
- **Vì sao không huấn luyện lại từ đầu với nhãn mới?** Chỉ có 1.739 câu nhãn mới. Gán lại 52.000 câu cũ quá tốn; tinh chỉnh tiếp từ v3 tận dụng được những gì nó đã học.
- **Điểm yếu lớn nhất?** Lớp hate, và việc không có ngữ cảnh.
- **Nếu có thêm thời gian?** Gán thêm ví dụ hate, trung bình nhiều lần huấn luyện, mô hình có ngữ cảnh.
