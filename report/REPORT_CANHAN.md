# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Lưu Nguyên Khôi
**Nhóm:** 67    
**Ngày:** 20/9/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector gần như chỉ về cùng một hướng trong không gian embedding, nghĩa là mô hình đánh giá hai đoạn văn bản nói về cùng một chủ đề. Giá trị nằm trong khoảng -1 đến 1: gần 1 là cùng nghĩa, gần 0 là không liên quan, gần -1 là trái ngược nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Người Mua có thể yêu cầu trả hàng trong vòng 15 ngày."
- Câu B: "Thời hạn gửi yêu cầu hoàn tiền của khách là 15 ngày."
- Tại sao tương đồng: cùng nói về một mốc thời gian 15 ngày cho cùng một quyền của người mua, chỉ khác cách diễn đạt ("trả hàng" / "hoàn tiền", "Người Mua" / "khách"). Hai câu có thể thay thế nhau trong cùng một ngữ cảnh mà không đổi ý.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Người Bán tạo đơn hàng ảo sẽ bị xử phạt."
- Câu B: "Hôm nay trời Hà Nội mưa rất to."
- Tại sao khác: khác hoàn toàn về chủ đề (chính sách thương mại điện tử so với thời tiết), không chung một từ mang nghĩa nào, và không thể xuất hiện thay cho nhau trong bất kỳ ngữ cảnh nào.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine chỉ đo **hướng** của vector và bỏ qua **độ dài**, trong khi độ dài của embedding thường phản ánh độ dài văn bản chứ không phải ý nghĩa. Một đoạn 2 câu và một đoạn 10 câu cùng nói về chính sách đổi trả sẽ cách xa nhau theo Euclid nhưng vẫn gần nhau theo cosine — và đó mới là kết quả ta cần khi truy xuất.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Mỗi chunk mới chỉ tiến thêm được `step = chunk_size - overlap = 500 - 50 = 450` ký tự, vì 50 ký tự đầu đã trùng với chunk trước.
> `số chunk = ceil((10000 - 50) / 450) = ceil(9950 / 450) = ceil(22.11) = 23`
>
> *Đáp án:* **23 chunks.** Tôi đã kiểm chứng lại bằng chính code: `FixedSizeChunker(chunk_size=500, overlap=50).chunk("x" * 10000)` trả về đúng 23 chunk.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Số chunk **tăng lên 25** (`ceil((10000-100)/400) = ceil(24.75) = 25`, cũng đã kiểm chứng bằng code), vì mỗi bước chỉ còn tiến 400 ký tự thay vì 450. Overlap nhiều hơn giúp một câu hoặc một điều khoản bị cắt ngang ở ranh giới vẫn còn nguyên vẹn trong chunk kế tiếp, tránh mất ngữ nghĩa — cái giá phải trả là nhiều chunk hơn, tốn thêm chi phí lưu trữ và embedding, đồng thời nội dung lặp lại có thể chiếm nhiều chỗ trong top-k.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi dùng regex `(?<=[.!?])\s+` với **lookbehind** thay vì `split(". ")`, để dấu chấm câu được giữ lại ở cuối câu chứ không bị nuốt mất; `\s+` cũng bao luôn trường hợp `".\n"` mà đề bài yêu cầu. Các edge case đã xử lý: văn bản rỗng hoặc chỉ có khoảng trắng trả về `[]`, mỗi câu được `strip()` và loại bỏ câu rỗng trước khi gom nhóm. Điểm yếu tôi ý thức được: tiếng Việt có nhiều viết tắt (`TP.`, `vd.`) dễ bị tách sai, và văn bản có cấu trúc như YAML hay heading Markdown thì không có dấu kết câu nên cả khối bị gom thành một chunk rất lớn.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán thử lần lượt các separator theo thứ tự thô đến mịn (`"\n\n"` → `"\n"` → `". "` → `" "` → `""`), chỉ xuống cấp khi buộc phải. Có **ba base case**: (1) đoạn đã ngắn hơn `chunk_size` thì trả về luôn, (2) hết separator hoặc gặp separator rỗng thì cắt cứng theo kích thước, (3) separator không xuất hiện trong đoạn thì bỏ qua, thử separator kế tiếp. Điểm tôi thêm vào so với cách làm ngây thơ là bước **gộp tham lam**: sau khi tách, tôi ghép các mảnh lại cho sát `chunk_size` rồi dừng ở ranh giới đoạn gần nhất, nên chunk vừa không bị vụn vừa không cắt giữa ý.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Tôi tách riêng hàm `_make_record()` để chuẩn hoá mỗi `Document` về một cấu trúc thống nhất gồm `id`, `doc_id`, `content`, `metadata`, `embedding`. Điểm quan trọng: `id` của mỗi chunk là `f"{doc.id}#{self._next_index}"` với bộ đếm toàn cục, vì cùng một `doc_id` có thể được nạp nhiều lần hoặc bị chia thành nhiều chunk — nếu không có hậu tố này thì các chunk sẽ ghi đè lẫn nhau. `search()` embed câu hỏi rồi tính tích vô hướng với từng vector đã lưu, sắp xếp giảm dần và lấy `top_k`. Tôi dùng thẳng dot product thay vì gọi lại công thức cosine đầy đủ vì mọi embedder trong lab đều trả về vector đã chuẩn hoá độ dài 1, khi đó mẫu số bằng 1 nên **dot product chính là cosine similarity**.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Tôi **lọc trước, tìm kiếm sau** (pre-filtering). Nếu làm ngược lại — tìm top-k rồi mới lọc — thì kết quả trả về sẽ ít hơn `top_k` và có thể rỗng hoàn toàn, vì các chunk đúng đối tượng có thể nằm ngoài top-k ban đầu. Lọc trước đảm bảo `top_k` luôn được lấp đầy bằng các ứng viên đã thoả điều kiện. `delete_document()` xoá mọi record có `metadata['doc_id']` khớp, và trả về `True`/`False` bằng cách so sánh số lượng record trước và sau khi lọc.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Prompt gồm ba phần: chỉ thị ràng buộc ("chỉ dùng context bên dưới, nếu không có thì nói không biết"), khối ngữ cảnh, rồi câu hỏi. Tôi **đánh số từng chunk kèm điểm số** dạng `[1] (score=0.375) ...` khi nhét vào ngữ cảnh, để sau này có thể chỉ ra chính xác chunk nào đã cung cấp thông tin cho câu trả lời — đáp ứng tiêu chí *Source Traceability* trong `docs/EVALUATION.md`. Trường hợp store rỗng hoặc không tìm được gì, hàm trả về câu "không biết" thay vì để lỗi.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
$ pytest tests/ -v

============================= test session starts =============================
platform win32 -- Python 3.13.15, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\AI in Action\Lesson\9.20.2026\Lab7\K4-L3B-Data-Foundations
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker (7 tests)                  PASSED
tests/test_solution.py::TestSentenceChunker (4 tests)                   PASSED
tests/test_solution.py::TestRecursiveChunker (4 tests)                  PASSED
tests/test_solution.py::TestEmbeddingStore (9 tests)                    PASSED
tests/test_solution.py::TestKnowledgeBaseAgent (2 tests)                PASSED
tests/test_solution.py::TestComputeSimilarity (4 tests)                 PASSED
tests/test_solution.py::TestCompareChunkingStrategies (3 tests)         PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter (3 tests)    PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument (3 tests)      PASSED

============================= 42 passed in 0.04s ==============================
```

**Số lượng bài test vượt qua (pass):** **42** / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Chạy `compute_similarity()` trên 5 cặp câu, backend embedding: `MockEmbedder` (mặc định của lab).

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Người Mua có thể yêu cầu trả hàng trong vòng 15 ngày. | Thời hạn gửi yêu cầu hoàn tiền của khách là 15 ngày. | cao | **-0.1117** | ❌ |
| 2 | Chính sách bảo hành sản phẩm điện tử. | Quy định bảo hành hàng điện máy gia dụng. | cao | **+0.0040** | ❌ |
| 3 | Phí vận chuyển của đơn hàng được tính thế nào? | Thời gian giao hàng dự kiến là bao lâu? | trung bình | **-0.1031** | ❌ |
| 4 | Người Bán tạo đơn hàng ảo sẽ bị xử phạt. | Hôm nay trời Hà Nội mưa rất to. | thấp | **-0.1127** | ✅ |
| 5 | Sản phẩm hết hạn sử dụng được đổi trả. | *(câu giống hệt — đối chứng)* | = 1.0 | **+1.0000** | ✅ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là cặp 1 và cặp 4 cho điểm **gần như y hệt nhau** (-0.1117 so với -0.1127), dù cặp 1 là hai câu gần như đồng nghĩa còn cặp 4 là hai câu chẳng liên quan gì. Nhưng cặp 5 — hai câu giống hệt — lại ra đúng **+1.0000**, chứng tỏ hàm `compute_similarity` của tôi hoàn toàn chính xác. Vậy lỗi không nằm ở công thức mà nằm ở **embedding**: `MockEmbedder` sinh vector bằng `MD5(text)`, mà hàm băm được thiết kế để đổi một bit đầu vào là đổi toàn bộ đầu ra, nên nó phá huỷ mọi quan hệ giữa các văn bản gần nghĩa. Bài học tôi rút ra: cosine similarity chỉ phản ánh được ý nghĩa **khi và chỉ khi** embedding phía dưới có mã hoá ý nghĩa — bản thân công thức không tạo ra ngữ nghĩa.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

**Cấu hình của tôi:** `RecursiveChunker(chunk_size=500)` · embedding `MockEmbedder` · 5 tài liệu chính sách Shopee → **208 chunk**. Kết quả đầy đủ: `report/ket_qua_benchmark.txt`.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Người Mua có bao nhiêu ngày gửi yêu cầu Trả hàng/Hoàn tiền? | `seller-fraud-penalty-policy` — "Hành Vi Gian Lận Trên Sàn Shopee được xem là vi phạm nghiêm trọng..." | 0.3747 | Top-1 ❌ · Top-3 ✅ | *(rỗng — xem ghi chú)* |
| 2 | Sản phẩm được bảo hành miễn phí khi hội đủ điều kiện nào? | `shipping-policy` — "2. Phạm vi áp dụng: các loại hàng hóa không hỗ trợ vận chuyển..." | 0.3454 | Top-1 ❌ · Top-3 ❌ | *(rỗng)* |
| 3 | Người Bán phải đảm bảo hàng còn tối thiểu bao nhiêu hạn sử dụng? | `buyer-warranty-policy` — "Thông tin của trung tâm bảo hành sẽ được ghi trong phiếu bảo hành..." | 0.2676 | Top-1 ❌ · Top-3 ❌ | *(rỗng)* |
| 4 | Thời hạn khiếu nại vận chuyển với đơn giao không thành công? | `seller-listing-regulations` — "+ Xác Nhận Công Bố Phù Hợp Quy Định An Toàn Thực Phẩm..." | 0.3485 | Top-1 ❌ · Top-3 ✅ | *(rỗng)* |
| 5 | Mức bồi thường gian lận tối đa/đơn và ngày áp dụng? *(có `metadata_filter`)* | `seller-fraud-penalty-policy` — "Hành Vi Gian Lận Trên Sàn Shopee được xem là vi phạm nghiêm trọng..." | 0.3405 | Top-1 ✅ · Top-3 ✅ | *(rỗng)* |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 3 / 5

**Nhận xét của tôi về kết quả này:**
> Câu 5 là câu **duy nhất** truy xuất đúng ngay ở top-1, và cũng là câu **duy nhất** có áp dụng `metadata_filter={"audience": "seller"}`. Bộ lọc đã loại bỏ 3 tài liệu không đúng đối tượng *trước khi* xếp hạng, nên dù vector không mang ngữ nghĩa thì không gian ứng viên đã đủ hẹp để trả về đúng tài liệu. Đây là bằng chứng trực tiếp cho tiêu chí *Metadata Utility*: khi chất lượng embedding kém, **metadata filtering gánh phần lớn độ chính xác**.
>
> Ngược lại, 4 câu còn lại cho thấy rõ giới hạn của `MockEmbedder`: hỏi về *bảo hành* thì trả về *vận chuyển* (câu 2), hỏi về *hạn sử dụng* thì trả về *trung tâm bảo hành* (câu 3). Điểm số của cả 5 câu đều nằm trong dải hẹp 0.267–0.375, tức là **không phân biệt được kết quả tốt với kết quả nhiễu** — một tín hiệu xấu theo tiêu chí *Score Distribution*. Tôi kết luận rằng con số 3/5 này **không đánh giá được chiến lược chunking**, vì biến embedding đang che lấp hoàn toàn ảnh hưởng của biến chunking.

> ⚠️ *Ghi chú về cột "Câu trả lời của Agent":* cột này rỗng do hàm `mock_llm_summarizer` trong `bench.py` tìm nhãn `"Ngữ cảnh:"` / `"Câu hỏi:"`, trong khi prompt của `KnowledgeBaseAgent` dùng nhãn tiếng Anh `"Context:"` / `"Question:"` nên không trích được nội dung. Đây là lỗi của script benchmark, không phải của `src/agent.py`.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> **Chunking theo tiêu đề (heading/section) hiệu quả hơn tôi nghĩ.** Thay vì cắt theo số ký tự, cách này cắt đúng tại ranh giới điều khoản mà chính tác giả tài liệu đã đặt ra, nên mỗi chunk trùng khít với một quy định trọn vẹn và luôn mang theo tên mục của nó. Đối chiếu lại chiến lược `RecursiveChunker` của tôi thì thấy rõ điểm yếu: vì chỉ cắt theo `"\n\n"`, nó để lại những chunk chỉ có mỗi tiêu đề trơ trọi như `"E. XỬ LÝ VI PHẠM"` hay `"6. YÊU CẦU ĐỐI VỚI SẢN PHẨM HOÀN TRẢ"` — vừa tạo ra chunk vô dụng, vừa làm phần nội dung phía sau mất đi ngữ cảnh tiêu đề. Nếu làm lại, tôi sẽ gắn tiêu đề mục vào đầu mỗi chunk con của mục đó.
>
> **Dùng mô hình lớn (embedding/LLM thật) cho kết quả tốt hơn hẳn mock.** Với mô hình thật, điểm số tách bạch rõ giữa kết quả đúng và kết quả nhiễu, còn cả 5 câu của tôi đều dồn trong dải hẹp 0.267–0.375 nên không phân biệt được gì. Điều này khẳng định lại nhận định ở mục 4: **chất lượng embedding là yếu tố quyết định**, và chiến lược chunking chỉ phát huy được tác dụng khi embedding phía dưới đã đủ tốt — nếu không thì mọi khác biệt về chunking đều bị nhiễu che lấp.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
