# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
>  Khi gọi `call_openai` temperature thấp (0.0 – 0.5), phản hồi mang tính xác định cao với câu trả lời cô đọng nội dung, được thể hiện dưới các gạch đầu dòng, logic lập luận chặt chẽ. Khi temperature tăng lên (1.0 – 1.5), câu trả lời trở nên đa dạng, phong phú và sáng tạo hơn về góc nhìn cũng như cách trình bay, tuy nhiên ở mức 1.5 cách hành văn bắt đầu kém ổn định và có xu hướng lan man hơn - xuất hiện hiện tượng hallucination và phóng đại số liệu sai thực tế. 

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Đối với chatbot hỗ trợ khách hàng, em sẽ đặt temperature ở ngưỡng từ 0.0 đến 0.3 (tối đa 0.5), bởi vì nhằm mục đích đảm bảo tính chính xác và nhất quán tuyệt đối về các thông tin quan trọng như giá cả, chính sách và quy trình dịch vụ, tránh việc đưa ra câu trả lời mâu thuẫn cho các khách hàng khác nhau. Đồng thời, giúp chatbot phản hồi luôn ngắn gọn, đi thẳng vào trọng tâm và hạn chế tối đa hiện tượng hallucination, ngăn ngừa rủi ro model tự bịa đặt thông tin sai lệch.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Đối với các tác vụ yêu cầu suy luận logic sâu, phân tích hợp đồng/tài chính phức tạp, sinh mã nguồn kiến trúc nhiều tầng hoặc giải quyết các bài toán có tính rủi ro cao cần độ chính xác tối đa, ưu tiên việc sử dụng GPT-4o. Đối với Các tác vụ quy mô lớn, tần suất cao nhưng logic đơn giản như phân loại email/ý định khách hàng (intent routing), chatbot giải đáp FAQ cơ bản, tóm tắt văn bản ngắn hoặc kiểm tra chính tả/ngữ pháp, ưu tiên sử dụng GPT-4o mini để giúp tiết kiệm được chi phí. 

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Khi quan sát hai phản hồi đến từ hai system prompt, em nhận thấy có sự khác biệt rõ về phong cách, từ vựng và ví dụ minh hoạ. Nếu phản hồi từ giáo viên tiểu học bao gồm các câu trả lời ngắn gọn, giọng văn thân thiện cùng với việc sử dụng các từ ngữ mộc mạc và gần gũi với mọi người; ngược lại, chuyên gia tài chính lại sử dụng các cấu trúc câu mang tính học thuật, chuyên môn, phân tích sâu hơn và sử dụng các thuật ngữ chuyên ngành nhiều hơn. Từ đây, em nhận thấy được việc sử dụng system prompt khác nhau nó sẽ đóng vai trò như là một bộ lọc ngữ cảnh, từ đó góp phần định hình phong cách trả lời, độ am hiểu về kiến thức chuyên môn và tập từ vựng mà model sẽ ưu tiên kích hoạt. Nhờ vào đó, model có thể tuỳ biến linh hoạt câu trả lời phù hợp với từng đối tượng khách hàng/người dùng mà không cần thay đổi câu hỏi gốc của người dùng*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn Tiếng Việt 113, ước lượng thôn (113/0.75) cho ~150 token, trong khi count_tokens (tiktoken) đo được 123 token - hai con số có tỷ lệ chênh lệnh khoảng 18.4%. Công thức ước lượng thô bị sai lệch đáng kể vì tỷ lệ 0.75 từ/token được xây dựng dựa trên đặc điểm của tiếng Anh, không phản ánh đúng cách tokenizer xử lý tiếng Việt. Tiếng Việt thường tốn nhiều token hơn tiếng Anh cùng độ dài vì tokenizer (BPE) được huấn luyện chủ yếu trên dữ liệu tiếng Anh, nơi các từ phổ biến thường được mã hóa thành 1 token duy nhất; còn tiếng Việt chứa nhiều ký tự có dấu thanh và ký tự Unicode đa byte, khiến nhiều âm tiết bị tách thành 2–3 sub-tokens thay vì 1.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các giao diện tương tác trực tiếp với người dùng (như chatbot, trợ lý AI, công cụ viết văn bản) hoặc khi câu trả lời dài, giúp giảm triệt để thời gian chờ ký tự đầu tiên, mang lại cảm giác hệ thống phản hồi tức thì và người dùng có thể đọc dần nội dung thay vì phải chờ đợi toàn bộ phản hồi hoàn tất. Ngược lại, non-streaming phù hợp hơn trong các tác vụ xử lý ngầm (background jobs, batch processing), giao tiếp giữa các dịch vụ nội bộ, hoặc khi ứng dụng cần nhận về toàn bộ dữ liệu có cấu trúc để kiểm tra schema, chấm điểm hoặc kiểm duyệt an toàn trước khi xử lý tiếp.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> So với delay cố định, exponential backoff có lợi thế là giãn cách thời gian chờ tăng theo cấp số nhân sau mỗi lần thử lại (0.1s → 0.2s → 0.4s...), giúp giảm dần áp lực lên máy chủ theo thời gian và tạo khoảng trống đủ lớn để hệ thống giải phóng tài nguyên, xử lý xong hàng đợi và phục hồi sau sự cố quá tải. Nếu hàng nghìn client cùng retry với một khoảng thời gian cố định giống nhau (ví dụ đều chờ đúng 1 giây), toàn bộ các client sẽ đồng loạt gửi request ập vào máy chủ tại cùng một thời điểm sau 1 giây, tạo ra hiện tượng "bão retry" khiến máy chủ vừa hồi phục đã lập tức bị đánh sập trở lại.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt đã chọn: "Bạn là trợ giảng thân thiện của khóa AI Practical Competency, luôn giải thích các khái niệm kỹ thuật một cách trực quan, dễ hiểu và trả lời ngắn gọn bằng tiếng Việt." Giải thích lựa chọn từ ngữ:
>   - "thân thiện, trực quan": Định hình phong cách sư phạm gần gũi, giúp model ưu tiên sử dụng các ví dụ thực tế và phép ẩn dụ thay vì đưa ra các định nghĩa hàn lâm, khô khan.
>   - "trả lời ngắn gọn": Kiểm soát số lượng token đầu ra, giúp tăng tốc độ phản hồi khi streaming và tiết kiệm tối đa chi phí API qua các lượt hội thoại.
>   - "bằng tiếng Việt": Cố định ngôn ngữ phản hồi nhất quán, ngăn tình trạng model tự động chuyển sang tiếng Anh khi người dùng hỏi các thuật ngữ kỹ thuật mượn tiếng Anh.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: Trợ lý chỉ duy trì bộ nhớ ngắn hạn tạm thời trong RAM và bị cắt cứng ở 3 lượt hội thoại gần nhất (history = history[-6:]), dẫn đến việc mất toàn bộ ngữ cảnh quan trọng ở đầu phiên nếu cuộc trò chuyện kéo dài; đồng thời toàn bộ dữ liệu phiên chat sẽ bị xóa sạch khi người dùng gõ quit hoặc tắt ứng dụng. 

> Đề xuất cải thiện & Cách triển khai: Bổ sung cơ chế tóm tắt ngữ cảnh và lưu trữ lâu dài:
>    - Tóm tắt ngữ cảnh: Khi lịch sử chat vượt quá 3 lượt, thay vì xóa bỏ trực tiếp các tin nhắn cũ, hệ thống sẽ dùng model nhỏ để tóm tắt các ý chính và thông tin quan trọng của các lượt cũ thành một đoạn văn ngắn gọn.
>    - Lưu trữ phiên: Lưu đoạn tóm tắt và lịch sử hội thoại vào cơ sở dữ liệu cục bộ (SQLite hoặc file JSON) gắn theo từng phiên làm việc.
>    - Tái sử dụng: Ở các lượt chat tiếp theo, đoạn tóm tắt này sẽ được tự động chèn vào sau persona trong system prompt (dưới dạng <memory>...</memory>), giúp trợ lý duy trì trí nhớ xuyên suốt cuộc trò chuyện mà vẫn tối ưu chi phí token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
