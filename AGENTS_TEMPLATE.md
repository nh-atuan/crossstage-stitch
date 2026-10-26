# Khuôn mẫu quy tắc cho AI agent của thành viên

File này là bộ quy tắc **tối thiểu** mà agent của mỗi thành viên phải tuân thủ khi làm workshop paper PARMA-DITAM 2027 trong repo `crossstage-stitch`. File được cập nhật thường xuyên trên nhánh `main`.

## Cách sử dụng

1. Sao chép toàn bộ nội dung file này thành file hướng dẫn agent của riêng bạn ở thư mục gốc repo. Tên file theo công cụ đang dùng, ví dụ `AGENTS.md` (Codex và các agent đọc `AGENTS.md`) hoặc `CLAUDE.md` (Claude Code).
2. Bạn được thêm, sửa hoặc sắp xếp lại nội dung trong file của mình cho phù hợp với cách làm việc riêng, nhưng **phải giữ ít nhất toàn bộ nội dung của khuôn mẫu này**. Nếu quy tắc riêng mâu thuẫn với khuôn mẫu, khuôn mẫu được ưu tiên.
3. **Không commit** file hướng dẫn agent của riêng bạn. Các tên file này đã có trong `.gitignore`.
4. Mỗi khi kéo code mới về (`git pull`), kiểm tra xem `AGENTS_TEMPLATE.md` có thay đổi không (ví dụ bằng `git log -p -- AGENTS_TEMPLATE.md`). Nếu có, cập nhật file của bạn cho đầy đủ trước khi giao việc tiếp cho agent.
5. Gợi ý, không bắt buộc: kết nối cho agent một công cụ chạy được Colab, ví dụ [Colab MCP server](https://github.com/googlecolab/colab-mcp) của Google hoặc một MCP server, plugin tương đương. Khi đó agent tự build và chạy test trên Colab T4.

Phần dưới đây là nội dung dành cho agent.

## 1. Phạm vi và nguồn yêu cầu

1. Repo này chỉ phục vụ workshop paper. Đồ án môn học CSC14120 là một phạm vi riêng: chỉ dùng làm tham khảo về domain (panorama stitching trên GPU), không áp dụng các yêu cầu của đồ án (các version V0-V4, dataset tự chụp, notebook...) cho repo này, trừ khi người dùng yêu cầu.
2. Agent đọc và tuân thủ các nguồn sau, theo thứ tự ưu tiên giảm dần:
    1. Yêu cầu hiện tại của người dùng.
    2. Yêu cầu chính thức của workshop (tóm tắt ở mục 1 của `docs/plan.md`, trang gốc là chuẩn).
    3. Kế hoạch và quy định của Nhóm trong `docs/plan.md`.
    4. Các tài liệu khác trong `docs/`, ví dụ `docs/stages.md`, quy ước chung và các ghi chép quyết định kỹ thuật khi đã có.
    5. Khuôn mẫu này và các quy tắc riêng của thành viên.
3. Nếu các nguồn mâu thuẫn nhau, agent phải dừng phần việc bị ảnh hưởng, chỉ ra chỗ mâu thuẫn kèm trích dẫn cụ thể và hỏi người dùng.

## 2. Không tự quyết

1. **Không tự ý thực hiện khi chưa được người dùng cho phép.** Mọi ý tưởng và đề xuất, kể cả việc tạo, sửa hay xóa file, phải được trình bày và thảo luận với người dùng trước (kèm ưu, nhược điểm và khuyến nghị); chỉ khi người dùng đồng ý mới được thực hiện.
2. Khi người dùng chỉ hỏi ý kiến hoặc nhờ đề xuất, chỉ trả lời và đề xuất; chờ người dùng yêu cầu rõ rồi mới thực hiện.
3. Những việc luôn phải hỏi trước khi làm:
    - chọn hoặc đổi phương pháp, tham số, cấu hình của một stage hay module;
    - thay đổi quy ước chung của Nhóm;
    - sửa file thuộc phần phụ trách của thành viên khác hoặc file dùng chung;
    - thêm thư viện, công cụ hoặc phụ thuộc mới;
    - tạo, đổi tên, di chuyển hoặc xóa file và thư mục ngoài phạm vi task;
    - push, tạo pull request, merge, hoặc bất kỳ thao tác nào ảnh hưởng ra bên ngoài máy của người dùng;
    - mọi thao tác khó hoàn tác.
4. Nếu yêu cầu còn thiếu, mơ hồ hoặc có nhiều cách hiểu, agent phải **hỏi lại người dùng** trước khi làm phần bị ảnh hưởng. Câu hỏi ngắn gọn, cụ thể và nói rõ mỗi lựa chọn dẫn tới kết quả khác nhau thế nào. Agent được tiếp tục các bước khảo sát chỉ đọc không phụ thuộc vào câu trả lời.
5. Nếu người dùng cũng chưa chắc, agent đọc lại tài liệu của Nhóm và trích dẫn những đoạn liên quan; nếu vẫn không có câu trả lời, đề nghị người dùng báo lại cho cả Nhóm để cùng quyết định và tạm dừng phần việc đó.

## 3. Trước khi thực hiện

1. Xác định task đang làm, phase chứa task và deadline của phase trong `docs/plan.md`.
2. Khảo sát trạng thái hiện tại bằng thao tác chỉ đọc: cấu trúc repo, `git status`, nhánh hiện tại và mức đồng bộ với `main`, code, test và tài liệu liên quan. Nếu nhánh đang thiếu thay đổi mới của `main`, hỏi người dùng có muốn kéo `main` về trước khi làm không.
3. Tóm tắt cho người dùng: mục tiêu, phạm vi, tiêu chí nghiệm thu, các giả định và các câu hỏi còn mở. Chỉ bắt đầu sửa đổi khi người dùng đã xác nhận.
4. **Kiểm tra công cụ để tự kiểm chứng.** Liệt kê các tool, MCP server, skill hoặc plugin cần dùng để build, chạy test và rà soát (ví dụ shell, `git`, `nvcc`, Python, LaTeX). Nếu thiếu công cụ hoặc thiếu quyền, báo rõ cho người dùng là cần cấp gì, và không bắt đầu phần việc cần kiểm chứng khi chưa có đủ.
5. Làm việc trên đúng nhánh theo quy tắc Git trong quy ước chung của Nhóm. Nếu đang ở nhánh không đúng, hỏi người dùng trước khi tạo hoặc chuyển nhánh.

## 4. Trong khi thực hiện

1. Chỉ làm đúng phạm vi task. Thay đổi nhỏ, tập trung, nhất quán với code hiện có; không tự sửa định dạng, đổi tên hoặc tái cấu trúc phần không liên quan.
2. Giữ nguyên các thay đổi sẵn có của người dùng. Nếu phát hiện xung đột hoặc thay đổi ngoài dự kiến, dừng phần bị ảnh hưởng, đưa bằng chứng và hỏi cách xử lý.
3. Mỗi quyết định kỹ thuật đã được người dùng chốt phải được ghi lại trong `docs/` theo mẫu của Nhóm: bối cảnh, phương án chọn, trade-off hoặc phương án phụ, fallback, người quyết định, ngày. Mọi quyết định hay phương án kỹ thuật đều phải cân nhắc các phương án tương tự, phân tích kỹ trade-off và nhận định rõ lựa chọn có thực sự phù hợp, đảm bảo tính đúng đắn, chính xác trong ngữ cảnh hiện tại hay không.
4. Build CUDA bằng `nvcc -arch=sm_75` (GPU Tesla T4). Mọi yếu tố ngẫu nhiên (ví dụ RANSAC) dùng seed cố định và được in ra cùng kết quả.
5. Code hoặc ý tưởng lấy từ bên ngoài phải ghi nguồn (URL, ngày truy cập, dùng cho phần nào).
6. **Chia nhỏ phần việc thành nhiều file, hàm và module**, mỗi đơn vị làm một việc rõ ràng, để dễ đọc, dễ viết test, dễ kiểm chứng riêng và dễ khoanh vùng khi debug.
7. Xử lý lỗi ở nguyên nhân gốc. Không vô hiệu hóa test, không bỏ qua cảnh báo quan trọng, không nới lỏng điều kiện kiểm tra để cho qua.
8. Báo tiến độ ngắn gọn khi việc kéo dài, khi cần quyết định hoặc khi gặp trở ngại.
9. Không bịa đặt kết quả, số đo, trạng thái test hoặc mức độ hoàn thành. Luôn phân biệt rõ điều đã kiểm chứng, điều suy luận và điều chưa kiểm tra được.
10. Không tiết lộ bí mật như token hay mật khẩu, không ghi chúng vào code, output hoặc commit.

## 5. Kiểm chứng và nghiệm thu

1. **Mọi test liên quan phải pass trước khi coi là xong.** Nếu có test fail, task chưa xong: sửa và chạy lại cho tới khi pass, hoặc báo rõ lý do bị chặn.
2. Môi trường kiểm chứng và đo hiệu năng chính thức là **Google Colab với GPU Tesla T4**. Nếu agent không chạy được trên Colab T4, phải nói rõ kết quả chưa được kiểm chứng trên môi trường này và đưa cho người dùng lệnh cụ thể cùng output mong đợi để tự chạy.
3. Mọi con số trong paper phải được sinh từ code và file kết quả của Nhóm, không chép tay.
4. Rà soát lại **toàn bộ** thay đổi trước khi bàn giao:
    - diff chỉ chứa thay đổi thuộc task; không có file tạm, output sinh tự động ngoài ý muốn hay bí mật;
    - tài liệu liên quan đã được cập nhật nếu hành vi hoặc quy ước thay đổi;
    - `git diff --check` không báo lỗi.
5. Không sửa lỗi không liên quan nếu việc sửa có thể làm tăng phạm vi hoặc rủi ro. Ghi nhận ngắn gọn để người dùng quyết định.

## 6. Sau khi thực hiện

1. Báo cáo ngắn gọn cho người dùng: đã thay đổi gì, ở file nào; đã chạy kiểm tra nào và kết quả; kiểm tra nào chưa chạy được và lý do; tiêu chí nào đã đạt, tiêu chí nào chưa; trạng thái working tree.
2. **Sau mỗi thay đổi đều commit** để Git theo dõi, theo các quy tắc:
    - commit message bằng tiếng Anh theo [Conventional Commits](https://www.conventionalcommits.org/);
    - không thêm dòng `Co-Authored-By` của AI vào commit message, không thêm dòng ghi công AI (như "Generated with Claude Code") vào mô tả pull request;
    - chỉ stage các file thuộc task và xem lại nội dung đã stage trước khi commit;
    - sau khi commit, kiểm tra bằng `git show --check --oneline HEAD`;
    - không force-push lên `main`, không sửa lịch sử hay amend commit của người khác;
    - không commit file hướng dẫn agent cá nhân, dataset, output, token hoặc file build;
    - không commit output của notebook: vẫn chạy notebook trước để kiểm tra, chỉ xóa toàn bộ output khi commit.
3. Push và tạo pull request chỉ khi người dùng yêu cầu. Pull request ghi rõ task đã làm, mô tả thay đổi và output build/test. Agent không tự merge pull request.
4. Nếu bị chặn, nêu nguyên nhân cụ thể, những gì đã thử, và thông tin hoặc quyền cần người dùng bổ sung.

## 7. Quy ước tài liệu, ngôn ngữ và paper

1. Tài liệu `.md`, phần giải thích lựa chọn và phân tích kết quả viết bằng **tiếng Việt**. Giữ nguyên thuật ngữ tiếng Anh khi không nên dịch hoặc bản dịch không phổ biến (keypoint, descriptor, homography, kernel, thread, warp, latency...).
2. Tên file, tên thư mục, tên nhánh, code, tên biến, tên hàm, comment trong code và commit message viết bằng **tiếng Anh**. Ngoại lệ là tên riêng như tên thành viên; khi dùng trong tên file hoặc thư mục thì viết không dấu.
3. Không dùng blockquote Markdown (dòng bắt đầu bằng `>`). Mọi sơ đồ hoặc flowchart trong Markdown viết bằng Mermaid trong code fence `mermaid`.
4. Đường dẫn trong tài liệu dùng đường dẫn tương đối, không dùng đường dẫn tuyệt đối.
5. Gọi thành viên bằng tên, không dùng ký hiệu chữ cái thay cho tên.
6. Mốc ngày ghi dạng `dd/mm (Thứ n)` (ví dụ `15/10 (Thứ 5)`), viết đầy đủ, không viết tắt như T5; không ghi theo khoảng ngày. Deadline chỉ ghi cho cả phase, không ghi cho từng task.
7. Kế hoạch được lập cuốn chiếu: chỉ ghi task chi tiết cho phase hiện tại; các phase sau chỉ ghi ý chính và được lập chi tiết khi phase trước hoàn thành.
8. Paper viết bằng **tiếng Anh** bằng LaTeX trong `paper/`, theo template OASIcs. Paper được review double-blind: không đưa tên thành viên, tên trường, link GitHub cá nhân hay thông tin nhận diện nhóm vào paper và metadata của PDF.
9. Khi viết văn bản (tài liệu, paper), agent phải đọc trước trang [Wikipedia:Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) và tránh các dấu hiệu được liệt kê ở đó. Sau khi viết xong, rà lại văn bản theo trang này và sửa các chỗ còn dấu hiệu.
