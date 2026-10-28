# Controller Contract v1

- **Task:** Chốt giao diện Controller vào 27/10 (Thứ 3), năm 2026.
- **Người phụ trách:** Đế; Tuấn phụ trách tích hợp S1/harness và shared interface.
- **Trạng thái:** Đặc tả kỹ thuật chính thức cho Phase 2. Các API Controller và scene signals dưới đây là yêu cầu triển khai, không phải bằng chứng code đã hoàn thành hoặc đã được kiểm thử.

## 1. Phạm vi và kiến trúc

Contract xác định Controller C++ API, scene signals S1, feedback S5 và quyết định gửi harness. Mọi module phải tuân theo các quy tắc này. Giá trị classifier, threshold, `probe_scale` và configuration ladder thuộc configuration/policy Controller, không được gán giá trị mặc định tùy ý trong integration.

Kiến trúc v1 là S1 probe → scene-aware initial selection → S1–S5 → S5 feedback → escalation/ACCEPT → S6; REJECT kết thúc run. Mỗi run ứng với một pair trong một repeat, có state và cache riêng.

Các kiểu/API stage hiện tại phải được tái sử dụng, không đổi nghĩa: [common types](../src/common/types.hpp), [S1](../src/s1/s1.hpp), [S2](../src/s2_s3/s2.hpp), [S4](../src/s4/s4.hpp), [S5](../src/s5/s5.hpp), [S6](../src/s6/s6.hpp), [harness config](../harness/harness_types.hpp) và [runner](../harness/pipeline_runner.hpp). Phân công theo [kế hoạch Phase 2](plan.md) và [stages.md](stages.md); stream/geometry theo [quy ước chung](conventions.md), với ngoại lệ synchronization probe tại mục 4.2.

## 2. Bảng giao tiếp

| Bên gửi → bên nhận | Dữ liệu | Readiness và trách nhiệm |
|---|---|---|
| S1 → harness → Controller | `SceneSignals`: `ScaleInfo` và gradient probe của từng ảnh | S1 tạo output cho A/B; harness chỉ gọi `begin_pair` sau khi GPU work, copy và synchronization đã hoàn tất thành công |
| Controller → harness | `Knobs` từ `begin_pair`, sau đó `Decision` từ `observe` | Đế chọn cấu hình/vị trí ladder; Tuấn thực thi, không tự đổi cấu hình |
| Harness → S2/S5/S6 | K → `s2::Config::max_keypoints`; R → `s5::Config::hypotheses`; B → `s6::BlendMode` | S đi vào S1 của cả hai ảnh; harness bảo đảm capacity/workspace đủ cho mọi cấu hình có thể được chọn |
| S5 → harness → Controller | `Feedback`, chứa nguyên `HomographyResult` | Harness kiểm tra return status và đồng bộ trước khi đọc pinned output S5 |
| Controller → harness → S6 | ACCEPT cùng H từ feedback vừa được chấp nhận, B từ cấu hình đầu và giữ nguyên trong `Decision.next` | Harness gọi `prepare_geometry` rồi `render`; lỗi S6 vẫn là failure của pipeline |

Harness sở hữu device buffers, pinned outputs, stream, workspace và cache S5. Controller chỉ xử lý host data, không gọi CUDA, không cấp phát hoặc làm I/O trong các lời gọi per-pair.

## 3. Controller C++ API

Header C++17 dưới đây quy định giao diện phải triển khai tại `src/controller/controller.hpp`. Tái sử dụng `Status`, `ScaleInfo`, `HomographyResult` và `s6::BlendMode`; không thêm mã vào enum `stitch::Status`. Khai báo này không cung cấp implementation.

```cpp
#pragma once

#include <cstddef>
#include "common/types.hpp"
#include "s6/s6.hpp"

namespace stitch::controller {

struct Knobs {
    double S;
    int K;
    int R;
    s6::BlendMode B;
};

struct ImageSceneSignals {
    ScaleInfo scale;
    double mean_abs_gradient;
};

struct SceneSignals {
    ImageSceneSignals a;
    ImageSceneSignals b;
};

struct Feedback {
    Knobs executed;
    Status execution_status;
    HomographyResult homography;
};

enum class Action : int { ACCEPT = 0, RETRY = 1, REJECT = 2 };

enum class Reason : int {
    SELF_CHECK_PASSED = 0,
    TOO_FEW_MATCHES,
    TOO_FEW_INLIERS,
    LOW_INLIER_RATIO,
    LOW_INLIER_COVERAGE,
    HIGH_RESIDUAL,
    DEGENERATE_MODEL,
    EXECUTION_ERROR,
    BUDGET_EXHAUSTED
};

struct Decision {
    Action action;
    Reason reason;
    Knobs next;
};

class Controller {
public:
    Controller() noexcept = default;
    Controller(const Controller&) = delete;
    Controller& operator=(const Controller&) = delete;

    Status initialize(const Knobs* ladder, std::size_t count) noexcept;
    Status begin_pair(const SceneSignals& scene, Knobs* first) noexcept;
    Status observe(const Feedback& feedback, Decision* decision) noexcept;
    void reset_pair() noexcept;

private:
    enum class Phase { Uninitialized, Idle, AwaitingFeedback, Finished };
    const Knobs* ladder_ = nullptr;
    std::size_t count_ = 0;
    std::size_t ladder_position_ = 0;
    Phase phase_ = Phase::Uninitialized;
};

}  // namespace stitch::controller
```

### 3.1. Input/output và ownership

| Hàm | Điều kiện gọi và input | Kết quả khi trả `Status::OK` |
|---|---|---|
| `initialize(ladder, count)` | Chỉ khi không AwaitingFeedback; ladder khác null, count > 0, không trùng cấu hình, mọi `Knobs` hợp lệ và cùng B | Mượn ladder, đặt state Idle. Caller giữ array bất biến tới lần initialize thành công tiếp theo hoặc khi hủy Controller |
| `begin_pair(scene, first)` | Chỉ từ Idle; probe đã hoàn tất, scene hợp lệ, `first != nullptr` | Controller dùng scene để chọn vị trí ban đầu, ghi cấu hình S/K/R/B vào `*first`, chuyển AwaitingFeedback |
| `observe(feedback, decision)` | Chỉ từ AwaitingFeedback; `decision != nullptr`; `executed` khớp cấu hình đang chờ | Ghi một Decision và cập nhật state theo bảng transition |
| `reset_pair()` | Caller đã xử lý xong mọi GPU work/output của run cũ | Xóa vị trí và trạng thái pair, giữ ladder. Về Idle nếu đã initialize, nếu chưa thì vẫn Uninitialized |

`Knobs` hợp lệ khi S hữu hạn trong `(0, 1]`, K và R là số nguyên dương, B là `Feather`, `Multiband3` hoặc `Multiband5`. K là giới hạn trên mỗi ảnh, không phải số keypoint thật. R là tổng budget, không phải số vòng cộng thêm. `ScaleInfo` mới là nguồn kích thước thực và phép đổi tọa độ sau resize.

B thuộc cấu hình ban đầu và phải cố định trong toàn bộ adaptive run v1. `initialize` phải từ chối ladder có các phần tử khác B bằng `INVALID_ARGUMENT`. RETRY chỉ được thay đổi S/K/R. Muốn dùng B khác cho run sau phải initialize với ladder tương ứng; không đổi B trong một run đang hoạt động.

Classifier, ngưỡng self-check và `probe_scale` do configuration/policy Controller cung cấp, bất biến trong một run. Harness phải lấy `probe_scale` từ cùng configuration/policy dùng với Controller trước khi chạy probe; tham số này không thuộc `Knobs`. `probe_scale` phải hữu hạn trong `(0, 1]` và tạo ảnh probe hợp lệ theo mục 4.1. Không cần thêm tham số vào API per-pair để truyền lại cấu hình này.

Harness cung cấp ladder cho `initialize`, không truyền callback hoặc tự tính quyết định. Cách chọn/tune/nạp các quy tắc nội bộ không thuộc contract này. `initialize` phải từ chối ladder mà module không hỗ trợ bằng `INVALID_ARGUMENT`; không tự sửa ladder hoặc thay đổi policy để tiếp tục.

Controller không giữ reference tới scene/feedback/output sau khi hàm trả về. Caller sở hữu các object này; output không được alias input hoặc array ladder. Khởi tạo feedback bằng `{}` rồi điền các trường có hiệu lực. Không dùng chung một Controller cho các run đồng thời; mỗi object có tối đa một pair đang hoạt động.

### 3.2. Per-pair state và feedback sai

Controller là bên duy nhất quản lý `ladder_position_`. Harness không tăng chỉ số, không chọn cấu hình thay thế và không tự chạy fallback. Mỗi run mới phải reset state/cache, chạy probe một lần rồi gọi `begin_pair()` theo mục 4.2; reset cache S5 là việc riêng của harness, không phải tác dụng của `reset_pair()`.

`executed` phải được chép nguyên từ `first` hoặc `Decision.next` của RETRY gần nhất. So sánh S bằng đúng giá trị đã trả, không tự làm tròn/tái dựng từ chuỗi. Harness phải gắn feedback với đúng run hiện tại và không chuyển feedback cũ sang run mới. API không có pair token và không thể phát hiện một feedback cũ có cùng cấu hình chỉ qua giá trị số.

Con trỏ output null, sai lifecycle, enum không nhận diện, cấu hình không khớp, hoặc payload không hợp lệ theo mục 5: trả `INVALID_ARGUMENT`, không sửa state/output. Caller dừng lượt tích hợp lỗi; không coi lỗi gọi API này là chất lượng thấp để tự retry. Chỉ sau khi nguồn dữ liệu được sửa mới được gọi lại, hoặc reset để hủy run. Không đọc Decision nếu return status khác OK.

Ngược lại, feedback báo lỗi execution đúng schema là một input hợp lệ: `observe` trả OK với REJECT/EXECUTION_ERROR. Đây là quyết định kết thúc run, không phải lỗi gọi Controller API.

## 4. S1 → Controller

### 4.1. Scene schema

`SceneSignals.a` và `.b` phải tương ứng đúng thứ tự hai ảnh pipeline. `scale` phải là `ScaleInfo` của ảnh GRAY8 probe tại `probe_scale` trong policy, không bắt buộc bằng scale của attempt S1–S5 sau đó. Metadata phải khớp `make_scale_info` của ảnh gốc và scale probe này.

| Trường | Đơn vị và điều kiện hợp lệ |
|---|---|
| `scale.orig_w`, `orig_h` | Pixel ảnh gốc, số nguyên dương |
| `scale.work_w`, `work_h` | Pixel ảnh GRAY8 nguồn của gradient, ít nhất 2 trên mỗi chiều, không vượt kích thước gốc |
| `scale.sx`, `sy`, `D` | Hữu hạn, đúng kết quả `make_scale_info` theo quy ước hình học hiện có; sx/sy là tỉ lệ kích thước thực |
| `mean_abs_gradient` | Không đơn vị, hữu hạn trong `[0, 1]`, theo công thức bên dưới |

```text
G = (sum_{y=0..h-1, x=0..w-2} |I(x+1,y) - I(x,y)|
   + sum_{y=0..h-2, x=0..w-1} |I(x,y+1) - I(x,y)|)
    / (255 * (h*(w-1) + w*(h-1)))
```

I là GRAY8 sau tiền xử lý S1. Không tính padding của pitch, không dùng pixel ngoài ảnh; phép trừ phải có dấu. Tích lũy và tính số cạnh bằng số nguyên 64-bit, chuyển sang double khi chuẩn hóa. Ảnh hằng cho G = 0. Giữ riêng hai giá trị A/B; không thêm trường texture chưa có định nghĩa. Controller kiểm tra miền giá trị/metadata; Tuấn chịu trách nhiệm công thức trích xuất.

### 4.2. Readiness và synchronization

Harness phải thực hiện đúng thứ tự sau:

1. Chuẩn bị configuration/policy, ladder và buffers/workspace trước run. Sau khi GPU work của run cũ đã hoàn tất, gọi `reset_pair()` và reset cache S5 cho run mới.
2. Chạy S1 probe đúng một lần cho A/B ở `probe_scale` cố định; S1 tạo `SceneSignals`. Caller phải cấp phát output host pinned cho copy async và giữ buffer sống tới khi GPU work/copy hoàn tất.
3. Kiểm tra launch, copy và `cudaStreamSynchronize` thành công trước khi đọc scene signals và gọi `begin_pair()`. Metadata host có sẵn không thay thế gradient đã hoàn tất trên GPU.
4. Chạy attempt S1–S5 theo cấu hình đầu. Được reuse ảnh S1 của probe nếu cùng input, tham số preprocessing, ScaleInfo và dữ liệu output còn nguyên; nếu không tương thích thì chạy lại S1.
5. Sau mỗi attempt, kiểm tra completion S5 và gửi feedback vào `observe()`. RETRY phải chạy phần pipeline bị ảnh hưởng theo cấu hình trả về, không chạy probe lại. ACCEPT mới chạy S6; REJECT kết thúc run.

Synchronization cuối probe là ngoại lệ của Controller v1 đối với quy tắc không đồng bộ sớm trong [conventions.md](conventions.md), mục 4. Probe nằm trước initial selection; trong mỗi attempt S1–S5 vẫn dùng cùng stream và không đồng bộ giữa các stage. Sau S5 phải đồng bộ trước khi đọc feedback; sau S6 phải kiểm tra completion như API hiện tại. Chi phí probe, copy và synchronization thuộc cùng ranh giới end-to-end hiện có.

Controller không tự synchronize và không nhận cờ `ready` thay cho trách nhiệm readiness của caller. Nếu probe lỗi, thiếu scene signals hoặc metadata/gradient không hợp lệ thì phải kết thúc run; không gọi `begin_pair()` bằng dữ liệu giả và không chuyển sang cấu hình đầu cố định.

API S1 hiện tại chưa có output scene signals; phần bổ sung này là integration requirement ở mục 7, không phải tính năng đã được triển khai.

## 5. S5 → Controller

### 5.1. Feedback và validity theo status

`Feedback` giữ nguyên `HomographyResult` theo giá trị, không nhân đôi field S5. `generation`, `previous_hypotheses` và số R đã hoàn tất thuộc harness/S5 workspace; không cần lặp lại trong feedback Controller.

`execution_status` do harness tổng hợp từ các lời gọi stage và điểm đồng bộ: OK chỉ khi S5 output đã sẵn sàng; nếu lỗi API/workspace/launch/completion thì dùng `INVALID_ARGUMENT`, `INSUFFICIENT_WORKSPACE` hoặc `CUDA_ERROR`. Không gán `NOT_ENOUGH_POINTS`/`DEGENERATE` vào trường này; chúng thuộc `homography.status`.

Các quy tắc dưới đây đã được đối chiếu với `prepare`, `finish`, `estimate` trong [s5.cu](../src/s5/s5.cu), [s5.hpp](../src/s5/s5.hpp) và [validation cấu hình](../src/s5/s5_config.cpp). Return status của `estimate` và `HomographyResult.status` không được dùng thay nhau.

| Trạng thái | Payload hợp lệ và cách xử lý |
|---|---|
| `execution_status != OK`, thuộc ba lỗi execution trên | Không đọc bất kỳ trường `homography` nào; REJECT/EXECUTION_ERROR, không retry chất lượng |
| Execution OK, `homography.status == OK` | Kiểm tra toàn bộ model/tín hiệu bên dưới; đạt ngưỡng thì ACCEPT, chưa đạt thì có thể RETRY |
| Execution OK, S5 `NOT_ENOUGH_POINTS` | `num_matches` trong `[0, 3]`, sau khi counts, indices và tọa độ input đã được kiểm tra hợp lệ. Chỉ dùng status/count để quyết định; có thể RETRY với S hoặc K khác |
| Execution OK, S5 `DEGENERATE` | `num_matches >= 4`, nhưng không tìm được model hợp lệ, còn dưới 4 inlier hoặc model sau đổi hệ tọa độ không hợp lệ. Chỉ dùng status/count; có thể RETRY |
| Execution OK, S5 `INVALID_ARGUMENT` | Lỗi dữ liệu device hoặc trạng thái reuse; không dùng count/model làm tín hiệu chất lượng; REJECT/EXECUTION_ERROR |
| Execution OK nhưng S5 status là `INSUFFICIENT_WORKSPACE`/`CUDA_ERROR`; execution status là `NOT_ENOUGH_POINTS`/`DEGENERATE`; enum lạ hoặc payload trái điều kiện status | Feedback không hợp lệ: trả `INVALID_ARGUMENT`, không sinh Decision và không đổi state |

Trong implementation S5 hiện tại, `INSUFFICIENT_WORKSPACE` và `CUDA_ERROR` chỉ đi qua return status của API và phải được harness ghi vào `execution_status`; kernel không sinh hai giá trị này trong `HomographyResult.status`.

Với cả `NOT_ENOUGH_POINTS` và `DEGENERATE`, `finish` hiện ghi `num_inliers = 0`, ratio/coverage bằng 0, H/H_s và residual là NaN, mask bằng 0. Đây không phải model hoặc metric chất lượng để kiểm tra threshold hay render. Controller chỉ cần kiểm tra status/count theo bảng; không buộc kiểm tra giá trị các field không được sử dụng.

Counts phải nằm trong capacity thật mà harness đã cấp; Controller không có device buffer để kiểm chứng allocation hoặc nguồn gốc output. Với S5 `INVALID_ARGUMENT`, `num_matches` còn có thể đã bị đưa về 0 do count ngoài capacity, nên không được diễn giải thành thiếu match. Không đọc pinned output cũ nếu launch/copy/completion thất bại.

### 5.2. Ý nghĩa field khi S5 status OK

| Trường | Kiểu và nghĩa |
|---|---|
| `H`, `H_s` | `double[9]`, row-major, B → A; H trong RGB gốc, H_s trong hệ S1. Phải hữu hạn; không bắt buộc h33 = 1 |
| `num_matches`, `num_inliers` | `int`; `4 <= num_inliers <= num_matches`; số phần tử thật, không phải capacity |
| `inlier_ratio` | `double`, `num_inliers / num_matches`, hữu hạn trong `[0, 1]` |
| `inlier_coverage` | `double`, hữu hạn trong `[0, 1]`; bounding box inlier ở ảnh A trong hệ S1 theo công thức dưới |
| `reproj_residual` | `double`, hữu hạn, không âm; RMS forward reprojection B → A trên inlier cuối, đơn vị pixel S1 |

```text
inlier_coverage =
    min(1, (max_x - min_x) / max(1, work_w_A - 1))
  * min(1, (max_y - min_y) / max(1, work_h_A - 1))
```

Inlier count/ratio/coverage và residual là kết quả sau refinement. Không đồng nhất inlier coverage với overlap coverage của evaluator, không coi residual là pixel gốc khi S thay đổi. Ngưỡng theo từng cấu hình thuộc quy tắc nội bộ Controller, không được khóa giá trị trong contract. ACCEPT từ self-check không chứng nhận QoR offline và không thay thế validation geometry của S6.

## 6. Controller → Harness

### 6.1. Decision và quy tắc chọn lượt tiếp

| Action | `Decision.next` | Trách nhiệm harness |
|---|---|---|
| ACCEPT | Bằng `feedback.executed`, giữ B của cấu hình đầu | Dùng H của feedback hiện tại và `next.B` để chạy S6; kiểm tra geometry rồi render. Không lấy H từ lượt khác |
| RETRY | Cấu hình ở vị trí ladder lớn hơn vị trí hiện tại; B phải bằng B của cấu hình đầu | Chưa chạy S6; chỉ thay S/K/R, áp dụng invalidation, chạy phần bị ảnh hưởng rồi gửi đúng một feedback của lượt mới |
| REJECT | Bằng `feedback.executed` để xác định lượt kết thúc, không phải lệnh thực thi | Dừng run, không render, không tự chuyển thành ACCEPT hoặc chạy fallback ngoài ladder |

Ladder phải hữu hạn, không trùng cấu hình và có cùng B trên mọi phần tử; `initialize` phải kiểm tra các điều kiện này trước run. Chỉ vị trí ladder phải tiến về phía trước; S/K/R có thể tăng hoặc giảm. Mỗi RETRY phải thay ít nhất một trong S/K/R; thay B trong run là lỗi contract, không phải ứng viên để bỏ qua. Không quay về vị trí cũ, số lượt không vượt độ dài ladder. Cấu hình cuối vẫn phải qua self-check.

Nếu S5 báo thiếu 4 match, bỏ qua mọi ứng viên giữ nguyên cả S lẫn K: tăng/giảm R không tạo thêm match. Chọn ứng viên hợp lệ đầu tiên phía sau theo các điều kiện trên; không có thì REJECT/BUDGET_EXHAUSTED. Không yêu cầu S/K tăng chỉ để đáp ứng quy tắc transition; thứ tự ladder do policy riêng cung cấp.

Với S5 OK nhưng fail threshold, reason đầu tiên theo thứ tự: TOO_FEW_INLIERS, LOW_INLIER_RATIO, LOW_INLIER_COVERAGE, HIGH_RESIDUAL. S5 NOT_ENOUGH_POINTS dùng TOO_FEW_MATCHES, DEGENERATE dùng DEGENERATE_MODEL. Khi không còn ứng viên, reason là BUDGET_EXHAUSTED; feedback cuối vẫn giữ nguyên để caller biết lỗi chất lượng. Execution error luôn REJECT/EXECUTION_ERROR trước khi xét threshold.

### 6.2. Bảng state transition

State trong bảng thuộc Controller, không phải trạng thái CUDA hoặc S6.

| State hiện tại | Sự kiện | Output/return | State sau |
|---|---|---|---|
| Uninitialized | `initialize` hợp lệ, ladder cùng B | OK | Idle |
| Idle hoặc Finished | `initialize` hợp lệ, ladder cùng B | OK, thay binding ladder | Idle |
| Idle | Probe hoàn tất, `begin_pair` hợp lệ | OK, first = cấu hình được chọn từ scene signals | AwaitingFeedback |
| Idle | Probe lỗi/không có scene signals hợp lệ | Không gọi Controller, kết thúc run | Idle |
| AwaitingFeedback | Feedback OK và self-check pass | OK, ACCEPT/SELF_CHECK_PASSED | Finished |
| AwaitingFeedback | Quality fail, còn ứng viên | OK, RETRY; Controller tăng vị trí, giữ B trước khi trả | AwaitingFeedback, chờ cấu hình mới |
| AwaitingFeedback | Quality fail, không còn ứng viên | OK, REJECT/BUDGET_EXHAUSTED | Finished |
| AwaitingFeedback | Execution error hợp lệ | OK, REJECT/EXECUTION_ERROR | Finished |
| Bất kỳ | Lỗi input/lifecycle của hàm trả Status | INVALID_ARGUMENT, output không đổi | Không đổi |
| Đã initialize | `reset_pair` | Không output; caller đã kết thúc GPU work cũ | Idle |
| Uninitialized | `reset_pair` | Không output | Uninitialized |

Finished chỉ có nghĩa Controller không nhận thêm feedback cho run này. Sau ACCEPT, run chỉ thành công khi S6 và điểm completion cuối thành công. Gọi `observe` sau ACCEPT/REJECT hoặc gửi lặp feedback lượt trước sau RETRY là lỗi lifecycle/cấu hình, không phải yêu cầu chạy thêm.

### 6.3. Cache reuse/invalidation

S5 đã có API nối tiếp raw RANSAC state. Harness phải quản lý `generation`, `previous_hypotheses`, địa chỉ/nội dung buffer và điều kiện reuse; Controller không quản lý cache này.

| Thay đổi so với input/cấu hình có trong cache | Có thể giữ | Phải thực hiện |
|---|---|---|
| S/K giữ nguyên, R tăng | S1–S4 và workspace raw S5 nếu toàn bộ điều kiện extension thỏa | S5: `hypotheses = R_new`, `previous_hypotheses = R_old`; chấm thêm hypothesis, refinement/self-check lại |
| S/K giữ nguyên, R giảm | S1–S4 và match | Restart S5 với `previous_hypotheses = 0`; raw winner của R lớn có thể đến từ hypothesis vượt R mới |
| K thay đổi theo bất kỳ chiều nào, S giữ nguyên | S1 | Chạy lại S2–S5, đổi generation; không giả định output K mới là prefix của lần cũ |
| S thay đổi theo bất kỳ chiều nào | RGB gốc | Chạy lại S1–S5, tính ScaleInfo mới, đổi generation |
| B thay đổi trong run | Không áp dụng | Lỗi contract; harness không được thực thi cấu hình này |
| Đổi pair/repeat, ghi đè input hoặc đổi tham số ngoài S/K/R/B | Chỉ vùng cấp phát còn đủ dung lượng | Reset các kết quả bị phụ thuộc; S5 restart, đổi generation dù địa chỉ pointer không đổi |

Để extension đúng [S5 API](../src/s5/s5.hpp), phải giữ nguyên keypoint/match arrays, counts/capacities, ScaleInfo, seed, threshold, batch size, generation, stream và workspace. Contract giữ cả refinement steps cố định trong một run. `previous_hypotheses` phải bằng R của lần S5 đã hoàn tất trong workspace. Nếu không chứng minh được điều kiện reuse thì restart S5 với `previous_hypotheses = 0`, không giả định cache hợp lệ.

Với execution OK và S5 DEGENERATE trên input đủ match, code hiện tại vẫn lưu trạng thái raw search; có thể extension khi các điều kiện trên còn đúng. Không dùng H/metric của kết quả DEGENERATE để render. Execution/CUDA error kết thúc run; không nối tiếp workspace lỗi. Seed và tham số stage khác không được tự đổi như một phần RETRY.

## 7. Integration Requirements

| Module | Trách nhiệm kỹ thuật |
|---|---|
| S1 — Tuấn | Bổ sung output `SceneSignals` A/B của probe theo công thức/ScaleInfo mục 4; không tự đồng bộ trong stage hoặc đổi nghĩa API cũ |
| S5 — Đế | Giữ đúng status, count, model và đơn vị metric hiện tại; xuất pinned output sau async copy; bảo toàn raw state cho continuation đúng điều kiện mục 6.3 |
| Controller — Đế | Triển khai header/lifecycle mục 3; kiểm tra ladder cùng B và input; chọn cấu hình từ scene signals, quản lý ladder position và trả Decision đúng state machine |
| Harness — Tuấn | Dùng cùng policy/probe scale với Controller; cấp phát và quản lý lifetime, chạy probe một lần, kiểm tra completion rồi gọi begin/observe; thực thi S/K/R, giữ B, quản lý invalidation; chỉ ACCEPT mới gửi H và B tới S6 |

S2–S3 phải nhận K của mỗi attempt; S6 phải nhận H gốc và B cố định khi ACCEPT. Thay đổi này chỉ xác định contract; không cung cấp implementation Controller, output scene signals S1 hoặc kết quả kiểm thử integration/T4.
