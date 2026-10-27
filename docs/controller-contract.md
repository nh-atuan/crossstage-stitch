# Controller Contract v1 — Phase 2

- **Task:** Chốt giao diện Controller vào 27/10 (Thứ 3), năm 2026.
- **Người phụ trách:** Đế; Tuấn phụ trách tích hợp S1/harness và shared interface.
- **Trạng thái:** Proposed; chưa được phê duyệt, chưa triển khai Controller.

## 1. Phạm vi và mức xác nhận

Contract chỉ xác định Controller C++ API, scene signals S1, feedback S5 và quyết định gửi harness. Không chọn classifier, threshold hoặc giá trị configuration ladder; không đổi benchmark, QoR evaluator hay telemetry.

| Nhãn | Ý nghĩa trong tài liệu |
|---|---|
| Confirmed | Đã đối chiếu với tài liệu hoặc code hiện tại; không có nghĩa toàn contract đã được duyệt |
| Proposed | Giao diện hoặc quy tắc mới đề nghị khóa cho Phase 2 |
| Pending approval | Cần người phụ trách xác nhận trước khi coi là cam kết integration |

Confirmed: [kế hoạch Phase 2](plan.md) giao Đế chốt contract; [quy ước chung](conventions.md) giao Tuấn sửa shared interface. Các kiểu/API hiện có được giữ nguyên: [common types](../src/common/types.hpp), [S1](../src/s1/s1.hpp), [S2](../src/s2_s3/s2.hpp), [S4](../src/s4/s4.hpp), [S5](../src/s5/s5.hpp), [S6](../src/s6/s6.hpp), [harness config](../harness/harness_types.hpp) và [runner](../harness/pipeline_runner.hpp). Vai trò Controller theo [stages.md](stages.md).

## 2. Bảng giao tiếp

Các schema mới trong bảng là Proposed; các type stage được dẫn chiếu là Confirmed.

| Bên gửi → bên nhận | Dữ liệu | Readiness và trách nhiệm |
|---|---|---|
| S1 → harness → Controller | `SceneSignals`: `ScaleInfo` và gradient của từng ảnh | Tuấn tạo output; harness chỉ gọi `begin_pair` khi dữ liệu host đã sẵn sàng, đúng pair |
| Controller → harness | `Knobs` từ `begin_pair`, sau đó `Decision` từ `observe` | Đế chọn cấu hình/vị trí ladder; Tuấn thực thi, không tự đổi cấu hình |
| Harness → S2/S5/S6 | K → `s2::Config::max_keypoints`; R → `s5::Config::hypotheses`; B → `s6::BlendMode` | S đi vào S1 của cả hai ảnh; harness bảo đảm capacity/workspace đủ cho mọi cấu hình có thể được chọn |
| S5 → harness → Controller | `Feedback`, chứa nguyên `HomographyResult` | Harness kiểm tra return status và đồng bộ trước khi đọc pinned output S5 |
| Controller → harness → S6 | ACCEPT cùng H từ feedback vừa được chấp nhận, B từ `Decision.next` | Harness gọi `prepare_geometry` rồi `render`; lỗi S6 vẫn là failure của pipeline |

Harness sở hữu device buffers, pinned outputs, stream, workspace và cache S5. Controller chỉ xử lý host data, không gọi CUDA, không cấp phát hoặc làm I/O trong các lời gọi per-pair.

## 3. Controller C++ API — Proposed

Ví dụ header C++17 dưới đây là khai báo để triển khai tại `src/controller/controller.hpp`, không phải Controller đã được cài đặt. Tái sử dụng `Status`, `ScaleInfo`, `HomographyResult` và `s6::BlendMode`; không thêm mã vào enum `stitch::Status`.

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
| `initialize(ladder, count)` | Chỉ khi không AwaitingFeedback; ladder khác null, count > 0, không trùng cấu hình, mọi `Knobs` hợp lệ | Mượn ladder, đặt state Idle. Caller giữ array bất biến tới lần initialize thành công tiếp theo hoặc khi hủy Controller |
| `begin_pair(scene, first)` | Chỉ từ Idle; scene hợp lệ, `first != nullptr` | Controller chọn vị trí ban đầu, ghi cấu hình vào `*first`, chuyển AwaitingFeedback |
| `observe(feedback, decision)` | Chỉ từ AwaitingFeedback; `decision != nullptr`; `executed` khớp cấu hình đang chờ | Ghi một Decision và cập nhật state theo bảng transition |
| `reset_pair()` | Caller đã xử lý xong mọi GPU work/output của run cũ | Xóa vị trí và trạng thái pair, giữ ladder. Về Idle nếu đã initialize, nếu chưa thì vẫn Uninitialized |

`Knobs` hợp lệ khi S hữu hạn trong `(0, 1]`, K và R là số nguyên dương, B là `Feather`, `Multiband3` hoặc `Multiband5`. K là giới hạn trên mỗi ảnh, không phải số keypoint thật. R là tổng budget, không phải số vòng cộng thêm. `ScaleInfo` mới là nguồn kích thước thực và phép đổi tọa độ sau resize.

Classifier và ngưỡng self-check do Đế cung cấp bên trong module Controller, bất biến trong một run; harness chỉ cung cấp ladder cho `initialize`, không truyền callback hoặc tự tính quyết định. Cách chọn/tune/nạp các quy tắc nội bộ không thuộc contract này. Đế phải cung cấp quy tắc phù hợp với ladder được tích hợp; `initialize` từ chối ladder mà module không hỗ trợ bằng `INVALID_ARGUMENT`.

Controller không giữ reference tới scene/feedback/output sau khi hàm trả về. Caller sở hữu các object này; output không được alias input hoặc array ladder. Khởi tạo feedback bằng `{}` rồi điền các trường có hiệu lực. Không dùng chung một Controller cho các run đồng thời; mỗi object có tối đa một pair đang hoạt động.

### 3.2. Per-pair state và feedback sai

Controller là bên duy nhất quản lý `ladder_position_`. Harness không tăng chỉ số, không chọn cấu hình thay thế và không tự chạy fallback. Mỗi pair/repeat mới phải gọi `reset_pair()` rồi `begin_pair()`; reset cả cache S5 là việc riêng của harness, không phải tác dụng của `reset_pair()`.

`executed` phải được chép nguyên từ `first` hoặc `Decision.next` của RETRY gần nhất. So sánh S bằng đúng giá trị đã trả, không tự làm tròn/tái dựng từ chuỗi. Harness phải gắn feedback với đúng run hiện tại và không chuyển feedback cũ sang run mới. API không có pair token và không thể phát hiện một feedback cũ có cùng cấu hình chỉ qua giá trị số.

Con trỏ output null, sai lifecycle, enum không nhận diện, cấu hình không khớp, hoặc payload không hợp lệ theo mục 5: trả `INVALID_ARGUMENT`, không sửa state/output. Caller dừng lượt tích hợp lỗi; không coi lỗi gọi API này là chất lượng thấp để tự retry. Chỉ sau khi nguồn dữ liệu được sửa mới được gọi lại, hoặc reset để hủy run. Không đọc Decision nếu return status khác OK.

Ngược lại, feedback báo lỗi execution đúng schema là một input hợp lệ: `observe` trả OK với REJECT/EXECUTION_ERROR. Đây là quyết định kết thúc run, không phải lỗi gọi Controller API.

## 4. S1 → Controller

### 4.1. Scene schema — Proposed

`SceneSignals.a` và `.b` tương ứng đúng thứ tự hai ảnh pipeline. `scale` tái sử dụng `ScaleInfo` của ảnh GRAY8 dùng để trích scene feature, không bắt buộc bằng scale của lần S1–S5 sau đó.

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

Confirmed: API S1 hiện chỉ xuất ảnh và `ScaleInfo`, không xuất scene features; kernel không đồng bộ nội bộ. Metadata host có sẵn không có nghĩa gradient GPU đã sẵn sàng.

Proposed: Tuấn cung cấp output scene signals bổ sung mà không đổi nghĩa API cũ. Khi copy async từ GPU, caller cấp phát output host pinned và giữ nó sống tới khi copy xong; harness kiểm tra stream completion trước `begin_pair`. Controller không tự synchronize và không nhận cờ `ready` thay cho việc caller bảo đảm readiness.

**Pending decision — Pending approval:** dùng S1 probe ở một scale cố định trước lượt chạy, copy gradient về host rồi đồng bộ để chọn S/K ban đầu. Scale probe chưa chốt; không mặc định chạy probe trong kiến trúc hiện tại.

[Quy ước hiện tại](conventions.md), mục 4 ghi “Không đồng bộ giữa S1 và S5”. Đọc gradient trên CPU trước khi chọn cấu hình đầu cần một điểm completion sớm; không được tự thêm điểm này vào luồng đã khóa. Đế và Tuấn phải thống nhất probe/điểm đọc trước khi nối đường scene-aware initial selection. Nếu probe được duyệt, nó nằm trước lượt S1–S5 và mọi chi phí thuộc cùng ranh giới đo hiện có. Chưa có scene signals hợp lệ thì không gọi `begin_pair` với dữ liệu giả hoặc ngầm dùng cấu hình cố định.

## 5. S5 → Controller

### 5.1. Feedback — Confirmed types, Proposed envelope

`Feedback` giữ nguyên `HomographyResult` theo giá trị, không nhân đôi field S5. `generation`, `previous_hypotheses` và số R đã hoàn tất thuộc harness/S5 workspace; không cần lặp lại trong feedback Controller.

`execution_status` do harness tổng hợp từ các lời gọi stage và điểm đồng bộ: OK chỉ khi S5 output đã sẵn sàng; nếu lỗi API/workspace/launch/completion thì dùng `INVALID_ARGUMENT`, `INSUFFICIENT_WORKSPACE` hoặc `CUDA_ERROR`. Không gán `NOT_ENOUGH_POINTS`/`DEGENERATE` vào trường này; chúng thuộc `homography.status`.

Đã đối chiếu trực tiếp [prepare/finish/estimate trong S5](../src/s5/s5.cu):

| Trạng thái | Payload được đọc và cách xử lý Proposed |
|---|---|
| `execution_status != OK`, thuộc ba lỗi execution trên | Không đọc bất kỳ trường `homography` nào; REJECT/EXECUTION_ERROR, không retry chất lượng |
| Execution OK, `homography.status == OK` | Kiểm tra toàn bộ model/tín hiệu bên dưới; đạt ngưỡng thì ACCEPT, chưa đạt thì có thể RETRY |
| Execution OK, S5 `NOT_ENOUGH_POINTS` | Chỉ dùng status và `num_matches` hợp lệ trong `[0, 3]`; không đọc H/residual/ratio/coverage để xét ngưỡng; có thể RETRY với S hoặc K khác |
| Execution OK, S5 `DEGENERATE` | Chỉ dùng status và `num_matches >= 4`; bỏ qua model/tín hiệu còn lại vì có thể NaN hoặc không có ý nghĩa; có thể RETRY |
| Execution OK, S5 `INVALID_ARGUMENT`, `INSUFFICIENT_WORKSPACE`, `CUDA_ERROR` | Output báo lỗi contract/execution, không phải chất lượng; REJECT/EXECUTION_ERROR, không retry |
| Enum lạ hoặc payload trái điều kiện của status | Feedback không hợp lệ: Controller trả `INVALID_ARGUMENT`, không sinh Decision và không đổi state |

Counts còn phải nằm trong capacity thật mà harness đã cấp; Controller không có device buffer để kiểm chứng allocation hoặc nguồn gốc output. Không đọc count/model từ pinned output cũ nếu S5 launch/copy/completion thất bại.

### 5.2. Ý nghĩa field khi S5 status OK — Confirmed

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

## 6. Controller → Harness — Proposed

### 6.1. Decision và quy tắc chọn lượt tiếp

| Action | `Decision.next` | Trách nhiệm harness |
|---|---|---|
| ACCEPT | Bằng `feedback.executed`, có hiệu lực để chạy S6 | Dùng H của feedback hiện tại và `next.B`; kiểm tra geometry rồi render. Không lấy H từ lượt khác |
| RETRY | Cấu hình được Controller chọn ở vị trí ladder lớn hơn vị trí hiện tại | Chưa chạy S6; áp dụng invalidation, chạy phần bị ảnh hưởng rồi gửi đúng một feedback của lượt mới |
| REJECT | Bằng `feedback.executed` để xác định lượt kết thúc, không phải lệnh thực thi | Dừng run, không render, không tự chuyển thành ACCEPT hoặc chạy fallback ngoài ladder |

Ladder hữu hạn, không trùng cấu hình. Chỉ **vị trí ladder** phải tiến về phía trước; S/K/R có thể tăng hoặc giảm. Mỗi RETRY phải thay ít nhất một trong S/K/R so với lượt vừa chạy; đổi B đơn thuần không sửa được self-check S5 nên bỏ qua ứng viên như vậy. Không quay về vị trí cũ, số lượt không vượt độ dài ladder. Cấu hình cuối vẫn phải qua self-check.

Nếu S5 báo thiếu 4 match, bỏ qua mọi ứng viên giữ nguyên cả S lẫn K: tăng/giảm R không tạo thêm match. Chọn ứng viên hợp lệ đầu tiên phía sau theo các điều kiện trên; không có thì REJECT/BUDGET_EXHAUSTED. Không yêu cầu S/K tăng chỉ để đáp ứng quy tắc transition; thứ tự ladder do policy riêng cung cấp.

Với S5 OK nhưng fail threshold, reason đầu tiên theo thứ tự: TOO_FEW_INLIERS, LOW_INLIER_RATIO, LOW_INLIER_COVERAGE, HIGH_RESIDUAL. S5 NOT_ENOUGH_POINTS dùng TOO_FEW_MATCHES, DEGENERATE dùng DEGENERATE_MODEL. Khi không còn ứng viên, reason là BUDGET_EXHAUSTED; feedback cuối vẫn giữ nguyên để caller biết lỗi chất lượng. Execution error luôn REJECT/EXECUTION_ERROR trước khi xét threshold.

### 6.2. Bảng state transition

State trong bảng thuộc Controller, không phải trạng thái CUDA hoặc S6.

| State hiện tại | Sự kiện | Output/return | State sau |
|---|---|---|---|
| Uninitialized | `initialize` hợp lệ | OK | Idle |
| Idle hoặc Finished | `initialize` hợp lệ | OK, thay binding ladder | Idle |
| Idle | `begin_pair` hợp lệ | OK, first = cấu hình đầu được chọn | AwaitingFeedback |
| AwaitingFeedback | Feedback OK và self-check pass | OK, ACCEPT/SELF_CHECK_PASSED | Finished |
| AwaitingFeedback | Quality fail, còn ứng viên | OK, RETRY; Controller tăng vị trí trước khi trả | AwaitingFeedback, chờ cấu hình mới |
| AwaitingFeedback | Quality fail, không còn ứng viên | OK, REJECT/BUDGET_EXHAUSTED | Finished |
| AwaitingFeedback | Execution error hợp lệ | OK, REJECT/EXECUTION_ERROR | Finished |
| Bất kỳ | Lỗi input/lifecycle của hàm trả Status | INVALID_ARGUMENT, output không đổi | Không đổi |
| Đã initialize | `reset_pair` | Không output; caller đã kết thúc GPU work cũ | Idle |
| Uninitialized | `reset_pair` | Không output | Uninitialized |

Finished chỉ có nghĩa Controller không nhận thêm feedback cho run này. Sau ACCEPT, run chỉ thành công khi S6 và điểm completion cuối thành công. Gọi `observe` sau ACCEPT/REJECT hoặc gửi lặp feedback lượt trước sau RETRY là lỗi lifecycle/cấu hình, không phải yêu cầu chạy thêm.

### 6.3. Cache reuse/invalidation

Confirmed: S5 đã có API nối tiếp raw RANSAC state. Harness quản lý `generation`, `previous_hypotheses`, địa chỉ/nội dung buffer và điều kiện reuse; Controller không quản lý cache này.

| Thay đổi so với input/cấu hình có trong cache | Có thể giữ | Phải thực hiện |
|---|---|---|
| S/K giữ nguyên, R tăng | S1–S4 và workspace raw S5 nếu toàn bộ điều kiện extension thỏa | S5: `hypotheses = R_new`, `previous_hypotheses = R_old`; chấm thêm hypothesis, refinement/self-check lại |
| S/K giữ nguyên, R giảm | S1–S4 và match | Restart S5 với `previous_hypotheses = 0`; raw winner của R lớn có thể đến từ hypothesis vượt R mới |
| K thay đổi theo bất kỳ chiều nào, S giữ nguyên | S1 | Chạy lại S2–S5, đổi generation; không giả định output K mới là prefix của lần cũ |
| S thay đổi theo bất kỳ chiều nào | RGB gốc | Chạy lại S1–S5, tính ScaleInfo mới, đổi generation |
| Chỉ B thay đổi | H/S1–S5 còn hợp lệ | Chỉ S6 bị ảnh hưởng; không phải một RETRY self-check S5 hợp lệ |
| Đổi pair/repeat, ghi đè input hoặc đổi tham số ngoài S/K/R/B | Chỉ vùng cấp phát còn đủ dung lượng | Reset các kết quả bị phụ thuộc; S5 restart, đổi generation dù địa chỉ pointer không đổi |

Để extension đúng [S5 API](../src/s5/s5.hpp), phải giữ nguyên keypoint/match arrays, counts/capacities, ScaleInfo, seed, threshold, batch size, generation, stream và workspace. Contract giữ cả refinement steps cố định trong một run. `previous_hypotheses` phải bằng R của lần S5 đã hoàn tất trong workspace. Nếu không chứng minh được điều kiện reuse thì restart S5 với `previous_hypotheses = 0`, không giả định cache hợp lệ.

Với execution OK và S5 DEGENERATE trên input đủ match, code hiện tại vẫn lưu trạng thái raw search; có thể extension khi các điều kiện trên còn đúng. Không dùng H/metric của kết quả DEGENERATE để render. Execution/CUDA error kết thúc run; không nối tiếp workspace lỗi. Seed và tham số stage khác không được tự đổi như một phần RETRY.

## 7. Decisions to Freeze

Chỉ các quyết định dưới đây còn cần xác nhận để khóa giao diện; không bao gồm chọn giá trị classifier, threshold hoặc ladder. Hiện chưa mục nào được ghi nhận là approved.

| Quyết định cần xác nhận | Trạng thái | Người phụ trách |
|---|---|---|
| Chấp nhận header host API, Controller sở hữu per-pair ladder position; harness sở hữu dữ liệu/cache và dùng reset/begin/observe như mục 3 | Pending approval | Đế chốt, Tuấn xác nhận tích hợp |
| Chấp nhận schema S1 gồm ScaleInfo và normalized mean absolute gradient riêng cho A/B, đúng ảnh nguồn và công thức mục 4 | Pending approval | Đế và Tuấn |
| Có dùng S1 probe hay không; nếu dùng, phê duyệt điểm completion trước initial selection và bổ sung quy ước synchronization | Pending decision — Pending approval | Tuấn và Đế |
| Chấp nhận envelope/validity S5, action/transition và invalidation mục 5–6, cho phép S/K/R không đơn điệu nhưng vị trí ladder tiến một chiều | Pending approval | Đế chốt, Tuấn xác nhận; Thành và Chương xác nhận ranh giới stage liên quan |
