# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): 
    - "class_id": 468, 
    - "class_name": "cab",
    - "rank": 1,
    - "score": 0.510915,
    - "taxonomy_name": "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào?
    - Record mô tả rằng chủ thể chính hoặc ngữ cảnh bao quát nhất của bức ảnh này là một chiếc taxi dựa trên bộ phân loại ImageNet-1K.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
    -  Danh sách lớp của ImageNet-1K sẽ được định nghĩa ở bước gán nhãn, do người tạo lớp định nghĩa
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
    - Giữ ID để máy tính xử lý tối ưu cho máy tính có thể xác định tại vì máy hiểu được các con số sẽ tốt hơn và định danh duy nhất; giữ tên lớp để con người dễ đọc hiểu; giữ tên taxonomy để xác định hệ quy chiếu cho vật thể.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
    - Guideline cần quy định tiêu chí xác định chủ thể chính hoặc có cho phép gán nhiều nhãn cho cùng một ảnh hay không.
- Vì sao model score không phải ground truth?
    - Vì ground truth là chỉ số thực tế của dữ liệu, model score là xác suất, chỉ số tin tưởng của mô hình đối với record đó.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
    - "class_name": "bus",
    - "score": 0.912558,
    - "bbox_format": "xyxy",
    - "bbox_xyxy": [
      93.17,
      187.95,
      223.01,
      320.91
    ],
    - "bbox_width": 129.84,
    - "bbox_height": 132.96
- Diễn giải vị trí box bằng lời: Model xác định các vật thể về xe buýt (bus) trong ảnh giao thông. Với độ dài và độ rộng của box là 129.8px và 132.96px. Ở vị trí 4 đỉnh của box sẽ là (93.17, 187.95), (93.17, 320.91) (223.01, 187.95), (223.01, 320.91). Với độ chính xác/xác suất là 91.25%
- So sánh số prediction ở hai threshold:
    - Ở threshold thấp, mô hình trả về rất nhiều prediction, bao gồm cả những box có độ tin cậy thấp, dễ dẫn đến nhiều lỗi nhận diện sai. Ở threshold cao, số lượng prediction giảm đi, chỉ giữ lại các box có độ tin cậy cao, giảm False Positive nhưng có nguy cơ bỏ sót vật thể.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
    - Khi threshold thấp: Độ bao phủ cao hơn, nhưng reviewer sẽ phải tốn nhiều thời gian và công sức để xem xét và xóa bỏ các box sai lệch.
    - Khi threshold cao: Khối lượng công việc của reviewer giảm đáng kể, nhưng reviewer cần chú ý kiểm tra và bổ sung các vật thể bị mô hình bỏ sót.
- Đề xuất một quy tắc box chặt:
    - Bounding box phải bao trọn toàn bộ phần có thể nhìn thấy của vật thể. Các cạnh của box phải ôm sát nhất có thể vào các điểm ngoài cùng của vật thể, khoảng cách viền thừa không được vượt quá 2-3 pixel.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
    - Guideline cần quy định rõ tỷ lệ hiển thị tối thiểu để được gán nhãn. Ngoài ra, cần quy định rõ box sẽ chỉ bao quanh phần nhìn thấy được hay ước lượng cả phần bị che khuất. Khi vật thể bị che khuất nghiêm trọng đến mức không thể nhận dạng chắc chắn, annotator cần escalation để xin ý kiến.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
```text
{
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "traffic-001",
    "class_id": 5,
    "class_name": "bus",
    "score": 0.925745,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      95.3,
      188.72,
      224.08,
      319.96
    ],
    "polygon_point_count": 120,
    "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0
      ],
      [
        143.0,
        192.0
      ],
      [
        142.0,
        192.0
      ],
    ...
```
- Polygon bổ sung chi tiết gì so với box?
    - Polygon cung cấp ranh giới hình học chính xác của vật thể, ôm sát đường viền thực tế thay vì chỉ là một hình chữ nhật bao quanh. Nó giúp phân định rõ các pixel nào thực sự thuộc về vật thể, loại bỏ các phần nhiễu/nền hay các vật thể khác nằm bên trong bounding box.
- `instance_id` dùng để làm gì và không phải loại ID nào?
    - `instance_id` dùng để định danh duy nhất cho từng cá thể cụ thể trong một bức ảnh. Nó không phải là `class_id` và thường không dùng làm ID toàn cục xuyên suốt nhiều ảnh.
- Đề xuất một quy tắc biên mask:
    - Biên mask phải bám sát viền thực tế của vật thể. Không được vẽ lẹm vào bên trong vật thể và không được vẽ thừa ra ngoài nền. Sai số biên cho phép thường được giới hạn ở mức 1-2 pixel.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
    - Guideline cần quy định cách xử lý viền mờ. Khi vật thể bị che khuất chia làm nhiều phần, cần quy định việc vẽ các polygon rời rạc hay vẽ một mask duy nhất. Nếu vật thể bị che khuất quá mức khó xác định đường viền, annotator cần escalation để xử lý.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
