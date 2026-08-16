# Báo cáo benchmark LightGBM trên AWS CPU

1. Benchmark được thực hiện trên dataset Credit Card Fraud Detection gồm 284.807 giao dịch.
2. Thời gian tải dữ liệu là 2,1986 giây và thời gian huấn luyện LightGBM là 1,7023 giây.
3. Mô hình đạt AUC-ROC 0,951654, cho thấy khả năng phân biệt giao dịch gian lận khá tốt.
4. Accuracy đạt 0,998947; tuy nhiên, chỉ số này cần được xem cùng các metric khác do dữ liệu gian lận bị mất cân bằng.
5. F1-Score đạt 0,727273, Precision đạt 0,655738 và Recall đạt 0,816327.
6. Recall cao cho thấy mô hình phát hiện được phần lớn các giao dịch gian lận, dù vẫn có một số cảnh báo dương tính giả.
7. Inference một dòng có latency 1,3037 ms; khi dự đoán 1.000 dòng, throughput đạt khoảng 562.763 dòng/giây.
8. Kết quả cho thấy instance CPU `t3.medium` đáp ứng tốt bài toán LightGBM mà không cần GPU.
