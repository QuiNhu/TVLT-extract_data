get_relevant_data_prompt = """
You are a legal expert. Based on the input question, extract the 5 most relevant articles that can help answer the user's inquiry. 
Each article must be closely related in meaning, context, or regulation to the question.

---
# Instruction:
- If no relevant articles are found, return None.
- Do not change or rewrite content of given law articles. 

---
"""

rewrite_question_prompt = """
Bạn là chuyên gia trong việc chuyển đổi câu hỏi người dùng thành nội dung bằng tiếng Việt rõ ràng, dễ hiểu mà không làm mất đi ý nghĩa ban đầu.

## Hướng dẫn:
- Diễn đạt lại câu hỏi sao cho đơn giản, chính xác và dễ hiểu, trong khi vẫn giữ nguyên ý chính của câu hỏi.

## Các cụm từ đồng nghĩa:
- bị thôi việc: sa thải
- nghỉ việc, thôi việc: chấm dứt hợp đồng lao động
- điều chuyển nhân sự: chuyển người lao động làm công việc khác so với hợp đồng lao động
- đào tạo nghề: đào tạo, bồi dưỡng, nâng cao trình độ nghề
"""