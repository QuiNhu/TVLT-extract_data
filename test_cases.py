from typing import List, Text

from transformers.models.blenderbot.convert_blenderbot_original_pytorch_checkpoint_to_pytorch import \
    rename_layernorm_keys

from relevant_extractor.domain import read_json_file
from relevant_extractor.application import get_relevant_articles

test_cases = [
    {
        "question": "NLĐ bị sa thải có được trả lương hay không?",
        "message": "Extract relevant articles success!!",
        "file_path": "45_2019_QH14_333670.txt",
        "queries": {
            "dieu_34": {
                "title": "Các trường hợp chấm dứt hợp đồng lao động",
                "text": "1.\nHết hạn hợp đồng lao động, trừ trường hợp quy định tại khoản 4 Điều 177 của Bộ luật này.\n2.\nĐã hoàn thành công việc theo hợp đồng lao động.\n3.\nHai bên thỏa thuận chấm dứt hợp đồng lao động.\n4.\nNgười lao động bị kết án phạt tù nhưng không được hưởng án treo hoặc không thuộc trường hợp được trả tự do theo quy định tại khoản 5 Điều 328 của Bộ luật Tố tụng hình sự, tử hình hoặc bị cấm làm công việc ghi trong hợp đồng lao động theo bản án, quyết định của Tòa án đã có hiệu lực pháp luật.\n5.\nNgười lao động là người nước ngoài làm việc tại Việt Nam bị trục xuất theo bản án, quyết định của Tòa án đã có hiệu lực pháp luật, quyết định của cơ quan nhà nước có thẩm quyền.\n6.\nNgười lao động chết; bị Tòa án tuyên bố mất năng lực hành vi dân sự, mất tích hoặc đã chết.\n7.\nNgười sử dụng lao động là cá nhân chết; bị Tòa án tuyên bố mất năng lực hành vi dân sự, mất tích hoặc đã chết.\nNgười sử dụng lao động không phải là cá nhân chấm dứt hoạt động hoặc bị cơ quan chuyên môn về đăng ký kinh doanh thuộc Ủy ban nhân dân cấp tỉnh ra thông báo không có người đại diện theo pháp luật, người được ủy quyền thực hiện quyền và nghĩa vụ của người đại diện theo pháp luật.\n8.\nNgười lao động bị xử lý kỷ luật sa thải.\n9.\nNgười lao động đơn phương chấm dứt hợp đồng lao động theo quy định tại Điều 35 của Bộ luật này.\n10.\nNgười sử dụng lao động đơn phương chấm dứt hợp đồng lao động theo quy định tại Điều 36 của Bộ luật này.\n11.\nNgười sử dụng lao động cho người lao động thôi việc theo quy định tại Điều 42 và Điều 43 của Bộ luật này.\n12.\nGiấy phép lao động hết hiệu lực đối với người lao động là người nước ngoài làm việc tại Việt Nam theo quy định tại Điều 156 của Bộ luật này.\n13.\nTrường hợp thỏa thuận nội dung thử việc ghi trong hợp đồng lao động mà thử việc không đạt yêu cầu hoặc một bên hủy bỏ thỏa thuận thử việc."
            },
            "dieu_48": {
                "title": "Trách nhiệm khi chấm dứt hợp đồng lao động",
                "text": "1.\nTrong thời hạn 14 ngày làm việc kể từ ngày chấm dứt hợp đồng lao động, hai bên có trách nhiệm thanh toán đầy đủ các khoản tiền có liên quan đến quyền lợi của mỗi bên, trừ trường hợp sau đây có thể kéo dài nhưng không được quá 30 ngày:\na) Người sử dụng lao động không phải là cá nhân chấm dứt hoạt động;\nb) Người sử dụng lao động thay đổi cơ cấu, công nghệ hoặc vì lý do kinh tế;\nc) Chia, tách, hợp nhất, sáp nhập; bán, cho thuê, chuyển đổi loại hình doanh nghiệp; chuyển nhượng quyền sở hữu, quyền sử dụng tài sản của doanh nghiệp, hợp tác xã;\nd) Do thiên tai, hỏa hoạn, địch họa hoặc dịch bệnh nguy hiểm.\n2.\nTiền lương, bảo hiểm xã hội, bảo hiểm y tế, bảo hiểm thất nghiệp, trợ cấp thôi việc và các quyền lợi khác của người lao động theo thỏa ước lao động tập thể, hợp đồng lao động được ưu tiên thanh toán trong trường hợp doanh nghiệp, hợp tác xã bị chấm dứt hoạt động, bị giải thể, phá sản.\n3.\nNgười sử dụng lao động có trách nhiệm sau đây:\na) Hoàn thành thủ tục xác nhận thời gian đóng bảo hiểm xã hội, bảo hiểm thất nghiệp và trả lại cùng với bản chính giấy tờ khác nếu người sử dụng lao động đã giữ của người lao động;\nb) Cung cấp bản sao các tài liệu liên quan đến quá trình làm việc của người lao động nếu người lao động có yêu cầu.\nChi phí sao, gửi tài liệu do người sử dụng lao động trả."
            },
            "dieu_125": {
                "title": "Áp dụng hình thức xử lý kỷ luật sa thải",
                "text": "Hình thức xử lý kỷ luật sa thải được người sử dụng lao động áp dụng trong trường hợp sau đây:\n1.\nNgười lao động có hành vi trộm cắp, tham ô, đánh bạc, cố ý gây thương tích, sử dụng ma túy tại nơi làm việc;\n2.\nNgười lao động có hành vi tiết lộ bí mật kinh doanh, bí mật công nghệ, xâm phạm quyền sở hữu trí tuệ của người sử dụng lao động, có hành vi gây thiệt hại nghiêm trọng hoặc đe dọa gây thiệt hại đặc biệt nghiêm trọng về tài sản, lợi ích của người sử dụng lao động hoặc quấy rối tình dục tại nơi làm việc được quy định trong nội quy lao động;\n3.\nNgười lao động bị xử lý kỷ luật kéo dài thời hạn nâng lương hoặc cách chức mà tái phạm trong thời gian chưa xóa kỷ luật.\nTái phạm là trường hợp người lao động lặp lại hành vi vi phạm đã bị xử lý kỷ luật mà chưa được xóa kỷ luật theo quy định tại Điều 126 của Bộ luật này;\n4.\nNgười lao động tự ý bỏ việc 05 ngày cộng dồn trong thời hạn 30 ngày hoặc 20 ngày cộng dồn trong thời hạn 365 ngày tính từ ngày đầu tiên tự ý bỏ việc mà không có lý do chính đáng.\nTrường hợp được coi là có lý do chính đáng bao gồm thiên tai, hỏa hoạn, bản thân, thân nhân bị ốm có xác nhận của cơ sở khám bệnh, chữa bệnh có thẩm quyền và trường hợp khác được quy định trong nội quy lao động."
            }
        }
    },
    {
        "question": "Người sử dụng lao động được sa thải người lao động nữ đang mang thai không?",
        "message": "Extract relevant articles success!!",
        "file_path": "45_2019_QH14_333670.txt",
        "queries": {
            "dieu_137": {
                "title": "Bảo vệ thai sản",
                "text": "1.\nNgười sử dụng lao động không được sử dụng người lao động làm việc ban đêm, làm thêm giờ và đi công tác xa trong trường hợp sau đây:\na) Mang thai từ tháng thứ 07 hoặc từ tháng thứ 06 nếu làm việc ở vùng cao, vùng sâu, vùng xa, biên giới, hải đảo;\nb) Đang nuôi con dưới 12 tháng tuổi, trừ trường hợp được người lao động đồng ý.\n2.\nLao động nữ làm nghề, công việc nặng nhọc, độc hại, nguy hiểm hoặc đặc biệt nặng nhọc, độc hại, nguy hiểm hoặc làm nghề, công việc có ảnh hưởng xấu tới chức năng sinh sản và nuôi con khi mang thai và có thông báo cho người sử dụng lao động biết thì được người sử dụng lao động chuyển sang làm công việc nhẹ hơn, an toàn hơn hoặc giảm bớt 01 giờ làm việc hằng ngày mà không bị cắt giảm tiền lương và quyền, lợi ích cho đến hết thời gian nuôi con dưới 12 tháng tuổi.\n3.\nNgười sử dụng lao động không được sa thải hoặc đơn phương chấm dứt hợp đồng lao động đối với người lao động vì lý do kết hôn, mang thai, nghỉ thai sản, nuôi con dưới 12 tháng tuổi, trừ trường hợp người sử dụng lao động là cá nhân chết, bị Tòa án tuyên bố mất năng lực hành vi dân sự, mất tích hoặc đã chết hoặc người sử dụng lao động không phải là cá nhân chấm dứt hoạt động hoặc bị cơ quan chuyên môn về đăng ký kinh doanh thuộc Ủy ban nhân dân cấp tỉnh ra thông báo không có người đại diện theo pháp luật, người được ủy quyền thực hiện quyền và nghĩa vụ của người đại diện theo pháp luật.\nTrường hợp hợp đồng lao động hết hạn trong thời gian lao động nữ mang thai hoặc nuôi con dưới 12 tháng tuổi thì được ưu tiên giao kết hợp đồng lao động mới.\n4.\nLao động nữ trong thời gian hành kinh được nghỉ mỗi ngày 30 phút, trong thời gian nuôi con dưới 12 tháng tuổi được nghỉ mỗi ngày 60 phút trong thời gian làm việc.\nThời gian nghỉ vẫn được hưởng đủ tiền lương theo hợp đồng lao động."
            }
        }
    },
    {
        "question": "Quy định về điều chuyển nhân sự được quy định như thế nào?",
        "message": "Extract relevant articles success!!",
        "file_path": "45_2019_QH14_333670.txt",
        "queries": {
            "dieu_29": {
                "title": "Chuyển người lao động làm công việc khác so với hợp đồng lao động",
                "text": "1.\nKhi gặp khó khăn đột xuất do thiên tai, hỏa hoạn, dịch bệnh nguy hiểm, áp dụng biện pháp ngăn ngừa, khắc phục tai nạn lao động, bệnh nghề nghiệp, sự cố điện, nước hoặc do nhu cầu sản xuất, kinh doanh thì người sử dụng lao động được quyền tạm thời chuyển người lao động làm công việc khác so với hợp đồng lao động nhưng không được quá 60 ngày làm việc cộng dồn trong 01 năm; trường hợp chuyển người lao động làm công việc khác so với hợp đồng lao động quá 60 ngày làm việc cộng dồn trong 01 năm thì chỉ được thực hiện khi người lao động đồng ý bằng văn bản.\nNgười sử dụng lao động quy định cụ thể trong nội quy lao động những trường hợp do nhu cầu sản xuất, kinh doanh mà người sử dụng lao động được tạm thời chuyển người lao động làm công việc khác so với hợp đồng lao động.\n2.\nKhi tạm thời chuyển người lao động làm công việc khác so với hợp đồng lao động quy định tại khoản 1 Điều này, người sử dụng lao động phải báo cho người lao động biết trước ít nhất 03 ngày làm việc, thông báo rõ thời hạn làm tạm thời và bố trí công việc phù hợp với sức khỏe, giới tính của người lao động.\n3.\nNgười lao động chuyển sang làm công việc khác so với hợp đồng lao động được trả lương theo công việc mới.\nNếu tiền lương của công việc mới thấp hơn tiền lương của công việc cũ thì được giữ nguyên tiền lương của công việc cũ trong thời hạn 30 ngày làm việc.\nTiền lương theo công việc mới ít nhất phải bằng 85% tiền lương của công việc cũ nhưng không thấp hơn mức lương tối thiểu.\n4.\nNgười lao động không đồng ý tạm thời làm công việc khác so với hợp đồng lao động quá 60 ngày làm việc cộng dồn trong 01 năm mà phải ngừng việc thì người sử dụng lao động phải trả lương ngừng việc theo quy định tại Điều 99 của Bộ luật này."
            },
            "dieu_99": {
                "title": "Tiền lương ngừng việc",
                "text": "Trường hợp phải ngừng việc, người lao động được trả lương như sau:\n1.\nNếu do lỗi của người sử dụng lao động thì người lao động được trả đủ tiền lương theo hợp đồng lao động;\n2.\nNếu do lỗi của người lao động thì người đó không được trả lương; những người lao động khác trong cùng đơn vị phải ngừng việc thì được trả lương theo mức do hai bên thỏa thuận nhưng không được thấp hơn mức lương tối thiểu;\n3.\nNếu vì sự cố về điện, nước mà không do lỗi của người sử dụng lao động hoặc do thiên tai, hỏa hoạn, dịch bệnh nguy hiểm, địch họa, di dời địa điểm hoạt động theo yêu cầu của cơ quan nhà nước có thẩm quyền hoặc vì lý do kinh tế thì hai bên thỏa thuận về tiền lương ngừng việc như sau:\na) Trường hợp ngừng việc từ 14 ngày làm việc trở xuống thì tiền lương ngừng việc được thỏa thuận không thấp hơn mức lương tối thiểu;\nb) Trường hợp phải ngừng việc trên 14 ngày làm việc thì tiền lương ngừng việc do hai bên thỏa thuận nhưng phải bảo đảm tiền lương ngừng việc trong 14 ngày đầu tiên không thấp hơn mức lương tối thiểu."
            }
        }
    },
    {
        "question": "Người sử dụng lao động đào tạo nghề nghiệp và phát triển kỹ năng nghề cho người lao động như thế nào?",
        "message": "Extract relevant articles success!!",
        "file_path": "45_2019_QH14_333670.txt",
        "queries": {
            "dieu_61": {
                "title": "Học nghề, tập nghề để làm việc cho người sử dụng lao động",
                "text": "1.\nHọc nghề để làm việc cho người sử dụng lao động là việc người sử dụng lao động tuyển người vào để đào tạo nghề nghiệp tại nơi làm việc.\nThời gian học nghề theo chương trình đào tạo của từng trình độ theo quy định của Luật Giáo dục nghề nghiệp.\n2.\nTập nghề để làm việc cho người sử dụng lao động là việc người sử dụng lao động tuyển người vào để hướng dẫn thực hành công việc, tập làm nghề theo vị trí việc làm tại nơi làm việc.\nThời hạn tập nghề không quá 03 tháng.\n3.\nNgười sử dụng lao động tuyển người vào học nghề, tập nghề để làm việc cho mình thì không phải đăng ký hoạt động giáo dục nghề nghiệp; không được thu học phí; phải ký hợp đồng đào tạo theo quy định của Luật Giáo dục nghề nghiệp.\n4.\nNgười học nghề, người tập nghề phải đủ 14 tuổi trở lên và phải có đủ sức khỏe phù hợp với yêu cầu học nghề, tập nghề.\nNgười học nghề, người tập nghề thuộc danh mục nghề, công việc nặng nhọc, độc hại, nguy hiểm hoặc đặc biệt nặng nhọc, độc hại, nguy hiểm do Bộ trưởng Bộ Lao động - Thương binh và Xã hội ban hành phải từ đủ 18 tuổi trở lên, trừ lĩnh vực nghệ thuật, thể dục, thể thao.\n5.\nTrong thời gian học nghề, tập nghề, nếu người học nghề, người tập nghề trực tiếp hoặc tham gia lao động thì được người sử dụng lao động trả lương theo mức do hai bên thỏa thuận.\n6.\nHết thời hạn học nghề, tập nghề, hai bên phải ký kết hợp đồng lao động khi đủ các điều kiện theo quy định của Bộ luật này."
            },
            "dieu_62": {
                "title": "Hợp đồng đào tạo nghề giữa người sử dụng lao động, người lao động và chi phí đào tạo nghề",
                "text": "1.\nHai bên phải ký kết hợp đồng đào tạo nghề trong trường hợp người lao động được đào tạo nâng cao trình độ, kỹ năng nghề, đào tạo lại ở trong nước hoặc nước ngoài từ kinh phí của người sử dụng lao động, kể cả kinh phí do đối tác tài trợ cho người sử dụng lao động.\nHợp đồng đào tạo nghề phải làm thành 02 bản, mỗi bên giữ 01 bản.\n2.\nHợp đồng đào tạo nghề phải có các nội dung chủ yếu sau đây:\na) Nghề đào tạo;\nb) Địa điểm, thời gian và tiền lương trong thời gian đào tạo;\nc) Thời hạn cam kết phải làm việc sau khi được đào tạo;\nd) Chi phí đào tạo và trách nhiệm hoàn trả chi phí đào tạo;\nđ) Trách nhiệm của người sử dụng lao động;\ne) Trách nhiệm của người lao động.\n3.\nChi phí đào tạo bao gồm các khoản chi có chứng từ hợp lệ về chi phí trả cho người dạy, tài liệu học tập, trường, lớp, máy, thiết bị, vật liệu thực hành, các chi phí khác hỗ trợ cho người học và tiền lương, tiền đóng bảo hiểm xã hội, bảo hiểm y tế, bảo hiểm thất nghiệp cho người học trong thời gian đi học.\nTrường hợp người lao động được gửi đi đào tạo ở nước ngoài thì chi phí đào tạo còn bao gồm chi phí đi lại, chi phí sinh hoạt trong thời gian đào tạo."
            },
            "dieu_59": {
                "title": "Đào tạo nghề nghiệp và phát triển kỹ năng nghề",
                "text": "1.\nNgười lao động được tự do lựa chọn đào tạo nghề nghiệp, tham gia đánh giá, công nhận kỹ năng nghề quốc gia, phát triển năng lực nghề nghiệp phù hợp với nhu cầu việc làm và khả năng của mình.\n2.\nNhà nước có chính sách khuyến khích người sử dụng lao động có đủ điều kiện đào tạo nghề nghiệp và phát triển kỹ năng nghề cho người lao động đang làm việc cho mình và người lao động khác trong xã hội thông qua hoạt động sau đây:\na) Thành lập cơ sở giáo dục nghề nghiệp hoặc mở lớp đào tạo nghề tại nơi làm việc để đào tạo, đào tạo lại, bồi dưỡng, nâng cao trình độ, kỹ năng nghề cho người lao động; phối hợp với cơ sở giáo dục nghề nghiệp đào tạo các trình độ sơ cấp, trung cấp, cao đẳng và các chương trình đào tạo nghề nghiệp khác theo quy định;\nb) Tổ chức thi kỹ năng nghề cho người lao động; tham gia hội đồng kỹ năng nghề; dự báo nhu cầu và xây dựng tiêu chuẩn kỹ năng nghề; tổ chức đánh giá và công nhận kỹ năng nghề; phát triển năng lực nghề nghiệp cho người lao động."
            }
        }
    },
    {
        "question": "Người lao động được thuê làm giám đốc doanh nghiệp Nhà nước được hưởng các chế độ về tiền lương, thưởng như thế nào?",
        "message": "Extract relevant articles success!!",
        "file_path": "145_2020_ND-CP_459400.txt",
        "queries": {
            "dieu_5": {
                "title": "Nội dung hợp đồng lao động đối với người lao động được thuê làm giám đốc trong doanh nghiệp do Nhà nước nắm giữ 100% vốn điều lệ hoặc Nhà nước nắm giữ trên 50% vốn điều lệ hoặc tổng số cổ phần có quyền biểu quyết",
                "text": "Hợp đồng lao động đối với người lao động được thuê làm giám đốc trong doanh nghiệp do Nhà nước nắm giữ 100% vốn điều lệ hoặc Nhà nước nắm giữ trên 50% vốn điều lệ hoặc tổng số cổ phần có quyền biểu quyết tại khoản 4 Điều 21 của Bộ luật Lao động gồm những nội dung chủ yếu:\n1.\nTên, địa chỉ trụ sở chính của doanh nghiệp theo giấy chứng nhận đăng ký doanh nghiệp; họ tên, ngày tháng năm sinh, số thẻ Căn cước công dân hoặc Chứng minh nhân dân hoặc hộ chiếu, số điện thoại, địa chỉ liên lạc của Chủ tịch Hội đồng thành viên hoặc Chủ tịch công ty hoặc Chủ tịch Hội đồng quản trị.\n2.\nHọ tên; ngày tháng năm sinh; giới tính; quốc tịch; trình độ đào tạo; địa chỉ nơi cư trú tại Việt Nam, địa chỉ nơi cư trú tại nước ngoài (đối với người lao động là người nước ngoài); số thẻ Căn cước công dân hoặc Chứng minh nhân dân hoặc hộ chiếu; số điện thoại, địa chỉ liên lạc; số Giấy phép lao động do cơ quan nhà nước có thẩm quyền cấp hoặc văn bản xác nhận không thuộc diện cấp Giấy phép lao động; các giấy tờ khác theo yêu cầu của người sử dụng lao động (đối với người lao động là người nước ngoài) nếu có của người lao động được thuê làm giám đốc.\n3.\nCông việc được làm, không được làm và nghĩa vụ gắn với kết quả thực hiện công việc của người lao động được thuê làm giám đốc.\n4.\nĐịa điểm làm việc của người lao động được thuê làm giám đốc.\n5.\nThời hạn của hợp đồng lao động do hai bên thỏa thuận tối đa không quá 36 tháng.\nĐối với người lao động là người nước ngoài được thuê làm giám đốc thì thời hạn hợp đồng lao động không vượt quá thời hạn của Giấy phép lao động do cơ quan nhà nước có thẩm quyền cấp.\n6.\nNội dung, thời hạn, trách nhiệm bảo vệ bí mật kinh doanh, bí mật công nghệ của doanh nghiệp đối với người lao động được thuê làm giám đốc và xử lý vi phạm.\n7.\nQuyền và nghĩa vụ của người sử dụng lao động, bao gồm:\na) Cung cấp thông tin cho người lao động được thuê làm giám đốc để thực hiện nhiệm vụ;\nb) Kiểm tra, giám sát, đánh giá hiệu quả thực hiện công việc của người được thuê làm giám đốc;\nc) Các quyền và nghĩa vụ theo quy định của pháp luật;\nd) Ban hành quy chế làm việc đối với giám đốc;\nđ) Thực hiện nghĩa vụ đối với người lao động được thuê làm giám đốc về: trả lương, thưởng; đóng bảo hiểm xã hội, bảo hiểm y tế, bảo hiểm thất nghiệp; trang bị phương tiện làm việc, đi lại, ăn, ở; đào tạo, bồi dưỡng;\ne) Các quyền và nghĩa vụ khác do hai bên thỏa thuận.\n8.\nQuyền và nghĩa vụ của người lao động được thuê làm giám đốc, bao gồm:\na) Thực hiện các công việc theo hợp đồng lao động;\nb) Báo cáo, đề xuất giải pháp xử lý những khó khăn, vướng mắc trong quá trình thực hiện công việc theo hợp đồng lao động;\nc) Báo cáo tình hình quản lý, sử dụng về vốn, tài sản, lao động và các nguồn lực khác;\nd) Được hưởng các chế độ về: tiền lương, thưởng; thời giờ làm việc, thời giờ nghỉ ngơi; trang bị phương tiện làm việc, đi lại, ăn, ở; bảo hiểm xã hội, bảo hiểm y tế, bảo hiểm thất nghiệp; đào tạo, bồi dưỡng; chế độ khác do hai bên thỏa thuận;\nđ) Các quyền và nghĩa vụ khác do hai bên thỏa thuận.\n9.\nĐiều kiện, quy trình, thủ tục sửa đổi, bổ sung hợp đồng lao động, đơn phương chấm dứt hợp đồng lao động.\n10.\nQuyền và nghĩa vụ của người sử dụng lao động và người lao động được thuê làm giám đốc khi chấm dứt hợp đồng lao động.\n11.\nKỷ luật lao động, trách nhiệm vật chất, giải quyết tranh chấp lao động và khiếu nại.\n12.\nCác nội dung khác do hai bên thỏa thuận."
            }
        }
    },
    {
        "question": "Làm việc 8h một ngày thì được nghỉ giữa giờ ít nhất bao nhiêu phút?",
        "message": "Extract relevant articles success!!",
        "file_path": "45_2019_QH14_333670.txt",
        "queries": {
            "dieu_105": {
                "title": "Thời giờ làm việc bình thường",
                "text": "1.\nThời giờ làm việc bình thường không quá 08 giờ trong 01 ngày và không quá 48 giờ trong 01 tuần.\n2.\nNgười sử dụng lao động có quyền quy định thời giờ làm việc theo ngày hoặc tuần nhưng phải thông báo cho người lao động biết; trường hợp theo tuần thì thời giờ làm việc bình thường không quá 10 giờ trong 01 ngày và không quá 48 giờ trong 01 tuần.\nNhà nước khuyến khích người sử dụng lao động thực hiện tuần làm việc 40 giờ đối với người lao động.\n3.\nNgười sử dụng lao động có trách nhiệm bảo đảm giới hạn thời gian làm việc tiếp xúc với yếu tố nguy hiểm, yếu tố có hại đúng theo quy chuẩn kỹ thuật quốc gia và pháp luật có liên quan."
            },
            "dieu_109": {
                "title": "Nghỉ trong giờ làm việc",
                "text": "1.\nNgười lao động làm việc theo thời giờ làm việc quy định tại Điều 105 của Bộ luật này từ 06 giờ trở lên trong một ngày thì được nghỉ giữa giờ ít nhất 30 phút liên tục, làm việc ban đêm thì được nghỉ giữa giờ ít nhất 45 phút liên tục.\nTrường hợp người lao động làm việc theo ca liên tục từ 06 giờ trở lên thì thời gian nghỉ giữa giờ được tính vào giờ làm việc.\n2.\nNgoài thời gian nghỉ quy định tại khoản 1 Điều này, người sử dụng lao động bố trí cho người lao động các đợt nghỉ giải lao và ghi vào nội quy lao động."
            }
        }
    },
    {
        "question": "Nguyên tắc cho thuê lại lao động là gì?",
        "message": "Extract relevant articles success!!",
        "file_path": "45_2019_QH14_333670.txt",
        "queries": {
            "dieu_53": {
                "title": "Nguyên tắc hoạt động cho thuê lại lao động",
                "text": "1.\nThời hạn cho thuê lại lao động đối với người lao động tối đa là 12 tháng.\n2.\nBên thuê lại lao động được sử dụng lao động thuê lại trong trường hợp sau đây:\na) Đáp ứng tạm thời sự gia tăng đột ngột về nhu cầu sử dụng lao động trong khoảng thời gian nhất định;\nb) Thay thế người lao động trong thời gian nghỉ thai sản, bị tai nạn lao động, bệnh nghề nghiệp hoặc phải thực hiện các nghĩa vụ công dân;\nc) Có nhu cầu sử dụng lao động trình độ chuyên môn, kỹ thuật cao.\n3.\nBên thuê lại lao động không được sử dụng lao động thuê lại trong trường hợp sau đây:\na) Để thay thế người lao động đang trong thời gian thực hiện quyền đình công, giải quyết tranh chấp lao động;\nb) Không có thỏa thuận cụ thể về trách nhiệm bồi thường tai nạn lao động, bệnh nghề nghiệp của người lao động thuê lại với doanh nghiệp cho thuê lại lao động;\nc) Thay thế người lao động bị cho thôi việc do thay đổi cơ cấu, công nghệ, vì lý do kinh tế hoặc chia, tách, hợp nhất, sáp nhập.\n4.\nBên thuê lại lao động không được chuyển người lao động thuê lại cho người sử dụng lao động khác; không được sử dụng người lao động thuê lại được cung cấp bởi doanh nghiệp không có Giấy phép hoạt động cho thuê lại lao động."
            }
        }
    },
    {
        "question": "Thời hạn của thỏa ước lao động tập thể như thế nào?",
        "message": "Extract relevant articles success!!",
        "file_path": "45_2019_QH14_333670.txt",
        "queries": {
            "dieu_78": {
                "title": "Hiệu lực và thời hạn của thỏa ước lao động tập thể",
                "text": "1.\nNgày có hiệu lực của thỏa ước lao động tập thể do các bên thỏa thuận và được ghi trong thỏa ước.\nTrường hợp các bên không thỏa thuận ngày có hiệu lực thì thỏa ước lao động tập thể có hiệu lực kể từ ngày ký kết.\nThỏa ước lao động tập thể sau khi có hiệu lực phải được các bên tôn trọng thực hiện.\n2.\nThỏa ước lao động tập thể doanh nghiệp có hiệu lực áp dụng đối với người sử dụng lao động và toàn bộ người lao động của doanh nghiệp.\nThỏa ước lao động tập thể ngành và thỏa ước lao động tập thể có nhiều doanh nghiệp có hiệu lực áp dụng đối với toàn bộ người sử dụng lao động và người lao động của các doanh nghiệp tham gia thỏa ước lao động tập thể.\n3.\nThỏa ước lao động tập thể có thời hạn từ 01 năm đến 03 năm.\nThời hạn cụ thể do các bên thỏa thuận và ghi trong thỏa ước lao động tập thể.\nCác bên có quyền thỏa thuận thời hạn khác nhau đối với các nội dung của thỏa ước lao động tập thể."
            },
            "dieu_83": {
                "title": "Thỏa ước lao động tập thể hết hạn",
                "text": "Trong thời hạn 90 ngày trước ngày thỏa ước lao động tập thể hết hạn, các bên có thể thương lượng để kéo dài thời hạn của thỏa ước lao động tập thể hoặc ký kết thỏa ước lao động tập thể mới.\nTrường hợp các bên thỏa thuận kéo dài thời hạn của thỏa ước lao động tập thể thì phải lấy ý kiến theo quy định tại Điều 76 của Bộ luật này.\nKhi thỏa ước lao động tập thể hết hạn mà các bên vẫn tiếp tục thương lượng thì thỏa ước lao động tập thể cũ vẫn được tiếp tục thực hiện trong thời hạn không quá 90 ngày kể từ ngày thỏa ước lao động tập thể hết hạn, trừ trường hợp các bên có thỏa thuận khác."
            }
        }
    },
    {
        "question": "Hợp đồng lao động được giao kết theo hình thức nào?",
        "message": "Extract relevant articles success!!",
        "file_path": "45_2019_QH14_333670.txt",
        "queries": {
            "dieu_13": {
                "title": "Hợp đồng lao động",
                "text": "1.\nHợp đồng lao động là sự thỏa thuận giữa người lao động và người sử dụng lao động về việc làm có trả công, tiền lương, điều kiện lao động, quyền và nghĩa vụ của mỗi bên trong quan hệ lao động.\nTrường hợp hai bên thỏa thuận bằng tên gọi khác nhưng có nội dung thể hiện về việc làm có trả công, tiền lương và sự quản lý, điều hành, giám sát của một bên thì được coi là hợp đồng lao động.\n2.\nTrước khi nhận người lao động vào làm việc thì người sử dụng lao động phải giao kết hợp đồng lao động với người lao động."
            },
            "dieu_14": {
                "title": "Hình thức hợp đồng lao động",
                "text": "1.\nHợp đồng lao động phải được giao kết bằng văn bản và được làm thành 02 bản, người lao động giữ 01 bản, người sử dụng lao động giữ 01 bản, trừ trường hợp quy định tại khoản 2 Điều này.\nHợp đồng lao động được giao kết thông qua phương tiện điện tử dưới hình thức thông điệp dữ liệu theo quy định của pháp luật về giao dịch điện tử có giá trị như hợp đồng lao động bằng văn bản.\n2.\nHai bên có thể giao kết hợp đồng lao động bằng lời nói đối với hợp đồng có thời hạn dưới 01 tháng, trừ trường hợp quy định tại khoản 2 Điều 18, điểm a khoản 1 Điều 145 và khoản 1 Điều 162 của Bộ luật này."
            },
            "dieu_20": {
                "title": "Loại hợp đồng lao động",
                "text": "1.\nHợp đồng lao động phải được giao kết theo một trong các loại sau đây:\na) Hợp đồng lao động không xác định thời hạn là hợp đồng mà trong đó hai bên không xác định thời hạn, thời điểm chấm dứt hiệu lực của hợp đồng;\nb) Hợp đồng lao động xác định thời hạn là hợp đồng mà trong đó hai bên xác định thời hạn, thời điểm chấm dứt hiệu lực của hợp đồng trong thời gian không quá 36 tháng kể từ thời điểm có hiệu lực của hợp đồng.\n2.\nKhi hợp đồng lao động quy định tại điểm b khoản 1 Điều này hết hạn mà người lao động vẫn tiếp tục làm việc thì thực hiện như sau:\na) Trong thời hạn 30 ngày kể từ ngày hợp đồng lao động hết hạn, hai bên phải ký kết hợp đồng lao động mới; trong thời gian chưa ký kết hợp đồng lao động mới thì quyền, nghĩa vụ và lợi ích của hai bên được thực hiện theo hợp đồng đã giao kết;\nb) Nếu hết thời hạn 30 ngày kể từ ngày hợp đồng lao động hết hạn mà hai bên không ký kết hợp đồng lao động mới thì hợp đồng đã giao kết theo quy định tại điểm b khoản 1 Điều này trở thành hợp đồng lao động không xác định thời hạn;\nc) Trường hợp hai bên ký kết hợp đồng lao động mới là hợp đồng lao động xác định thời hạn thì cũng chỉ được ký thêm 01 lần, sau đó nếu người lao động vẫn tiếp tục làm việc thì phải ký kết hợp đồng lao động không xác định thời hạn, trừ hợp đồng lao động đối với người được thuê làm giám đốc trong doanh nghiệp có vốn nhà nước và trường hợp quy định tại khoản 1 Điều 149, khoản 2 Điều 151 và khoản 4 Điều 177 của Bộ luật này."
            }
        }
    },
    {
        "question": "Nội dung về đào tạo lao động có bắt buộc phải ghi vào hợp đồng lao động?",
        "message": "Extract relevant articles success!!",
        "file_path": "45_2019_QH14_333670.txt",
        "queries": {
            "dieu_21": {
                "title": "Nội dung hợp đồng lao động",
                "text": "1.\nHợp đồng lao động phải có những nội dung chủ yếu sau đây:\na) Tên, địa chỉ của người sử dụng lao động và họ tên, chức danh của người giao kết hợp đồng lao động bên phía người sử dụng lao động;\nb) Họ tên, ngày tháng năm sinh, giới tính, nơi cư trú, số thẻ Căn cước công dân, Chứng minh nhân dân hoặc hộ chiếu của người giao kết hợp đồng lao động bên phía người lao động;\nc) Công việc và địa điểm làm việc;\nd) Thời hạn của hợp đồng lao động;\nđ) Mức lương theo công việc hoặc chức danh, hình thức trả lương, thời hạn trả lương, phụ cấp lương và các khoản bổ sung khác;\ne) Chế độ nâng bậc, nâng lương;\ng) Thời giờ làm việc, thời giờ nghỉ ngơi;\nh) Trang bị bảo hộ lao động cho người lao động;\ni) Bảo hiểm xã hội, bảo hiểm y tế và bảo hiểm thất nghiệp;\nk) Đào tạo, bồi dưỡng, nâng cao trình độ, kỹ năng nghề.\n2.\nKhi người lao động làm việc có liên quan trực tiếp đến bí mật kinh doanh, bí mật công nghệ theo quy định của pháp luật thì người sử dụng lao động có quyền thỏa thuận bằng văn bản với người lao động về nội dung, thời hạn bảo vệ bí mật kinh doanh, bảo vệ bí mật công nghệ, quyền lợi và việc bồi thường trong trường hợp vi phạm.\n3.\nĐối với người lao động làm việc trong lĩnh vực nông nghiệp, lâm nghiệp, ngư nghiệp, diêm nghiệp thì tùy theo loại công việc mà hai bên có thể giảm một số nội dung chủ yếu của hợp đồng lao động và thỏa thuận bổ sung nội dung về phương thức giải quyết trong trường hợp thực hiện hợp đồng chịu ảnh hưởng của thiên tai, hỏa hoạn, thời tiết.\n4.\nChính phủ quy định nội dung của hợp đồng lao động đối với người lao động được thuê làm giám đốc trong doanh nghiệp có vốn nhà nước.\n5.\nBộ trưởng Bộ Lao động - Thương binh và Xã hội quy định chi tiết các khoản 1, 2 và 3 Điều này."
            },
            "dieu_61": {
                "title": "Học nghề, tập nghề để làm việc cho người sử dụng lao động",
                "text": "1.\nHọc nghề để làm việc cho người sử dụng lao động là việc người sử dụng lao động tuyển người vào để đào tạo nghề nghiệp tại nơi làm việc.\nThời gian học nghề theo chương trình đào tạo của từng trình độ theo quy định của Luật Giáo dục nghề nghiệp.\n2.\nTập nghề để làm việc cho người sử dụng lao động là việc người sử dụng lao động tuyển người vào để hướng dẫn thực hành công việc, tập làm nghề theo vị trí việc làm tại nơi làm việc.\nThời hạn tập nghề không quá 03 tháng.\n3.\nNgười sử dụng lao động tuyển người vào học nghề, tập nghề để làm việc cho mình thì không phải đăng ký hoạt động giáo dục nghề nghiệp; không được thu học phí; phải ký hợp đồng đào tạo theo quy định của Luật Giáo dục nghề nghiệp.\n4.\nNgười học nghề, người tập nghề phải đủ 14 tuổi trở lên và phải có đủ sức khỏe phù hợp với yêu cầu học nghề, tập nghề.\nNgười học nghề, người tập nghề thuộc danh mục nghề, công việc nặng nhọc, độc hại, nguy hiểm hoặc đặc biệt nặng nhọc, độc hại, nguy hiểm do Bộ trưởng Bộ Lao động - Thương binh và Xã hội ban hành phải từ đủ 18 tuổi trở lên, trừ lĩnh vực nghệ thuật, thể dục, thể thao.\n5.\nTrong thời gian học nghề, tập nghề, nếu người học nghề, người tập nghề trực tiếp hoặc tham gia lao động thì được người sử dụng lao động trả lương theo mức do hai bên thỏa thuận.\n6.\nHết thời hạn học nghề, tập nghề, hai bên phải ký kết hợp đồng lao động khi đủ các điều kiện theo quy định của Bộ luật này."
            }
        }
    }
]

def check_similar_articles():
    for case in test_cases:
        question = case["question"]
        data_path = f"/Users/cerris/PycharmProjects/TVPL-test/dataset/{case.get('file_path').replace('.txt', '.json')}"
        articles = read_json_file(file_path=data_path)
        relevant_articles = get_relevant_articles(question, articles)
        keys = [key for key, article in relevant_articles.items()]
        real_keys =  [key for key, article in case["queries"].items()]

        print(keys, real_keys)

    return None


if __name__ == "__main__":
    check_similar_articles()
