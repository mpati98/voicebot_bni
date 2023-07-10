from flask import Flask, jsonify, render_template, request
from flask_restful import Resource, Api
import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
# Load data
from keras.models import load_model
import json
model = load_model('data/model/model_July10.h5')
intents = json.loads(open('data/intents/intents_July_10_2023.json').read())
words = pickle.load(open('data/model/textsJuly10.pkl','rb'))
classes = pickle.load(open('data/model/labelsJuly10.pkl','rb'))

def transText(text_input, scr_input='user'):
    from googletrans import Translator
    # define a translate object
    translate = Translator()
    if scr_input == "bot":
        result = translate.translate(text_input, src='en', dest='vi')
        result = result.text
    elif scr_input == "user":
        result = translate.translate(text_input, src='vi', dest='en')
        result = result.text
    else:
        result = "We not support this language, please use English or Vietnamese!"
    return result

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    print(res)
    # AMBIGOUS_THRESHOLD = 0.0
    CERTAIN_THRESHOLD = 0.7
    results = [[i,r] for i,r in enumerate(res) if r>CERTAIN_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = i['responses']
            break
    return result, tag

def chatbot_response(msg):
    if msg == 'không nghe rõ':
        res = ["Tín hiệu không ổn định, vui lòng lặp lại rõ hơn nhé", "fbad6e35-3933-4388-be7b-d6dda276e114"]
        tag = "Error"
    else:
        ints = predict_class(msg, model)
        if ints:
            res, tag = getResponse(ints, intents)
        else:
            res = ["Rất xin lỗi vì thông tin bạn cần không tồn tại trong hệ thống, chúng tôi sẽ kiểm tra và cập nhật trong thời gian tới. Bạn còn muốn biết thêm thông tin gì khác không?", "930e5fa5-827a-454f-bcac-84e1b9dd5b4f"]
            tag = "Other"
    return res, tag

def chat_rulebased_01(msg):
    if "Nguyễn Thị Bích Hằng" in msg.lower():
        res = ["GIÁM ĐỐC VÙNG BNI HCM CENTRAL 6 là bà ANNA Nguyễn Thị Bích Hằng. Nguyên Tổng Giám Đốc BNI Việt Nam. Nguyên Tổng Giám Đốc ActionCOACH Việt Nam. Tổng Giám đốc ActionCOACH CBD Firm. Đã điều hành và giúp ActionCOACH CBD Firm đạt giải Top 2 Thế giới (Xếp hạng 2/83 quốc gia). Là người phụ nữ đầu tiên đưa nhà huấn luyện Việt Nam và Doanh Nghiệp Việt Nam được vinh doanh trên Toàn Cầu. Với mục tiêu xây dựng vùng BNI HCM Central 6 trở thành vùng mang danh Diamond và mang lại nhiều giá trị hơn nữa dành riêng cho cộng đồng doanh nghiệp trong bối cảnh nền kinh tế Việt Nam đang bước vào một quỹ đạo phát triển mới, bà Anna Hằng Nguyễn hy vọng được đồng hành cùng các doanh nghiệp phát triển bền vững, góp phần vào sự phát triển chung & khẳng định vị thế Việt Nam trên nền kinh tế toàn cầu.", "708c9126-0532-44a2-8c74-3fd73f3cf949"]
        tag = "BNI_Director"
    elif "lợi ích" in msg.lower():
        res = ["Những lợi ích tuyệt vời dành cho chủ doanh nghiệp khi tham gia BNI Hồ Chí Minh Central 6: 1. KẾT NỐI KINH DOANH, giúp chủ doanh nghiệp quảng bá thương hiệu, trao đổi giao thương và kết nối các cơ hội hợp tác kinh doanh trong cộng đồng CEO BNI đồng thời kiến tạo thương hiệu cá nhân cho Chủ doanh nghiệp - Bí quyết để sở hữu nguồn Marketing vô hạn. 2. ĐỒNG HÀNH, chỉ có tại các Chapter thuộc vùng BNI HCM Central 6. Chủ doanh nghiệp được huấn luyện theo Power Team bởi các nhà Huấn luyện doanh nghiệp xuất sắc nhất đến từ ActionCOACH CBD Firm, giúp phát triển nền móng vững chắc cho doanh nghiệp, nâng cao năng lực lãnh đạo, quản trị, chiến lược và vận hành doanh nghiệp hiệu quả. 3. ĐÓN ĐẦU XU THẾ, ứng dụng thực tế theo nền kinh tế số cùng kỷ nguyên số, cập nhật thêm các kiến thức, kỹ năng và xu hướng mới nhất ở cả Việt Nam và Thế giới.", "765edfe1-b9c5-412c-b3cb-9e2098e07c60"]
        tag = "BNI_Benefits" 
    elif "đối tượng tham gia" in msg.lower():
        res = ["Đối tượng tham gia là chủ doanh nghiệp tại thành phố Hồ Chí Minh (Riêng ngành Bảo hiểm nhân thọ/Phi nhân thọ/Ngân hàng thì Giám đốc chi nhánh hoặc Trưởng phòng có thể tham gia) và chưa là thành viên của bất kì một chapter nào.", "29dd2a45-400c-4e5b-b9ee-bdce76bf9846"]
        tag = "BNI_Participate" 
    elif "địa chỉ" in msg.lower() or "thông tin liên hệ" in msg.lower():
        res = ["Thông tin liên hệ: Địa chỉ: 90-92 Lê Thị Riêng, P. Bến Thành, Q.1, TP.HCM. Email:hcmc6@bni.vn. Hotline: 1800.8087", "7787ae04-eca5-48f5-bacf-94fcf2fb7b88"]
        tag = "BNI_Address" 
    elif "tầm nhìn" in msg.lower():
        res = ["Tầm nhìn của BNI - Thay đổi cách Thế giới làm kinh doanh. Giúp Thành viên nhận thức rằng Networking là gieo trồng không săn bắt và phải vun đắp các mối quan hệ. Gắn liền thương hiệu BNI với marketing truyền miệng. Khi nói về marketing truyền miệng người ta sẽ nghĩ tới BNI. Trở thành tổ chức kết nối kinh doanh tốt nhất, lớn nhất và được thừa nhận rộng khắp trên toàn cầu. Lãnh đạo sáng tạo, đề xuất phương hướng và hỗ trợ cho các Giám đốc, Ban điều hành và Thành viên của từng Chapter. Đào tạo, hướng dẫn và thông tin tốt hơn cho Thành viên về quy trình triển khai kinh doanh áp dụng hình thức marketing truyền miệng. Liên tục tiếp nhận và trả lời phản hồi từ Thành viên. Gia tăng số lượng Thành viên trên toàn Thế giới. Đánh giá thành tựu đạt được bằng những phương pháp đo lường truyền thống và cải tiến.", "9df0a232-cdb4-44e9-b5cd-4b026ef76879"]
        tag = "BNI_Vision"
    elif "sứ mệnh" in msg.lower():
        res = ["Sứ mệnh của BNI là giúp các Thành viên gia tăng cơ hội kinh doanh thông qua chương trình trao đổi cơ hội kinh doanh (referral) một cách có cấu trúc, tích cực và chuyên nghiệp, để giúp các Thành viên phát triển những mối quan hệ lâu dài và có ý nghĩa, với những ngành nghề kinh doanh chất lượng.", "96277e47-d729-45a8-8a4b-9857e1d6c9c5"]
        tag = "BNI_Statement" 
    elif "triết lý" in msg.lower():
        res = ["Triết lý thành công tại BNI nghĩa là trở thành một Thành viên tích cực và hỗ trợ của tổ chức, dựa trên sự tương trợ lẫn nhau. Điều này đòi hỏi sự cam kết chung với các Thành viên khác theo triết lý “Cho là nhận”: Bằng việc cho đi cơ hội kinh doanh, bạn cũng sẽ nhận lại cơ hội kinh doanh cho mình.", "e30b526e-ba2e-49bb-8e18-0fb5ff8a8cf1"]
        tag = "BNI_Philosophy"
    elif "giá trị cốt lõi" in msg.lower():
        res = ["BNI luôn giữ 7 giá trị cốt lõi: 1. Cho là Nhận, Cho Là Nhận là triết lý cơ bản của BNI. Chúng tôi minh chứng cho điều đó bằng việc trao cơ hội kinh doanh cho người khác, đổi lại bạn cũng sẽ nhận được cơ hội kinh doanh. 2. Học tập suốt đời, Chúng tôi tin vào sự hoàn thiện liên tục những kỹ năng phát triển bản thân và kỹ năng chuyên môn. BNI cung cấp các cơ hội đa dạng để hỗ trợ học tập suốt đời. 3. Truyền thống và đổi mới, Truyền thống trong một tổ chức cho chúng ta biết nguồn gốc chúng ta phát triển từ đâu và đặt nền móng. Chúng ta là ai, tuy nhiên chúng ta vẫn luôn phải tìm cách đổi mới sáng tạo. 4. Thái độ tích cực, BNI cung cấp một môi trường cho phép bạn bao quanh mình những người luôn muốn giúp bạn thành công. 5. Xây dựng các mối quan hệ, Kết nối kinh doanh là nuôi trồng các kết nối với những con người mới thay vì đi săn các mối quan hệ đó. Mọi người luôn muốn làm kinh doanh với những người mà họ biết và tin tưởng. Đó chính là việc nuôi dưỡng các mối quan hệ đó. 6. Tinh thần trách nhiệm. Nếu bạn muốn có được một hệ thống kết nối cá nhân có quyền năng, bạn phải có tính trách nhiệm. Nếu không, hệ thống đó trở thành một nhóm hoạt động xã hội. 7. Sự công nhận, Sự ghi nhận đóng góp của mọi người là rất quan trọng.", "3abae59a-5809-4265-85cd-7fc56c1cab4d"]
        tag = "BNI_Core" 
    else:
        res = ["BNI - BUSINESS NETWORK INTERNATIONAL là tổ chức kết nối thương mại lớn nhất và thành công nhất trên thế giới hiện nay, được Tiến sĩ Ivan Misner sáng lập vào năm 1985. Cho đến nay BNI Toàn cầu đã trao cho nhau 12.2 triệu cơ hội kinh doanh (referrals) với tổng trị giá 16.9 tỷ USD. BNI VIỆT NAM đã phôi thai ở Việt Nam từ đầu năm 2007 nhưng do một số khó khăn do sự khác biệt về văn hóa, hội nhập nên đến tháng 8 năm 2010 BNI mới chính thức thành lập tại Việt Nam. Hiện tại, BNI Việt Nam đã có 8,266 thành viên với 213 Chapter tại các vùng, tỉnh thành lớn trên cả nước. Trong đó, vùng BNI HCM CENTRAL 6 nằm trong Top 10 vùng phát triển tốt nhất trong hệ thống BNI toàn cầu do bà Anna Nguyễn Thị Bích Hằng điều hành với 07 Chapter cùng 446 thành viên Chủ doanh nghiệp.", "29bef721-ab0d-4265-b14a-8093916bab56"]
        tag = "BNI_Intro" 
    return res, tag


app = Flask(__name__)
api = Api(app)



@app.route("/")
def home():
    return render_template("index.html")

@app.route('/welcome', methods=["POST"])
def voice_welcome():
    resp = "Xin chào, em là trợ lý ảo của BNI. Quý khách cần biết thông tin gì không ạ?"
    output = {
            "res_text": resp,
            "res_audio": "BNI_welcome"
        }
    return jsonify(output)


class Chatbot(Resource):

    def post(self):
        text_input = request.get_json().get("message")
        try:
            if "bni" in text_input.lower():
                resp, tag = chat_rulebased_01(text_input)
            elif "không nghe rõ" == text_input:
                resp = ["Tín hiệu không ổn định, vui lòng lặp lại rõ hơn nhé", "fbad6e35-3933-4388-be7b-d6dda276e114"]
                tag = "Error"
            else:
                text_input = transText(text_input)
                try:
                    resp, tag = chatbot_response(text_input)
                except:
                    resp = ["Tín hiệu không ổn định, vui lòng lặp lại rõ hơn nhé", "fbad6e35-3933-4388-be7b-d6dda276e114"]
                    tag = "Error"
                print(resp)
            output = {
                "res_text": resp[0],
                "audio_token": resp[1],
                "res_audio": tag
            }
        except:
            resp = ["Tín hiệu không ổn định, vui lòng lặp lại rõ hơn nhé", "fbad6e35-3933-4388-be7b-d6dda276e114"]
            tag = "Error"
            output = {
                "res_text": resp[0],
                "audio_token": resp[1],
                "res_audio": tag
            }
        print(output)
        return jsonify(output)

api.add_resource(Chatbot, '/response')

if __name__ == "__main__":
    app.run(debug=True)