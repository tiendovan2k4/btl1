# Import các thư viện cần thiết
import random  # Để chọn ngẫu nhiên các phản hồi
import json  # Để đọc và phân tích cú pháp tệp JSON chứa các intents
import pickle  # Để tải các đối tượng Python đã lưu
import numpy as np  # Để thực hiện các phép toán số học
import nltk  # Để xử lý ngôn ngữ tự nhiên

# Từ thư viện NLTK, import WordNetLemmatizer để lemmatization từ ngữ
from nltk.stem import WordNetLemmatizer
# Từ thư viện Keras, import hàm load_model để tải mô hình đã huấn luyện
from keras.models import load_model

# Khởi tạo lemmatizer từ NLTK
lemmatizer = WordNetLemmatizer()

# Tải tệp intents chứa các mẫu và các tags
intents = json.loads(open('intents.json').read())

# Tải danh sách từ và classes đã lưu từ tệp pickle
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Tải mô hình đã huấn luyện từ tệp
model = load_model('chatbot_model.h5')

# Hàm để làm sạch câu đầu vào bằng cách tokenization và lemmatization các từ
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Tokenize câu thành danh sách các từ
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]  # Lemmatize từng từ
    return sentence_words

# Hàm để tạo bag of words từ câu đầu vào
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)  # Làm sạch câu
    bag = [0] * len(words)  # Khởi tạo bag of words với tất cả các phần tử bằng 0
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:  # Nếu từ hiện tại trong câu có trong danh sách từ
                bag[i] = 1  # Đặt phần tử tương ứng trong bag bằng 1
    return np.array(bag)  # Trả về bag of words dưới dạng mảng numpy

# Hàm để dự đoán lớp (intent) của câu đầu vào
def predict_class(sentence):
    bow = bag_of_words(sentence)  # Tạo bag of words từ câu đầu vào
    res = model.predict(np.array([bow]))[0]  # Dự đoán lớp với mô hình đã huấn luyện
    ERROR_THRESHOLD = 0.25  # Ngưỡng lỗi để lọc các dự đoán không chắc chắn
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Lọc các dự đoán có xác suất lớn hơn ngưỡng

    results.sort(key=lambda x: x[1], reverse=True)  # Sắp xếp các dự đoán theo xác suất giảm dần
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})  # Thêm kết quả dự đoán vào danh sách trả về
    return return_list

# Hàm để lấy phản hồi từ chatbot dựa trên danh sách các dự đoán intents
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']  # Lấy intent dự đoán có xác suất cao nhất
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:  # Tìm intent tương ứng trong tệp intents
            result = random.choice(i['responses'])  # Chọn ngẫu nhiên một phản hồi
            break
    return result

print("GO! Bot is running!")  # Thông báo rằng chatbot đã sẵn sàng

# Vòng lặp chính để nhận và xử lý các tin nhắn đầu vào
while True:
    message = input("")  # Nhận tin nhắn đầu vào từ người dùng
    ints = predict_class(message)  # Dự đoán lớp của tin nhắn
    res = get_response(ints, intents)  # Lấy phản hồi từ chatbot
    print(res)  # In phản hồi ra màn hình
