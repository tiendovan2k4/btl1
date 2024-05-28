# Import các thư viện cần thiết
import random  # Để ngẫu nhiên hóa dữ liệu huấn luyện
import json  # Để phân tích cú pháp tệp JSON chứa các intent
import pickle  # Để lưu và tải các đối tượng Python
import numpy as np  # Để thực hiện các phép toán số học
import tensorflow as tf  # Để xây dựng và huấn luyện mạng nơ-ron

import nltk  # Để xử lý ngôn ngữ tự nhiên
from nltk.stem import WordNetLemmatizer  # Để lemmatization từ ngữ
nltk.download('wordnet')  # Tải dữ liệu WordNet
nltk.download('punkt')  # Tải dữ liệu bộ phân tích từ Punkt

lemmatizer = WordNetLemmatizer()  # Khởi tạo lemmatizer

# Tải tệp intents
intents = json.loads(open('intents.json').read())

words = []  # Danh sách để chứa tất cả các từ duy nhất
classes = []  # Danh sách để chứa tất cả các lớp/tags duy nhất
documents = []  # Danh sách để chứa các cặp mẫu và các tags tương ứng
ignoreLetters = ['?', '!', '.', ',']  # Các ký tự cần bỏ qua khi tokenization

# Xử lý từng intent
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)  # Tokenization từng mẫu
        words.extend(wordList)  # Thêm từ vào danh sách từ
        documents.append((wordList, intent['tag']))  # Thêm mẫu và tag vào documents
        if intent['tag'] not in classes:
            classes.append(intent['tag'])  # Thêm tag vào classes nếu chưa có

# Lemmatize và sắp xếp các từ, loại bỏ trùng lặp
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

# Sắp xếp các classes
classes = sorted(set(classes))

# Lưu từ và classes vào đĩa bằng pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []  # Danh sách để chứa dữ liệu huấn luyện
outputEmpty = [0] * len(classes)  # Khởi tạo một vector đầu ra trống

# Tạo dữ liệu huấn luyện
for document in documents:
    bag = []
    wordPatterns = document[0]  # Lấy mẫu
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]  # Lemmatize từng từ trong mẫu
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)  # Tạo một bag of words

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1  # Đánh dấu index của lớp là 1
    training.append(bag + outputRow)  # Thêm bag of words và output row vào dữ liệu huấn luyện

random.shuffle(training)  # Ngẫu nhiên hóa dữ liệu huấn luyện
training = np.array(training)  # Chuyển đổi thành mảng numpy

# Tách dữ liệu huấn luyện thành input (trainX) và output (trainY)
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Xây dựng mô hình mạng nơ-ron
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))  # Lớp đầu vào
model.add(tf.keras.layers.Dropout(0.5))  # Lớp dropout để ngăn chặn overfitting
model.add(tf.keras.layers.Dense(64, activation='relu'))  # Lớp ẩn
model.add(tf.keras.layers.Dropout(0.5))  # Một lớp dropout khác
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))  # Lớp đầu ra

# Biên dịch mô hình
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Huấn luyện mô hình
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# Lưu mô hình đã huấn luyện
model.save('chatbot_model.h5', hist)
print('Done')
