import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Download necessary NLTK data
nltk.download('punkt')

# Dữ liệu mẫu
questions = [
    "What is the weather like today?",
    "How do I reset my password?",
    "What is the capital of France?",
    "How to bake a chocolate cake?",
    "Where can I find the nearest gas station?",
    "What is 2+2?",
    "Tell me a joke.",
    "How to learn Python?"
]

labels = [
    "weather",
    "technical support",
    "general knowledge",
    "cooking",
    "location",
    "math",
    "entertainment",
    "education"
]

# Tạo pipeline cho mô hình
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Huấn luyện mô hình
model.fit(questions, labels)

# Hàm chatbot để phân loại câu hỏi
def chatbot_response(question):
    label = model.predict([question])[0]
    return f"This question is related to: {label}"

# Ví dụ sử dụng chatbot
user_question = "How can I change my password?"
response = chatbot_response(user_question)
print(response)
