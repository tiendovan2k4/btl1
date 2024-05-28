import spacy

# Tải mô hình ngôn ngữ tiếng Anh
nlp = spacy.load("en_core_web_sm")

# Hàm chatbot để nhận diện thực thể
def chatbot_response(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Ví dụ sử dụng chatbot
user_input = "I will travel to Paris next week and meet John there."
entities = chatbot_response(user_input)
print("Recognized Entities:", entities)
