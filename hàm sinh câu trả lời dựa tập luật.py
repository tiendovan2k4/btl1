import re

# Định nghĩa các luật trả lời
rules = {
    r'.*weather.*': "The weather is sunny today.",
    r'.*your name.*': "My name is ChatBot.",
    r'.*help.*': "How can I assist you?",
    r'.*capital of (.+)\?': lambda match: f"The capital of {match.group(1)} is not available in my database.",
    r'.*bye.*': "Goodbye! Have a nice day!",
    r'.*how are you.*': "I'm just a bunch of code, but I'm doing fine!",
    r'.*what is your purpose.*': "I am here to help you with your questions."
}

# Hàm sinh câu trả lời dựa trên tập luật
def chatbot_response(user_input):
    for pattern, response in rules.items():
        match = re.match(pattern, user_input, re.IGNORECASE)
        if match:
            if callable(response):
                return response(match)
            return response
    return "I'm sorry, I don't understand the question."

# Ví dụ sử dụng chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() in ['bye', 'exit', 'quit']:
        print("ChatBot: Goodbye! Have a nice day!")
        break
    response = chatbot_response(user_input)
    print(f"ChatBot: {response}")
