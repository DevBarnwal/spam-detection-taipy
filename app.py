import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from taipy.gui import Gui

# -------------------------------
# 1. Dataset (Example Training Data)
# -------------------------------

data = {
    "message": [
        "Win a free iPhone now",
        "Congratulations you won lottery",
        "Claim your free prize today",
        "Meeting at 10 tomorrow",
        "Please send the report",
        "Let's have lunch today",
        "Free money offer just for you",
        "Call me when you reach home"
    ],
    "label": [
        "spam",
        "spam",
        "spam",
        "ham",
        "ham",
        "ham",
        "spam",
        "ham"
    ]
}

df = pd.DataFrame(data)

# -------------------------------
# 2. Convert text to numbers
# -------------------------------

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# -------------------------------
# 3. Train Model
# -------------------------------

model = MultinomialNB()
model.fit(X, y)

# -------------------------------
# 4. Taipy Variables
# -------------------------------

input_message = ""
prediction_result = "Enter a message and click Detect"

# -------------------------------
# 5. Prediction Function
# -------------------------------

def check_spam(state):
    text = [state.input_message]
    text_vector = vectorizer.transform(text)
    prediction = model.predict(text_vector)

    if prediction[0] == "spam":
        state.prediction_result = "🚨 Spam Message"
    else:
        state.prediction_result = "✅ Not Spam"

# -------------------------------
# 6. UI
# -------------------------------

page = """
# 📧 Spam Message Detector

Enter Message:

<|{input_message}|input|multiline=True|>

<br/>

<|Check Spam|button|on_action=check_spam|>

<br/><br/>

## Result:
<|{prediction_result}|text|>
"""

# -------------------------------
# 7. Run App
# -------------------------------

Gui(page).run(
    title="Spam Detection App",
    host="0.0.0.0",
    port=8080,
    dark_mode=True
)