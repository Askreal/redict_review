import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer



loaded_model = joblib.load('review_model.joblib')
loaded_cv = joblib.load('count_vectorizer_for_review.joblib')

def predict_good_bad(input_text):

    lemmatizer = WordNetLemmatizer()
    processed_input = re.sub('[^a-zA-Z]', ' ', input_text.lower())
    words = processed_input.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    processed_input = ' '.join(words)

    input_cv = loaded_cv.transform([processed_input])

    prediction = loaded_model.predict(input_cv)

    return "Good from GOOD_BAD.py" if prediction[0] == "good" else "Bad from GOOD_BAD.py"


input_txt = input("Enter your review : ")
result = predict_good_bad(input_txt)
print(f"It's a : {result} review.")

