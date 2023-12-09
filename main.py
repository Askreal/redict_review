import nltk
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('review_th.csv', encoding='utf-8')

# Drop specified columns
df.drop(['Unnamed: 0', 'ratingvalue', 'Unnamed: 2'], axis=1, inplace=True)

df.columns = ['label', 'text']

# df preprocessing
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text.lower())
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# create a new column in df
df['processed_text'] = df['text'].apply(preprocess_text)

df.dropna(subset=['processed_text', 'label'], inplace=True)

x, y = df['processed_text'], df['label']  # Fix typo 'lable' to 'label'

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)  # Removed 'train_size'

cv = CountVectorizer()
x_train_cv = cv.fit_transform(x_train)

# train model
lr = LogisticRegression()
lr.fit(x_train_cv, y_train)

# save
joblib.dump(lr, 'review_model.joblib')
joblib.dump(cv, 'count_vectorizer_for_review.joblib')

def predict_good_bad(input_text):
    processed_input = preprocess_text(input_text)

    loaded_cv = joblib.load('count_vectorizer.joblib')
    input_cv = loaded_cv.transform([processed_input])

    loaded_model = joblib.load('spam_model.joblib')

    prediction = loaded_model.predict(input_cv)

    print(prediction[0])

    return "Good from main.py" if prediction[0] == "good" else "Bad From main.py"  # Adjusted prediction comparison

input_txt = input("Enter your review: ")
result = predict_good_bad(input_txt)
print(f"Your restaurant is: {result}")
