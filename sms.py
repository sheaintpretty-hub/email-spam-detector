from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import streamlit as st
import pickle

data = pd.read_csv("spam.csv", encoding="latin-1")

data = data[['v1', 'v2']]
data.columns = ['label', 'message'] # rename the columns

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)
X_train = vectorizer.fit_transform(X_train) 
X_test = vectorizer.transform(X_test) 

model = MultinomialNB()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy: ", accuracy)

message = input("Enter a message: ")
message_vector = vectorizer.transform([message])
prediction = model.predict(message_vector)

if prediction[0] == "spam" :
    print("This message is SPAM!!!")
else:
    print("This message is NOT SPAM!!!")

print(model.classes_)
print(model.predict_proba(message_vector))

with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Model saved successfully!")