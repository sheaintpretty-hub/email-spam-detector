from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train) # fit is to learn vocab
X_test = vectorizer.transform(X_test) # transform is to convert the messages into numbers

model = MultinomialNB()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy: ", accuracy)

print(prediction)
# print("Training messages:", len(X_train))
# print("Testing messages:", len(X_test))

# print("Training labels:", len(y_train))
# print("Testing labels:", len(y_test))
# print(data["label"].value_counts()) # gives the count of data
# print(data.head()) # will print only the FIRST 5 entries of the data
# print(data.columns) # will print the NAMES of the columns 
# print(data) # will print ALL the data
# head() is a function hence it is written with () while column is an attribute therefore it is not written with ()

# data.info() # prints information about the dataframe

# Message
#    │
#    ▼
# TF-IDF Vectorizer
#    │
#    ▼
# Numbers
#    │
#    ▼
# MultinomialNB
#    │
#    ▼
# Spam / Ham