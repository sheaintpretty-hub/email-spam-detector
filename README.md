# 💌 Email Spam Detector

An AI-powered web application that classifies emails and SMS messages as **Spam** or **Not Spam** using Machine Learning.

## ✨ Features

- 🚨 Detects spam emails and SMS
- ✅ Predicts whether a message is safe
- 📊 Displays confidence score
- 📜 Maintains scan history
- 🧹 Clear history option
- 📩 Built-in spam and safe examples
- 🎨 Custom pink Streamlit interface

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- TF-IDF Vectorizer
- Multinomial Naive Bayes

---

## 📂 Project Structure

```
email-spam-detector/
│── app.py
│── main.py
│── sms.py
│── spam.csv
│── spam_model.pkl
│── vectorizer.pkl
│── requirements.txt
```

---

## 🚀 Installation & Usage

Clone the repository:

```bash
git clone https://github.com/sheaintpretty-hub/email-spam-detector.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📊 Machine Learning Model

- **Algorithm:** Multinomial Naive Bayes
- **Feature Extraction:** TF-IDF Vectorizer
- **Dataset:** Spam/Ham email and SMS dataset

---

## 👩‍💻 Author

**Mridula**  
B.Tech CSE Student  
NIT Delhi
