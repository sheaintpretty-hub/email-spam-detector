import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="💌",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>

.stApp{
    background: linear-gradient(to bottom, #fff0f6, #ffe4ec);
}

h1{
    color:#d63384 !important;
    text-align:center;
    font-weight:800;
}

textarea{
    border-radius:15px !important;
    background:white !important;
    color:#333 !important;
}

div.stButton > button{
    background-color:#ff69b4;
    color:white;
    border:none;
    border-radius:15px;
    font-size:20px;
    font-weight:bold;
    padding:12px 25px;
}

div.stButton > button:hover{
    background-color:#ff1493;
    color:white;
    transform:scale(1.04);
    transition:0.25s;
}

label {
    color: #6b2147 !important;
    font-weight: 600 !important;
}

</style>
""",
    unsafe_allow_html=True,
)

with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

st.sidebar.title("About")
st.sidebar.write("""
Model:
Naive Bayes

Vectorizer:
TF-IDF

Accuracy:
96.68%
""")

st.title("💌 Email Spam Detector")

st.markdown(
    "<h4 style='text-align:center; color:#6b2147;'>Paste an email or SMS below and click <b>Scan Message</b> </h4>",
    unsafe_allow_html=True,
)

if "message" not in st.session_state:
    st.session_state.message = ""

left_space, col1, col2, right_space = st.columns([0.4, 1, 1, 0.4])

with col1:
    if st.button("Try Spam Example"):
        st.session_state.message = (
            "Congratulations! You have won ₹50,000. "
            "Click the link below immediately to claim your prize!"
        )

with col2:
    if st.button("Try Safe Example"):
        st.session_state.message = (
            "Hey! Are we still meeting at 5 PM today? Let me know."
        )

message = st.text_area(
    "Message",
    value=st.session_state.message,
    placeholder="Type or paste your email or SMS here...",
    height=220
)

st.markdown(
    f"""
    <p style="
        color:#6b2147;
        font-size:14px;
        margin-top:-10px;
        margin-bottom:10px;
    ">
    Characters: {len(message)} | Words: {len(message.split())}
    </p>
    """,
    unsafe_allow_html=True,
)
    
predict_button = st.button("Scan Message")
if predict_button:
    if not message.strip():
        st.warning("⚠️ Please enter an email or SMS to scan.")
        st.stop()


    message_vector = vectorizer.transform([message])

    prediction = model.predict(message_vector)
    probability = model.predict_proba(message_vector)
    confidence = probability.max() * 100

    if prediction[0] == "spam":

        st.markdown("""
        <div style="
            background:#ffd6dc;
            color:#9b1c31;
            padding:16px 20px;
            border-radius:15px;
            font-size:18px;
            font-weight:bold;
            border-left:8px solid #e63946;
            margin-bottom:15px;
        ">
            🚨 This message is SPAM!
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="
            background:#d4edda;
            color:#155724;
            padding:16px 20px;
            border-radius:15px;
            font-size:18px;
            font-weight:bold;
            border-left:8px solid #28a745;
            margin-bottom:15px;
        ">
            ✅ This message is NOT SPAM!
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="
        background:#dbeafe;
        color:#1d4ed8;
        padding:14px 18px;
        border-radius:15px;
        font-size:18px;
        font-weight:600;
        border-left:8px solid #3b82f6;
    ">
    Confidence: {confidence:.2f}%
    </div>
    """, unsafe_allow_html=True)

    st.session_state.history.append({
        "Message": message[:40] + ("..." if len(message) > 40 else ""),
        "Prediction": "🚨 Spam" if prediction[0] == "spam" else "✅ Safe",
        "Confidence": f"{confidence:.2f}%"
    })

if st.session_state.history:
    st.markdown(
        """
        <h2 style="
            color:#d63384;
            font-weight:800;
            margin-top:35px;
            margin-bottom:15px;
    ">
    Scan History
    </h2>
    """,
    unsafe_allow_html=True,
)

    history_df = pd.DataFrame(st.session_state.history)
    if st.button("🗑 Clear History"):
        st.session_state.history = []
        st.rerun()
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
)

st.markdown(
    "<h5 style='text-align:center; color:#6b2147;'>Made by mridula</h5>",
    unsafe_allow_html=True,
)
