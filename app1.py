import streamlit as st
import requests
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ------------------ Load ML Model ------------------
model = joblib.load("fake_news_model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")


# ------------------ GNews API ------------------
API_KEY = "9958fc2d15cd900e8bb4dc32ef526b24"   # 🔑 Replace with your GNews key
BASE_URL = "https://gnews.io/api/v4/top-headlines"

# ------------------ Helper Function ------------------
def fetch_news(topic="world", lang="en", country="in", page=1):
    params = {
        "topic": topic,       # valid topics: breaking-news, world, nation, business, technology, entertainment, sports, science, health
        "lang": lang,         # language (en, hi, etc.)
        "country": country,   # country code (in, us, gb, etc.)
        "max": 10,   
        "page":page,         # free plan: max 10
        "apikey": API_KEY
    }
    response = requests.get(BASE_URL, params=params).json()

    if "articles" not in response:
        st.error(f"⚠️ Error fetching news: {response.get('message', 'Unknown error')}")
        return []

    return [a for a in response["articles"] if a.get("title")]

# Load BERT model once at startup
MODEL_NAME = "mrm8488/bert-tiny-finetuned-fake-news-detection"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def classify_news_bert(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    # Check how the model labels — “0” or “1” might correspond differently
    # In this model, likely label 0 = “Fake”, 1 = “Real” or vice versa. You’ll need to test.
    return "Fake ❌" if pred == 0 else "Real ✅"

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Fake News Detector", layout="wide", initial_sidebar_state="expanded")

# Custom background
# Custom dark blue background
dark_blue_bg = """
<style>
/* Main app background */
[data-testid="stAppViewContainer"] {
    background-color: #001f3f;   /* Dark blue */
    color: white;                /* White text for contrast */
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #002b5c;   /* Slightly lighter dark blue */
    color: white;
}

/* Header */
[data-testid="stHeader"] {
    background-color: #001737;   /* Deep navy */
    color: white;
}
</style>
"""

st.markdown(dark_blue_bg, unsafe_allow_html=True)

custom_css = """
<style>
/* Make pagination info text (st.info) white */
div.stAlert {
    color: white !important;
    background-color: #003366 !important;  /* Optional: dark blue box */
}

/* Make fake news text white inside st.error */
div.stAlert div[role="alert"] {
    color: white !important;
}

/* Optional: Change error box background to red + white text */
div.stAlert[data-baseweb="toast"] {
    background-color: #b71c1c !important; /* dark red */
    color: white !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

button_css = """
<style>
/* Style all buttons */
div.stButton > button {
    background-color: #002b5c;   /* Dark blue background */
    color: white !important;     /* White text */
    border-radius: 8px;
    border: 1px solid #ffffff;
    font-weight: bold;
}

/* Hover effect */
div.stButton > button:hover {
    background-color: #004080;   /* Lighter blue on hover */
    color: #ffcc00 !important;   /* Yellowish text on hover */
}
</style>
"""
st.markdown(button_css, unsafe_allow_html=True)


st.title("📰 Fake News Detection and Aggregator Dashboard")

# Sidebar for fake news
st.sidebar.title("🚨 Fake News Detected Today")

st.sidebar.markdown("""
## Fake News Detection with Automatic Aggregation
This dashboard aggregates **real-time news** from multiple categories (World, Business, Sports, Technology, etc.) 
and uses a **BERT-based AI model** to classify whether each headline is **Real ✅** or **Fake ❌**.  

🔹 Features:  
- Automatic news fetching from GNews API  
- Real-time fake news detection without manual input  
- Side panel that highlights trending fake news with images and sources  
- Pagination support to explore more news pages  
- Direct "Read More 🔗" links to full articles  
- Dark blue theme with improved readability  
""")

fake_news_list = []

# GNews topics
categories = [
    "breaking-news",
    "world",
    "nation",
    "business",
    "technology",
    "entertainment",
    "sports",
    "science",
    "health"
]

selected_category = st.selectbox("Choose News Category", categories)

# ------------------ Pagination Control ------------------
if "page" not in st.session_state:
    st.session_state.page = 1

col_prev, col_next = st.columns([1, 1])
with col_prev:
    if st.button("⬅️ Previous Page ") and st.session_state.page > 1:
        st.session_state.page -= 1
with col_next:
    if st.button("Next Page ➡️"):
        st.session_state.page += 1

st.info(f"📄 Showing Page {st.session_state.page}")

# Fetch news for category
articles = fetch_news(topic=selected_category,page=st.session_state.page)

if not articles:
    st.warning("⚠️ No news articles found. Try another category or check your API key.")

# Layout: two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Latest {selected_category.capitalize()} News (Page {st.session_state.page})")

    for article in articles:  # show first 10
        title = article.get("title")
        image=article.get('image')
        source = article.get("source", {}).get("name", "Unknown")
        url=article.get("url")

        if title:
            label = classify_news_bert(title)
            if label == "Fake ❌":
                fake_news_list.append(article)
            if image:
                st.image(image,width=400)
            st.write(f"**{title}**")
            st.caption(f"Source: {source}")
            st.markdown(f"Prediction: **{label}**")
            if url:
                st.markdown(f"Click below for more information \n[Read More🔗]({url})",unsafe_allow_html=True)
            st.markdown("---")

with col2:
    st.subheader("🚨 Fake News Today")
    if fake_news_list:
        for article in fake_news_list:  # ✅ now article is a dict
            image = article.get("image")
            title = article.get("title")
            source = article.get("source", {}).get("name", "Unknown")
            url=article.get("url")

            if image:
                st.image(image, width=200)
            st.error(f"**{title}**\n\nSource: {source}")
            if url:
                st.markdown(f"Click below for more information \n[Read More 🔗]({url})", unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.success("No fake news detected yet ✅")

        //python -m streamlit run app.py

