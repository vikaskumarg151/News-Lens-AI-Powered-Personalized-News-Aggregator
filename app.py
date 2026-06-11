# app1.py
import streamlit as st
import requests
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse
from collections import Counter
import re
from datetime import datetime, timedelta
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

# ------------------ FALLBACK IMAGE ------------------
IMAGE_FALLBACK = "/mnt/data/c040d3c5-935d-497c-b7f6-bb6c01070cef.png"

# ------------------ SESSION STATE MANAGEMENT ------------------
if "interests" not in st.session_state:
    st.session_state.interests = []
if "viewed_articles" not in st.session_state:
    st.session_state.viewed_articles = set()
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "last_recommendation_update" not in st.session_state:
    st.session_state.last_recommendation_update = None
if "classification_cache" not in st.session_state:
    st.session_state.classification_cache = {}

# ------------------ GNews API ------------------
API_KEY = "9958fc2d15cd900e8bb4dc32ef526b24"
BASE_URL = "https://gnews.io/api/v4/top-headlines"
SEARCH_URL = "https://gnews.io/api/v4/search"

# ------------------ FETCH NEWS FUNCTION ------------------
def fetch_news(topic="world", lang="en", country="in", page=1):
    params = {
        "topic": topic,
        "lang": lang,
        "country": country,
        "max": 10,
        "page": page,
        "apikey": API_KEY
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=8).json()
    except Exception as e:
        st.error(f"⚠️ Network error fetching news: {e}")
        return []

    if "articles" not in response:
        st.error(f"⚠️ Error fetching news: {response.get('message', 'Unknown error')}")
        return []
    return [a for a in response["articles"] if a.get("title")]

# ------------------ ENHANCED CONTENT EXTRACTION ------------------
def extract_full_article(url):
    """Enhanced content extraction with better error handling"""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Multiple strategies for content extraction
        content_selectors = [
            'article', '.article-content', '.story-content', 
            '.post-content', '.entry-content', '.content'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                text = "\n".join([elem.get_text().strip() for elem in elements])
                if len(text) > 300:
                    return text
        
        # Fallback to paragraphs
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50)
        
        return text if len(text) > 300 else None

    except Exception as e:
        print(f"Content extraction error: {e}")
        return None

def extract_alternative_content(url):
    """Alternative content extraction method"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text if len(text) > 300 else None
        
    except Exception as e:
        print(f"Alternative extraction failed: {e}")
        return None

# ------------------ ENHANCED CLASSIFICATION SYSTEM ------------------
MODEL_NAME = "mrm8488/bert-tiny-finetuned-fake-news-detection"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def preprocess_text(text):
    """Clean and preprocess text for better classification"""
    if not text:
        return ""
    
    # Remove extra whitespace, newlines, tabs
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    
    # Remove multiple consecutive punctuation
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)
    
    return text.strip()

def calculate_optimal_text_length(text, target_tokens=400):
    """Calculate optimal text length for BERT model"""
    if not text:
        return ""
    
    max_chars = target_tokens * 4
    
    if len(text) <= max_chars:
        return text
    
    # Try to truncate at sentence boundary
    sentences = re.split(r'[.!?]+', text)
    truncated = ""
    
    for sentence in sentences:
        if len(truncated + sentence) < max_chars:
            truncated += sentence + "."
        else:
            break
    
    if len(truncated) < max_chars * 0.5:
        truncated = text[:max_chars]
        if len(truncated) == max_chars:
            truncated = truncated.rsplit(' ', 1)[0]
    
    return truncated.strip()

def extract_meaningful_content(url, title, description):
    """Enhanced content extraction with multiple fallback strategies"""
    # Try to extract full article content
    full_text = extract_full_article(url)
    
    if full_text and len(full_text) > 500:
        return full_text
    
    # If full extraction fails, try alternative method
    alternative_content = extract_alternative_content(url)
    if alternative_content and len(alternative_content) > 300:
        return alternative_content
    
    # Final fallback: combine available information
    fallback_content = f"{title}. {description}"
    return fallback_content

def classify_with_confidence(text, model, tokenizer):
    """Perform classification with confidence scoring and error handling"""
    try:
        if not text or len(text.strip()) < 50:
            return "Insufficient Text ⚠", 0.5, "Text too short for reliable classification"
        
        # Preprocess text
        cleaned_text = preprocess_text(text)
        optimal_text = calculate_optimal_text_length(cleaned_text)
        
        if len(optimal_text) < 100:
            return "Insufficient Text ⚠", 0.5, "Text too short after cleaning"
        
        # Tokenize
        inputs = tokenizer(
            optimal_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)[0]
        
        pred = torch.argmax(probs).item()
        confidence = float(probs[pred])
        
        # Apply confidence threshold
        if confidence < 0.6:
            label = "Uncertain ⚠"
        else:
            label = "Fake ❌" if pred == 0 else "Real ✅"
        
        return label, confidence, "Good quality text"
        
    except Exception as e:
        print(f"Classification error: {e}")
        return "Classification Error ⚠", 0.5, f"Model inference failed: {str(e)}"

def enhanced_classify_news_full(article):
    """Enhanced news classification with better text processing"""
    title = article.get("title", "").strip()
    url = article.get("url", "")
    description = article.get("description", "").strip()
    
    # Check cache first
    cache_key = f"{url}_{title}"
    if cache_key in st.session_state.classification_cache:
        return st.session_state.classification_cache[cache_key]
    
    # Validate input
    if not title:
        result = ("No Title ⚠", 0.5, "Article has no title", "No content")
        st.session_state.classification_cache[cache_key] = result
        return result
    
    # Extract and process content
    full_content = extract_meaningful_content(url, title, description)
    
    if not full_content or len(full_content.strip()) < 100:
        fallback_content = f"Title: {title}. Description: {description}"
        if len(fallback_content) < 50:
            result = ("Insufficient Content ⚠", 0.5, "Not enough content", "Poor quality")
            st.session_state.classification_cache[cache_key] = result
            return result
        full_content = fallback_content
    
    # Perform classification
    label, confidence, quality_note = classify_with_confidence(full_content, bert_model, tokenizer)
    
    # Prepare display text
    display_text = full_content[:1500] + "..." if len(full_content) > 1500 else full_content
    
    result = (label, confidence, display_text, quality_note)
    st.session_state.classification_cache[cache_key] = result
    
    return result

# ------------------ ENHANCED FACT CHECKING SYSTEM ------------------
REPUTABLE_SOURCES = {
    "reuters.com", "bbc.com", "apnews.com", "theguardian.com", 
    "aljazeera.com", "washingtonpost.com", "nytimes.com",
    "indianexpress.com", "thehindu.com", "ndtv.com", "hindustantimes.com"
}

HIGH_REPUTE_DOMAINS = {"reuters", "ap", "bbc", "associated press", "pib"}

def extract_domain(url):
    """Extract domain from URL"""
    if not url:
        return "unknown"
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return "unknown"

def normalize(name):
    if not name:
        return "unknown"
    return name.strip().lower()

def is_reputable_source(article):
    """Check if article is from reputable source"""
    domain = extract_domain(article.get('url', ''))
    source_name = normalize(article.get('source', {}).get('name', ''))
    
    for rep_domain in REPUTABLE_SOURCES:
        if rep_domain in domain:
            return True
    
    for rep_source in HIGH_REPUTE_DOMAINS:
        if rep_source in source_name:
            return True
    
    return False

def calculate_content_similarity(text1, text2):
    """Calculate similarity between two texts using TF-IDF and cosine similarity"""
    if not text1 or not text2:
        return 0.0
    
    texts = [text1[:1000], text2[:1000]]
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        return 0.0

def extract_key_entities(text):
    """Extract key entities from text"""
    if not text:
        return set()
    
    entities = set()
    words = re.findall(r'\b[A-Z][a-z]+\b', text[:2000])
    
    common_words = {'The', 'A', 'An', 'And', 'But', 'Or', 'For', 'Nor', 'So', 'Yet'}
    
    for word in words:
        if (len(word) > 2 and 
            word not in common_words and 
            not word.isupper()):
            entities.add(word.lower())
    
    return entities

def search_related_news(query, lang="en", country="in", max_results=8):
    """Search for related news articles"""
    params = {
        "q": query,
        "lang": lang,
        "country": country,
        "max": max_results,
        "apikey": API_KEY
    }

    try:
        resp = requests.get(SEARCH_URL, params=params, timeout=8).json()
        return resp.get("articles", [])
    except Exception as e:
        print("Search error:", e)
        return []

def enhanced_fact_check(article, min_corroborating_sources=2, similarity_threshold=0.3):
    """Enhanced fact checking with multiple verification methods"""
    title = article.get('title', '')
    original_description = article.get('description', '')
    original_content = f"{title}. {original_description}"
    
    # Search for related articles
    related_articles = search_related_news(title)
    
    if not related_articles:
        return "Unverifiable ⚠ — No related articles found", []
    
    # Analyze articles for fact checking
    return analyze_articles_for_fact_checking(original_content, related_articles, similarity_threshold, min_corroborating_sources)

def analyze_articles_for_fact_checking(original_content, related_articles, similarity_threshold, min_sources):
    """Analyze related articles for fact checking verification"""
    original_entities = extract_key_entities(original_content)
    
    corroborating_articles = []
    reputable_corroborations = 0
    total_similar_articles = 0
    
    for related_article in related_articles:
        related_title = related_article.get('title', '')
        related_description = related_article.get('description', '')
        related_content = f"{related_title}. {related_description}"
        
        # Calculate content similarity
        similarity = calculate_content_similarity(original_content, related_content)
        
        # Extract entities from related article
        related_entities = extract_key_entities(related_content)
        
        # Calculate entity overlap
        entity_overlap = len(original_entities.intersection(related_entities)) if original_entities else 0
        
        # Consider article as corroborating if it meets similarity or entity overlap criteria
        if (similarity >= similarity_threshold or 
            (entity_overlap >= 2 and len(original_entities) >= 3)):
            
            total_similar_articles += 1
            corroborating_articles.append(related_article)
            
            # Check if from reputable source
            if is_reputable_source(related_article):
                reputable_corroborations += 1
    
    # Determine verdict based on analysis
    verdict = generate_fact_check_verdict(total_similar_articles, reputable_corroborations, min_sources)
    
    return verdict, corroborating_articles

def generate_fact_check_verdict(total_similar, reputable_count, min_sources):
    """Generate fact check verdict based on analysis results"""
    
    if total_similar == 0:
        return "Unverifiable ❌ — No corroborating sources found"
    
    elif reputable_count >= 2:
        return "Highly Credible ✅ — Multiple reputable sources confirm"
    
    elif total_similar >= min_sources and reputable_count >= 1:
        return "Credible ✅ — Corroborated by reputable sources"
    
    elif total_similar >= min_sources:
        return "Partially Corroborated ⚠ — Multiple sources but limited reputable confirmation"
    
    elif total_similar == 1:
        return "Limited Corroboration ⚠ — Only one source confirms"
    
    else:
        return "Weak Corroboration ❌ — Insufficient evidence"

# Add this debug function
def debug_classification(article):
    """Debug why an article is being misclassified"""
    title = article.get("title", "")
    
    print(f"=== DEBUGGING: {title} ===")
    
    # Check what content is being extracted
    ml_label, confidence, full_text, quality_note = enhanced_classify_news_full(article)
    print(f"ML Classification: {ml_label} (Confidence: {confidence})")
    print(f"Quality Note: {quality_note}")
    
    # Check fact checking
    fact_verdict, corroborating_articles = enhanced_fact_check(article)
    print(f"Fact Check Verdict: {fact_verdict}")
    print(f"Corroborating Sources Found: {len(corroborating_articles)}")
    
    # Check sources
    for i, source in enumerate(corroborating_articles[:3]):
        source_name = source.get('source', {}).get('name', 'Unknown')
        print(f"Source {i+1}: {source_name} - Reputable: {is_reputable_source(source)}")
    
    return ml_label, confidence, fact_verdict

# Use it in your main loop when you detect suspicious classification

def is_tech_rumor_mill(source_name):
    """Identify known rumor/clickbait websites"""
    rumor_sites = {
        'gizchina', 'gizmochina', 'phonearena', 'gadgets360', '91mobiles',
        'techradar', 'mysmartprice', 'yankodesign', 'concept-phones'
    }
    source_lower = source_name.lower()
    return any(rumor_site in source_lower for rumor_site in rumor_sites)

def enhanced_reputation_check(article):
    """More sophisticated source reputation checking"""
    source_name = normalize(article.get('source', {}).get('name', ''))
    domain = extract_domain(article.get('url', ''))
    
    # Check if it's a known rumor site
    if is_tech_rumor_mill(source_name) or is_tech_rumor_mill(domain):
        return "Rumor Site"
    
    # Existing reputable check
    if is_reputable_source(article):
        return "Reputable"
    
    return "Unknown"


def tech_news_specific_check(article):
    """Special checks for technology news"""
    title = article.get('title', '').lower()
    content = article.get('description', '').lower()
    
    # Patterns that indicate rumors/speculation
    rumor_indicators = [
        'leaked', 'rumor', 'upcoming', 'expected', 'might launch',
        'could feature', 'speculated', 'tipster', '2025', '2026',
        'concept', 'render', 'latest update', 'price leak'
    ]
    
    # Check for future dates (beyond current year + 1)
    future_year_pattern = r'20[2-9][4-9]'  # 2024+
    
    rumor_score = sum(1 for indicator in rumor_indicators if indicator in title or indicator in content)
    has_future_year = bool(re.search(future_year_pattern, title))
    
    if rumor_score >= 2 or has_future_year:
        return "Tech Rumor"
    
    return "Regular Tech News"

def enhanced_hybrid_decision_v2(ml_label, confidence, article):
    """Enhanced decision with tech rumor detection"""
    
    # Tech-specific check
    tech_verdict = tech_news_specific_check(article)
    if tech_verdict == "Tech Rumor":
        # Override to "Unverified" for tech rumors
        fact_verdict = "Unverified Tech Rumor ⚠"
        final_verdict = "Unverified Tech Rumor ⚠"
        return final_verdict, fact_verdict, []
    
    # Original fact checking
    fact_verdict, corroborating_articles = enhanced_fact_check(article)
    
    # Count reputable sources separately
    reputable_sources = sum(1 for article in corroborating_articles if is_reputable_source(article))
    total_corroborating = len(corroborating_articles)
    
    # Enhanced decision logic
    if tech_verdict == "Tech Rumor" and reputable_sources == 0:
        final_verdict = "Unverified Tech Rumor ⚠"
    else:
        # Use original decision logic
        final_verdict = ml_label
        
        # Override logic from before
        if ml_label == "Fake ❌" and reputable_sources >= 2:
            final_verdict = "Real ✅ (Strong corroboration)"
        elif ml_label == "Real ✅" and total_corroborating == 0 and confidence < 0.7:
            final_verdict = "Unverified ⚠"
    
    return final_verdict, fact_verdict, corroborating_articles

# ------------------ PERSONALIZED RECOMMENDATIONS ------------------
def extract_keywords(text, max_keywords=5):
    """Extract meaningful keywords from article text"""
    if not text:
        return []
    
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    words = [word for word in text.split() if len(word) > 3 and word not in stop_words]
    
    word_freq = Counter(words)
    keywords = [word for word, count in word_freq.most_common(max_keywords)]
    
    return keywords

def add_user_interest(article, full_text):
    """Add user interest based on article content"""
    title = article.get("title", "")
    url = article.get("url", "")
    
    if url in st.session_state.viewed_articles:
        return
    
    st.session_state.viewed_articles.add(url)
    
    content_for_keywords = f"{title} {full_text[:500]}"
    keywords = extract_keywords(content_for_keywords)
    
    if keywords:
        interest_data = {
            "timestamp": datetime.now(),
            "keywords": keywords,
            "title": title,
            "category": st.session_state.get('current_topic', 'general')
        }
        st.session_state.interests.append(interest_data)
        
        if len(st.session_state.interests) > 20:
            st.session_state.interests = st.session_state.interests[-20:]

def get_user_topics():
    """Extract most frequent topics from user interests"""
    if not st.session_state.interests:
        return []
    
    recent_cutoff = datetime.now() - timedelta(hours=24)
    recent_interests = [
        interest for interest in st.session_state.interests 
        if interest["timestamp"] > recent_cutoff
    ]
    
    if not recent_interests:
        recent_interests = st.session_state.interests[-10:]
    
    all_keywords = []
    for interest in recent_interests:
        all_keywords.extend(interest["keywords"])
    
    keyword_freq = Counter(all_keywords)
    return [keyword for keyword, count in keyword_freq.most_common(5)]

def generate_recommendations(force_refresh=False):
    """Generate personalized recommendations based on user interests"""
    
    current_time = datetime.now()
    if (not force_refresh and 
        st.session_state.last_recommendation_update and 
        (current_time - st.session_state.last_recommendation_update).seconds < 600 and
        st.session_state.recommendations):
        return st.session_state.recommendations
    
    user_topics = get_user_topics()
    if not user_topics:
        return []
    
    recommendations = []
    seen_urls = set(st.session_state.viewed_articles)
    
    for topic in user_topics[:3]:
        params = {
            "q": topic,
            "lang": "en",
            "country": "in",
            "max": 5,
            "apikey": API_KEY
        }
        try:
            resp = requests.get(SEARCH_URL, params=params, timeout=8).json()
            for article in resp.get("articles", []):
                url = article.get("url")
                if url and url not in seen_urls:
                    if (article.get("title") and 
                        len(article.get("title", "")) > 10 and
                        article.get("source", {}).get("name")):
                        
                        title_lower = article.get("title", "").lower()
                        relevance = sum(1 for t in user_topics if t in title_lower)
                        article["relevance_score"] = relevance
                        article["topic"] = topic
                        
                        recommendations.append(article)
                        seen_urls.add(url)
                        
                if len(recommendations) >= 15:
                    break
                    
        except Exception as e:
            print(f"Recommendation generation error for topic '{topic}': {e}")
            continue
    
    recommendations.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    final_recommendations = recommendations[:8]
    
    st.session_state.recommendations = final_recommendations
    st.session_state.last_recommendation_update = current_time
    
    return final_recommendations

# ------------------ UI DESIGN ------------------
st.set_page_config(page_title="Fake News Detector", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #001f3f; color: white; }
[data-testid="stSidebar"] { background-color: #002b5c; color: white; }
div.stButton > button { background-color:#002b5c; color:white; border-radius:8px; }
.recommendation-header { 
    background: linear-gradient(90deg, #002b5c, #0044cc);
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.verdict-real { color: #00ff00; font-weight: bold; }
.verdict-fake { color: #ff4444; font-weight: bold; }
.verdict-uncertain { color: #ffaa00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("📰 Enhanced Fake News Detector + Fact Checking + Personalized Recommendations")

# ------------------ SIDEBAR ------------------
st.sidebar.title("🚨 Fake News Detected")
fake_news_list = []

# ------------------ MAIN APPLICATION ------------------
categories = [
    "breaking-news", "world", "nation", "business",
    "technology", "entertainment", "sports", "science", "health"
]

topic = st.selectbox("Choose News Category", categories)
st.session_state.current_topic = topic

if "page" not in st.session_state:
    st.session_state.page = 1

col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("⬅ Previous") and st.session_state.page > 1:
        st.session_state.page -= 1

with col_next:
    if st.button("Next ➡"):
        st.session_state.page += 1

st.info(f"Page {st.session_state.page}")

# FIXED: Now fetch_news function is defined and will work
articles = fetch_news(topic=topic, page=st.session_state.page)

col1, col2 = st.columns([2, 1])

# ------------------ MAIN COLUMN ------------------
with col1:
    st.subheader(f"Latest {topic.capitalize()} News")

    if not articles:
        st.warning("No articles found. Please try a different category or check your internet connection.")
    else:
        for idx, a in enumerate(articles):
            st.markdown("---")
            title = a.get('title') or "Untitled"

            st.write(f"### {title}")

            image = a.get("image")
            url = a.get("url")

            try:
                if image:
                    st.image(image, width=450)
                else:
                    st.image(IMAGE_FALLBACK, width=450)
            except Exception:
                try:
                    st.image(IMAGE_FALLBACK, width=450)
                except:
                    pass

            # ENHANCED CLASSIFICATION
            ml_label, confidence, full_text, quality_note = enhanced_classify_news_full(a)

            # ENHANCED FACT CHECKING
            final, fact_verdict, related =  enhanced_hybrid_decision_v2(ml_label, confidence, a)

            # Display results with enhanced styling
            st.write(f"**AI Prediction:** {ml_label} (Confidence: {confidence:.2f})")
            if quality_note != "Good quality text":
                st.caption(f"*Note: {quality_note}*")
            
            st.write(f"**Fact Check:** {fact_verdict}")
            
            # Color-coded final verdict
            if "Real" in final:
                st.markdown(f"<p class='verdict-real'>**Final Verdict:** {final}</p>", unsafe_allow_html=True)
            elif "Fake" in final:
                st.markdown(f"<p class='verdict-fake'>**Final Verdict:** {final}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='verdict-uncertain'>**Final Verdict:** {final}</p>", unsafe_allow_html=True)

            if url:
                st.markdown(f"[Read Full Article 🔗]({url})")

            # Expand for analyzed text and interest tracking
            with st.expander("Show Analyzed Text"):
                st.write(full_text)
                add_user_interest(a, full_text)

            # If fake -> add to sidebar list
            if "Fake" in final or "fake" in final.lower():
                fake_news_list.append(a)

            # Fact check details
            with st.expander("🔍 Fact Check Details"):
                reputable_count = sum(1 for article in related if is_reputable_source(article))
                st.write(f"**Corroborating Sources:** {len(related)}")
                st.write(f"**Reputable Sources:** {reputable_count}")
                
                if related:
                    st.write("**Top Corroborating Articles:**")
                    for i, source in enumerate(related[:3]):
                        source_name = source.get('source', {}).get('name', 'Unknown')
                        is_rep = "✅" if is_reputable_source(source) else "⚪"
                        st.write(f"{is_rep} **{source_name}**: {source.get('title')}")
                        if source.get('url'):
                            st.markdown(f"[Read Source]({source.get('url')})")

# ------------------ PERSONALIZED RECOMMENDATIONS ------------------
st.markdown("---")
st.markdown('<div class="recommendation-header">', unsafe_allow_html=True)
st.subheader("⭐ Recommended For You (Based on Your Interests)")
st.markdown('</div>', unsafe_allow_html=True)

col_refresh, col_info = st.columns([1, 4])
with col_refresh:
    if st.button("🔄 Refresh Recommendations"):
        st.session_state.recommendations = []
        st.session_state.last_recommendation_update = None
        st.rerun()

with col_info:
    if st.session_state.interests:
        user_topics = get_user_topics()
        if user_topics:
            st.write(f"**Your interests:** {', '.join(user_topics[:3])}")

if not st.session_state.interests:
    st.info("👀 View and expand some articles to get personalized recommendations based on your interests!")
else:
    recommendations = generate_recommendations()
    
    if recommendations:
        cols = st.columns(2)
        for idx, rec in enumerate(recommendations[:6]):
            with cols[idx % 2]:
                with st.container():
                    st.markdown("---")
                    try:
                        st.image(rec.get("image", IMAGE_FALLBACK), width=300)
                    except Exception:
                        try:
                            st.image(IMAGE_FALLBACK, width=300)
                        except:
                            pass
                    
                    st.write(f"**{rec.get('title')}**")
                    st.caption(f"Source: {rec.get('source',{}).get('name')}")
                    st.caption(f"Topic: {rec.get('topic', 'General')}")
                    
                    if rec.get("url"):
                        st.markdown(f"[Read More 🔗]({rec.get('url')})")
                    
                    relevance = rec.get("relevance_score", 0)
                    if relevance > 0:
                        st.progress(min(relevance/3, 1.0), text=f"Relevance: {relevance}")
    else:
        st.info("No new recommendations found. Try refreshing or viewing more articles.")

# ------------------ SIDEBAR CONTENT ------------------
with col2:
    st.subheader("🚨 Fake News Today")
    if fake_news_list:
        for f in fake_news_list:
            st.error(f"**{f.get('title')}**")
            if f.get("url"):
                st.markdown(f"[Read More 🔗]({f.get('url')})")
    else:
        st.success("No fake news detected yet.")

# Debug section
with st.expander("🔧 Debug Information"):
    st.write(f"**Interests tracked:** {len(st.session_state.interests)}")
    st.write(f"**Articles viewed:** {len(st.session_state.viewed_articles)}")
    st.write(f"**Cache size:** {len(st.session_state.classification_cache)}")
    
    if st.button("Clear Cache"):
        st.session_state.classification_cache = {}
        st.session_state.recommendations = []
        st.session_state.last_recommendation_update = None
        st.rerun()