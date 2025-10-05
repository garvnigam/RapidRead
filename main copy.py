import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from newspaper import Article
from groq import Groq
import streamlit as st

# ---------------- Environment Setup ----------------
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not NEWS_API_KEY or not GROQ_API_KEY:
    st.error("❌ Please set NEWS_API_KEY and GROQ_API_KEY in the .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.1-8b-instant"


# ---------------- Utility Functions ----------------
def fetch_recent_articles(query, num_articles=4, days_back=30):
    url = 'https://newsapi.org/v2/everything'
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    params = {
        'q': query,
        'from': from_date,
        'sortBy': 'publishedAt',
        'apiKey': NEWS_API_KEY,
        'pageSize': num_articles,
        'language': 'en'
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    articles = []
    for item in data.get('articles', []):
        if item['url']:
            articles.append({
                'title': item['title'],
                'url': item['url'],
                'description': item['description'] or '',
                'publishedAt': item['publishedAt']
            })
    return articles


def extract_full_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error extracting {url}: {e}"


def summarize_article(text, max_tokens=150):
    if not text or len(text.strip()) < 50:
        return "Unable to summarize: Insufficient content."

    prompt = f"Summarize the key points of this article in 3-5 concise sentences:\n\n{text[:2000]}"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()


def generate_report(articles, summaries):
    report_content = ""
    for i, (art, summary) in enumerate(zip(articles, summaries), 1):
        report_content += f"### {i}. {art['title']}\n"
        report_content += f"- **Summary:** {summary}\n"
        report_content += f"- [Read full article]({art['url']})\n\n"

    prompt = f"""Create a cohesive report (200–300 words) summarizing the key insights and patterns 
    from these article summaries:\n\n{report_content}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def get_news_summary(query, num_articles=4, days_back=30):
    articles = fetch_recent_articles(query, num_articles, days_back)
    if not articles:
        return "No recent articles found for this topic.", [], []

    summaries = []
    for art in articles:
        full_text = extract_full_text(art['url'])
        summary = summarize_article(full_text)
        summaries.append(summary)

    report = generate_report(articles, summaries)
    return report, articles, summaries


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RapidReads", page_icon="logo.png", layout="wide")

# 🌈 Custom White-Themed CSS
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF;
        color: #000000;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    h1, h2, h3, h4, h5, h6, p {
        color: #111111 !important;
        font-family: "Inter", sans-serif;
    }

    .article-card {
        border: 1px solid #e6e6e6;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.2s ease-in-out;
    }
    .article-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.08);
    }

    .read-btn {
        text-decoration: none;
        background-color: #4CAF50;
        color: white;
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
    }
    .read-btn:hover {
        background-color: #45a049;
    }

    input, textarea {
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Header Section ----------------
col1, col2 = st.columns([1, 9])
with col1:
    st.image("logo.png", width=80)
with col2:
    st.markdown("<h1 style='font-size: 36px; margin-bottom: -10px;'>📖 RapidReads</h1>", unsafe_allow_html=True)
    st.markdown("#### Stay informed with quick summaries and reports from the latest articles.")

st.markdown("---")

# ---------------- User Input ----------------
query = st.text_input("🔍 Enter a topic or keyword:", placeholder="e.g., recent advancements in renewable energy")
num_articles = st.slider("Number of articles to fetch", 2, 10, 4)

# ---------------- Action Button ----------------
if st.button("🚀 Generate Report"):
    if not query.strip():
        st.warning("⚠️ Please enter a topic first.")
    else:
        with st.spinner("Fetching and processing articles..."):
            report, articles, summaries = get_news_summary(query, num_articles=num_articles)

        # ---------------- Report Section ----------------
        st.markdown("### 📝 Summary Report")
        st.markdown(
            f"""
            <div style="background-color:#f7f7f7; padding:20px; border-radius:12px; border:1px solid #e6e6e6;">
            {report}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # ---------------- Individual Articles ----------------
        st.markdown("### 📑 Article Summaries")

        for i, (art, summary) in enumerate(zip(articles, summaries), 1):
            st.markdown(
                f"""
                <div class="article-card">
                    <h4 style="margin-bottom: 10px;">{i}. {art['title']}</h4>
                    <p style="margin-bottom: 12px; font-size: 15px; line-height: 1.6;">
                        {summary}
                    </p>
                    <a href="{art['url']}" target="_blank" class="read-btn">Read More 👉</a>
                </div>
                """,
                unsafe_allow_html=True
            )
