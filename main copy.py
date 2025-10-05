import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from newspaper import Article
from groq import Groq
import streamlit as st

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not NEWS_API_KEY or not GROQ_API_KEY:
    st.error("❌ Please set NEWS_API_KEY and GROQ_API_KEY in the .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.1-8b-instant"


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

    prompt = f"""Generate a cohesive report (200-300 words) based on these article summaries. 
    Highlight common themes, advancements, and implications:\n\n{report_content}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def agentic_query(user_query, num_articles=4, days_back=30):
    articles = fetch_recent_articles(user_query, num_articles, days_back)
    if not articles:
        return "No recent articles found for this query.", [], []

    summaries = []
    for art in articles:
        full_text = extract_full_text(art['url'])
        summary = summarize_article(full_text)
        summaries.append(summary)

    report = generate_report(articles, summaries)
    return report, articles, summaries


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RapidReads", page_icon="logo.png", layout="wide")

col1, col2 = st.columns([1, 9])
with col1:
    st.image("logo.png", width=80)
with col2:
    st.title("📖 RapidReads")
    st.markdown("AI-powered agent that fetches recent articles, summarizes them, and generates a cohesive report.")

st.markdown("---")

query = st.text_input("🔍 Enter a topic or query:", placeholder="e.g., recent advancements in medical science")
num_articles = st.slider("Number of articles to fetch", 2, 10, 4)

if st.button("Fetch & Summarize"):
    if not query.strip():
        st.warning("⚠️ Please enter a query first.")
    else:
        with st.spinner("Fetching and analyzing articles..."):
            report, articles, summaries = agentic_query(query, num_articles=num_articles)

        st.subheader("📝 AI-Generated Report")
        st.write(report)

        st.markdown("---")
        st.subheader("📑 Individual Articles")

        for i, (art, summary) in enumerate(zip(articles, summaries), 1):
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #ddd;
                        border-radius: 12px;
                        padding: 16px;
                        margin-bottom: 15px;
                        background-color: #f9f9f9;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                    ">
                        <h4 style="margin-bottom: 10px;">{i}. {art['title']}</h4>
                        <p style="margin-bottom: 12px; font-size: 15px; line-height: 1.6;">
                            {summary}
                        </p>
                        <a href="{art['url']}" target="_blank" style="
                            text-decoration: none;
                            background-color: #4CAF50;
                            color: white;
                            padding: 8px 14px;
                            border-radius: 8px;
                            font-size: 14px;
                        ">
                            Read More 👉
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
