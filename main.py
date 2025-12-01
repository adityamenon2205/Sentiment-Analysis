import pandas as pd
import re
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --------------------- CONFIG ---------------------
CSV_FILE = "stock_data.csv"   # rename your csv to this

# --------------------- LOAD DATA ------------------
print("\n📌 Loading dataset...")
df = pd.read_csv(CSV_FILE, encoding="utf-8")

# Detect text column automatically
text_col = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())
print(f"✔ Using text column → {text_col}")

# ---------------- TEXT CLEANING -------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df[text_col].apply(clean_text)

# ---------------- FIX NLTK + SENTIMENT -----------------------
print("\n🧠 Running sentiment analysis...")

# Safe vader downloader – avoids SSL failures
try:
    sia = SentimentIntensityAnalyzer()
except:
    print("🔄 Vader Lexicon missing — downloading...\n")
    nltk.set_proxy('http://')  # bypass SSL check
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()

df["polarity"] = df["clean_text"].apply(lambda t: TextBlob(t).sentiment.polarity)
df["vader_score"] = df["clean_text"].apply(lambda t: sia.polarity_scores(t)["compound"])

# Label sentiment
def label(score):
    return "Positive" if score>=0.25 else "Negative" if score<=-0.25 else "Neutral"

df["sentiment"] = df["vader_score"].apply(label)
print("\n📊 Sentiment Distribution:\n", df["sentiment"].value_counts(), "\n")

# ---------------- WORD CLOUDS --------------------
print("☁ Generating sentiment-specific WordClouds...")

pos_text = " ".join(df[df.sentiment=="Positive"]["clean_text"])
neg_text = " ".join(df[df.sentiment=="Negative"]["clean_text"])
neu_text = " ".join(df[df.sentiment=="Neutral"]["clean_text"])

wc_pos = WordCloud(width=1000, height=600, background_color="white", colormap="Greens").generate(pos_text)
wc_neg = WordCloud(width=1000, height=600, background_color="white", colormap="Reds").generate(neg_text)
wc_neu = WordCloud(width=1000, height=600, background_color="white", colormap="Blues").generate(neu_text)

fig, axes = plt.subplots(1,3, figsize=(20,7))
axes[0].imshow(wc_pos); axes[0].set_title("Positive Sentiment", fontsize=18)
axes[1].imshow(wc_neg); axes[1].set_title("Negative Sentiment", fontsize=18)
axes[2].imshow(wc_neu); axes[2].set_title("Neutral Sentiment", fontsize=18)

for ax in axes: ax.axis("off")
plt.tight_layout()
plt.show()

# Save output
wc_pos.to_file("positive_wordcloud.png")
wc_neg.to_file("negative_wordcloud.png")
wc_neu.to_file("neutral_wordcloud.png")
df.to_csv("sentiment_results_with_wordcloud.csv", index=False)

print("\n💾 Wordclouds saved as PNG files")
print("   ➤ positive_wordcloud.png")
print("   ➤ negative_wordcloud.png")
print("   ➤ neutral_wordcloud.png")
print("\n✔ Everything completed successfully!")