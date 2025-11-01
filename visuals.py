# plots_helper.py
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import streamlit as st
from collections import defaultdict

def generate_wordcloud(text, title="Word Cloud"):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title, fontsize=16)
    st.pyplot(fig)

def plot_word_frequencies(text, top_n=20, title="Word Frequency"):
    words = text.lower().split()
    freq = Counter(words)
    common_words = freq.most_common(top_n)
    words, counts = zip(*common_words)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(counts), y=list(words), palette="viridis", ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Words")
    st.pyplot(fig)

def plot_transition_heatmap(term_dict, k, top_n=10):
    if k != 1:
        st.info("Heatmap only supported for k=1 for simplicity.")
        return
    
    transitions = defaultdict(Counter)
    for current_word, next_words in term_dict.items():
        if isinstance(current_word, tuple):
            current_word = current_word[0]
        for next_word in next_words:
            transitions[current_word][next_word] += 1

    df = pd.DataFrame(transitions).fillna(0)
    df = df.div(df.sum(axis=0), axis=1)

    top_words = df.sum(axis=1).nlargest(top_n).index
    df = df.loc[top_words, top_words]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, annot=False, cmap="YlGnBu", ax=ax)
    ax.set_title("Conditional Probability Heatmap")
    st.pyplot(fig)
