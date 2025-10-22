#Exercise 1
'''
from apputil import *
from collections import defaultdict

class MarkovText:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tokens = self.corpus.split()
        self.term_dict = self.get_term_dict()

    def get_term_dict(self):
        """
        Builds a term dictionary where keys are unique tokens and values are lists
        of tokens that follow each key.
        """
        term_dict = defaultdict(list)
        for i in range(len(self.tokens) - 1):
            current_term = self.tokens[i]
            next_term = self.tokens[i+1]
            term_dict[current_term].append(next_term)
        return term_dict

    def generate(self, term_count=100, seed_term=None):
        """
        Generates text using the Markov property.
        """
        if seed_term and seed_term not in self.term_dict:
            raise ValueError(f"Seed term '{seed_term}' not found in the corpus.")

        if seed_term:
            current_term = seed_term
        else:
            current_term = np.random.choice(list(self.term_dict.keys()))

        generated_text = [current_term]

        for _ in range(term_count - 1):
            if current_term in self.term_dict:
                next_terms = self.term_dict[current_term]
                if next_terms:
                    next_term = np.random.choice(next_terms)
                    generated_text.append(next_term)
                    current_term = next_term
                else:
                    # Handle cases where a term is the last word in the corpus
                    break
            else:
                # Handle cases where a term doesn't have a following term in the dict
                break


        return ' '.join(generated_text)
'''

#Exercise 2 Code
'''
# Generate text using the implemented generate method
generated_text = text_gen.generate(term_count=50)
print(generated_text)
'''

#Bonus Code
'''
from collections import defaultdict
import numpy as np

class MarkovText_k:
    def __init__(self, corpus, k=1):
        self.corpus = corpus
        self.k = k
        self.tokens = self.corpus.split()
        self.term_dict = self.get_term_dict()

    def get_term_dict(self):
        """
        Builds a term dictionary where keys are unique tokens (or tuples of k tokens)
        and values are lists of tokens that follow each key.
        """
        term_dict = defaultdict(list)
        for i in range(len(self.tokens) - self.k):
            current_state = tuple(self.tokens[i : i + self.k])
            next_term = self.tokens[i + self.k]
            term_dict[current_state].append(next_term)
        return term_dict

    def generate(self, term_count=100, seed_term=None):
        """
        Generates text using the Markov property with k-word states.
        """
        if seed_term:
            if isinstance(seed_term, str):
                seed_tokens = seed_term.split()
                if len(seed_tokens) != self.k:
                    raise ValueError(f"Seed term must be a string with {self.k} words for k={self.k}.")
                seed_state = tuple(seed_tokens)
            elif isinstance(seed_term, (list, tuple)):
                if len(seed_term) != self.k:
                     raise ValueError(f"Seed term must be a list or tuple with {self.k} words for k={self.k}.")
                seed_state = tuple(seed_term)
            else:
                 raise ValueError("Seed term must be a string, list, or tuple.")

            if seed_state not in self.term_dict:
                 raise ValueError(f"Seed state '{seed_state}' not found in the corpus.")
            current_state = seed_state
        else:
            # Fix: Get the keys as a list before passing to np.random.choice
            keys = list(self.term_dict.keys())
            current_state = keys[np.random.choice(len(keys))]


        generated_text = list(current_state)

        for _ in range(term_count - self.k):
            if current_state in self.term_dict:
                next_terms = self.term_dict[current_state]
                if next_terms:
                    next_term = np.random.choice(next_terms)
                    generated_text.append(next_term)
                    current_state = tuple(generated_text[-self.k:])
                else:
                    # Handle cases where a state is the last sequence in the corpus
                    break
            else:
                # Handle cases where a state doesn't have a following term in the dict
                break

        return ' '.join(generated_text)
''' 


# from collections import defaultdict
# import numpy as np
# import requests
# import re

# class MarkovText_k:
#     def __init__(self, corpus, k=1):
#         self.corpus = corpus
#         self.k = k
#         self.tokens = self.corpus.split()
#         self.term_dict = self.get_term_dict()

#     def get_term_dict(self):
#         """
#         Builds a term dictionary where keys are unique tokens (or tuples of k tokens)
#         and values are lists of tokens that follow each key.
#         """
#         term_dict = defaultdict(list)
#         for i in range(len(self.tokens) - self.k):
#             current_state = tuple(self.tokens[i : i + self.k])
#             next_term = self.tokens[i + self.k]
#             term_dict[current_state].append(next_term)
#         return term_dict

#     def generate(self, term_count=100, seed_term=None):
#         """
#         Generates text using the Markov property with k-word states.
#         """
#         if seed_term:
#             if isinstance(seed_term, str):
#                 seed_tokens = seed_term.split()
#                 if len(seed_tokens) != self.k:
#                     raise ValueError(f"Seed term must be a string with {self.k} words for k={self.k}.")
#                 seed_state = tuple(seed_tokens)
#             elif isinstance(seed_term, (list, tuple)):
#                 if len(seed_term) != self.k:
#                      raise ValueError(f"Seed term must be a list or tuple with {self.k} words for k={self.k}.")
#                 seed_state = tuple(seed_term)
#             else:
#                  raise ValueError("Seed term must be a string, list, or tuple.")

#             if seed_state not in self.term_dict:
#                  raise ValueError(f"Seed state '{seed_state}' not found in the corpus.")
#             current_state = seed_state
#         else:
#             keys = list(self.term_dict.keys())
#             current_state = keys[np.random.choice(len(keys))]


#         generated_text = list(current_state)

#         for _ in range(term_count - self.k):
#             if current_state in self.term_dict:
#                 next_terms = self.term_dict[current_state]
#                 if next_terms:
#                     next_term = np.random.choice(next_terms)
#                     generated_text.append(next_term)
#                     current_state = tuple(generated_text[-self.k:])
#                 else:
#                     break
#             else:
#                 break

#         return ' '.join(generated_text)

# def clean_text(quotes_raw):
#     quotes = quotes_raw.replace('\n', ' ')
#     quotes = re.split("[“”]", quotes)
#     quotes = quotes[1::2]
#     corpus = ' '.join(quotes)
#     corpus = re.sub(r"\s+", " ", corpus)
#     corpus = corpus.strip()
#     return corpus



from wordcloud import WordCloud
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd


from collections import defaultdict
import numpy as np

class MarkovText_k:
    def __init__(self, corpus, k=1):
        self.corpus = corpus
        self.k = k
        self.tokens = self.corpus.split()
        self.term_dict = self.get_term_dict()

    def get_term_dict(self):
        """
        Builds a term dictionary where keys are unique tokens (or tuples of k tokens)
        and values are lists of tokens that follow each key.
        """
        term_dict = defaultdict(list)
        for i in range(len(self.tokens) - self.k):
            current_state = tuple(self.tokens[i : i + self.k])
            next_term = self.tokens[i + self.k]
            term_dict[current_state].append(next_term)
        return term_dict

    def generate(self, term_count=100, seed_term=None):
        """
        Generates text using the Markov property with k-word states.
        """
        if seed_term:
            if isinstance(seed_term, str):
                seed_tokens = seed_term.split()
                if len(seed_tokens) != self.k:
                    raise ValueError(f"Seed term must be a string with {self.k} words for k={self.k}.")
                seed_state = tuple(seed_tokens)
            elif isinstance(seed_term, (list, tuple)):
                if len(seed_term) != self.k:
                    raise ValueError(f"Seed term must be a list or tuple with {self.k} words for k={self.k}.")
                seed_state = tuple(seed_term)
            else:
                raise ValueError("Seed term must be a string, list, or tuple.")

            if seed_state not in self.term_dict:
                raise ValueError(f"Seed state '{seed_state}' not found in the corpus.")
            current_state = seed_state
        else:
            keys = list(self.term_dict.keys())
            current_state = keys[np.random.choice(len(keys))]

        generated_text = list(current_state)

        for _ in range(term_count - self.k):
            if current_state in self.term_dict:
                next_terms = self.term_dict[current_state]
                if next_terms:
                    next_term = np.random.choice(next_terms)
                    generated_text.append(next_term)
                    current_state = tuple(generated_text[-self.k:])
                else:
                    break
            else:
                break

        return ' '.join(generated_text)


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

    # Convert to probability DataFrame
    df = pd.DataFrame(transitions).fillna(0)
    df = df.div(df.sum(axis=0), axis=1)  # Normalize columns to get probabilities

    # Limit to top N rows and columns
    top_words = df.sum(axis=1).nlargest(top_n).index
    df = df.loc[top_words, top_words]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, annot=False, cmap="YlGnBu", ax=ax)
    ax.set_title("Conditional Probability Heatmap")
    st.pyplot(fig)
import re

def clean_text(quotes_raw):
    quotes = quotes_raw.replace('\n', ' ')
    quotes = re.split("[“”]", quotes)
    quotes = quotes[1::2]  # Extract quoted content
    corpus = ' '.join(quotes)
    corpus = re.sub(r"\s+", " ", corpus)
    corpus = corpus.strip()
    return corpus
