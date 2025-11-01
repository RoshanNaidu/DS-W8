from collections import defaultdict
import numpy as np
import re

class MarkovText:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tokens = self.corpus.split()
        self.term_dict = self.get_term_dict()

    def get_term_dict(self):
        """Generate a dictionary where each term is mapped to a list of possible following terms."""
        term_dict = defaultdict(list)
        for i in range(len(self.tokens) - 1):
            current_term = self.tokens[i]
            next_term = self.tokens[i + 1]
            term_dict[current_term].append(next_term)
        return term_dict

    def generate(self, term_count=100, seed_term=None):
        """
        Generates text using a simple 1-word Markov chain.
        - Raises ValueError if seed_term is invalid or corpus too short.
        - Handles gracefully if seed_term is near the end of corpus.
        """
        # Ensure term_count is an integer
        try:
            term_count = int(term_count)
        except ValueError:
            raise ValueError(f"Invalid term_count value: '{term_count}'. It must be an integer.")

        # Edge case: corpus too short
        if len(self.tokens) < 2:
            raise ValueError("Corpus must contain at least two words to build transitions.")

        # Validate seed term
        if seed_term:
            if seed_term not in self.term_dict:
                # Raise an error if the seed term isn't in the term_dict (fix for 2.3)
                raise ValueError(f"Seed term '{seed_term}' not found in the corpus.")
            current_term = seed_term
        else:
            current_term = np.random.choice(list(self.term_dict.keys()))

        generated_text = [current_term]

        # Generate the remaining terms
        for _ in range(term_count - 1):
            next_terms = self.term_dict.get(current_term)
            if not next_terms:
                # If no next terms exist, we want to stop. (Fix for 2.2)
                break
            next_term = np.random.choice(next_terms)
            generated_text.append(next_term)
            current_term = next_term

        return ' '.join(generated_text)


class MarkovText_k:
    def __init__(self, corpus, k=1):
        self.corpus = corpus
        self.k = k
        self.tokens = self.corpus.split()
        self.term_dict = self.get_term_dict()

    def get_term_dict(self):
        term_dict = defaultdict(list)
        for i in range(len(self.tokens) - self.k):
            current_state = tuple(self.tokens[i:i + self.k])
            next_term = self.tokens[i + self.k]
            term_dict[current_state].append(next_term)
        return term_dict

    def generate(self, term_count=100, seed_term=None):
        if seed_term:
            if isinstance(seed_term, str):
                seed_tokens = seed_term.split()
                if len(seed_tokens) != self.k:
                    raise ValueError(f"Seed term must have {self.k} words.")
                seed_state = tuple(seed_tokens)
            elif isinstance(seed_term, (list, tuple)):
                if len(seed_term) != self.k:
                    raise ValueError(f"Seed term must have {self.k} words.")
                seed_state = tuple(seed_term)
            else:
                raise ValueError("Seed term must be string, list, or tuple.")

            if seed_state not in self.term_dict:
                raise ValueError(f"Seed state '{seed_state}' not found in the corpus.")  # Fix for 2.3
            current_state = seed_state
        else:
            keys = list(self.term_dict.keys())
            current_state = keys[np.random.choice(len(keys))]

        generated_text = list(current_state)

        # Ensure term_count is an integer
        try:
            term_count = int(term_count)
        except ValueError:
            raise ValueError(f"Invalid term_count value: '{term_count}'. It must be an integer.")

        for _ in range(term_count - self.k):
            next_terms = self.term_dict.get(current_state)
            if not next_terms:
                break
            next_term = np.random.choice(next_terms)
            generated_text.append(next_term)
            current_state = tuple(generated_text[-self.k:])

        return ' '.join(generated_text)


def clean_text(quotes_raw):
    quotes = quotes_raw.replace('\n', ' ')
    quotes = re.split("[“”]", quotes)
    quotes = quotes[1::2]
    corpus = ' '.join(quotes)
    corpus = re.sub(r"\s+", " ", corpus)
    return corpus.strip()
