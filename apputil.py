from collections import defaultdict
import numpy as np
import re

class MarkovText(object):

    def __init__(self, corpus):
        k=1
        self.corpus = corpus
        self.k = int(k)
        if self.k < 1:
            raise ValueError("k must be >= 1")
        # very light tokenization: split on whitespace
        self.tokens = self.corpus.split()
        if len(self.tokens) < self.k + 1:
            raise ValueError("Corpus too small for chosen k")
        self.term_dict = None  # you'll need to build this

    def get_term_dict(self):

        trans = defaultdict(list)
        toks = self.tokens
        k = self.k

        # iterate states and push following token
        for i in range(len(toks) - k):
            state = toks[i] if k == 1 else tuple(toks[i:i+k])
            nxt = toks[i + k]
            trans[state].append(nxt)

        # save
        self.term_dict = dict(trans)
        return self.term_dict


    def generate(self, seed=None, term_count=15):

        if self.term_dict is None:
            self.get_term_dict()

        k = self.k
        # Normalize/validate seed
        if seed is not None:
            if k == 1:
                state = seed
            else:
                # seed can be string or tuple/list of length k
                if isinstance(seed, str):
                    # allow string with spaces
                    parts = seed.split()
                else:
                    parts = list(seed)
                if len(parts) != k:
                    raise ValueError(f"Seed must have {k} tokens for k={k}")
                state = tuple(parts)
            if state not in self.term_dict:
                raise ValueError("Seed term not found in corpus/state space.")
        else:
            # pick a random valid starting state
            keys = list(self.term_dict.keys())
            state = keys[np.random.randint(0, len(keys))]

        # initialize output with state
        out = []
        if k == 1:
            out.append(state)
        else:
            out.extend(list(state))

        # generate
        for _ in range(max(0, term_count - (len(out)))):
            choices = self.term_dict.get(state, [])
            if not choices:
                # dead end: random re-seed
                keys = list(self.term_dict.keys())
                state = keys[np.random.randint(0, len(keys))]
                if k == 1:
                    out.append(state)
                else:
                    out.extend(list(state))
                continue

            nxt = np.random.choice(choices)
            out.append(nxt)

            # roll the state
            if k == 1:
                state = nxt
            else:
                state = tuple(list(state)[1:] + [nxt])

        return " ".join(out)


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
        # Log term_count to verify input
        print(f"Received term_count: {term_count}, Type: {type(term_count)}")
        
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
