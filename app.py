# import streamlit as st
# import requests
# from apputil import *

# st.write(
# '''
# # Week 8: NLP Basics

# ## Markov Chain Text Generator

# Generate text in the style of inspirational quotes using a Markov Chain...
# ''')

# # Get the corpus data
# url = 'https://raw.githubusercontent.com/leontoddjohnson/datasets/main/text/inspiration_quotes.txt'
# content = requests.get(url)
# quotes_raw = content.text
# corpus = clean_text(quotes_raw)

# # Add a text area for user input (optional, can use the predefined corpus)
# user_corpus = st.text_area("Or paste your own text corpus here (optional):", height=200)
# if user_corpus:
#     corpus = clean_text(user_corpus)

# # Add a slider for selecting the k value
# k_value = st.slider("Select the value of k (state window size):", 1, 5, 2)

# # Add an input field for the seed term
# seed_term = st.text_input(f"Enter a seed term (optional, requires {k_value} word(s)):")

# # Add a slider for the number of terms to generate
# term_count = st.slider("Select the number of terms to generate:", 50, 500, 100)


# # Add a button to generate text
# if st.button("Generate Text"):
#     if corpus:
#         try:
#             text_gen = MarkovText_k(corpus, k=k_value)
#             generated_text = text_gen.generate(term_count=term_count, seed_term=seed_term if seed_term else None)
#             st.write("### Generated Text:")
#             st.write(generated_text)
#         except ValueError as e:
#             st.error(f"Error: {e}")
#         except Exception as e:
#             st.error(f"An unexpected error occurred: {e}")
#     else:
#         st.warning("Please provide a text corpus to generate from.")





import streamlit as st
import requests
from apputil import *

st.write(
'''
# Week 8: NLP Basics

## Markov Chain Text Generator

Generate text in the style of inspirational quotes using a Markov Chain...
''')

# Load corpus
url = 'https://raw.githubusercontent.com/leontoddjohnson/datasets/main/text/inspiration_quotes.txt'
content = requests.get(url)
quotes_raw = content.text
corpus = clean_text(quotes_raw)

# User-provided corpus (optional)
user_corpus = st.text_area("Or paste your own text corpus here (optional):", height=200)
if user_corpus:
    corpus = clean_text(user_corpus)

# User inputs
k_value = st.slider("Select the value of k (state window size):", 1, 5, 2)
seed_term = st.text_input(f"Enter a seed term (optional, requires {k_value} word(s)):")
term_count = st.slider("Select the number of terms to generate:", 50, 500, 100)

# Initialize session state
if "generated_text" not in st.session_state:
    st.session_state.generated_text = None
if "term_dict" not in st.session_state:
    st.session_state.term_dict = None

# Generate Text
if st.button("Generate Text"):
    if corpus:
        try:
            text_gen = MarkovText_k(corpus, k=k_value)
            generated_text = text_gen.generate(term_count=term_count, seed_term=seed_term if seed_term else None)
            
            # Store in session state
            st.session_state.generated_text = generated_text
            st.session_state.term_dict = text_gen.term_dict
            
            st.write("### Generated Text:")
            st.success(generated_text)

        except ValueError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please provide a text corpus to generate from.")

# Show previously generated text (if any)
if st.session_state.generated_text:
    st.write("### Last Generated Text:")
    st.success(st.session_state.generated_text)

    # Visualization controls
    st.markdown("## 🔍 Visualizations")
    viz_type = st.selectbox("Choose what to visualize:", ["None", "Word Cloud", "Word Frequencies", "Transition Heatmap"])
    viz_target = st.radio("Visualize based on:", ["Corpus", "Generated Text"])
    text_to_viz = corpus if viz_target == "Corpus" else st.session_state.generated_text

    # Visualize
    if viz_type == "Word Cloud":
        generate_wordcloud(text_to_viz, title=f"Word Cloud ({viz_target})")
    elif viz_type == "Word Frequencies":
        plot_word_frequencies(text_to_viz, top_n=20, title=f"Top Words in {viz_target}")
    elif viz_type == "Transition Heatmap":
        if k_value > 1:
            st.info("Transition Heatmap is only supported for k = 1.")
        else:
            plot_transition_heatmap(st.session_state.term_dict, k=k_value)
