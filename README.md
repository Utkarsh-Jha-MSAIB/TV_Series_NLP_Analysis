
# 📊 Character Analysis Across Iconic TV Series

This project explores character dynamics, dialogue sentiment, and text-based similarity across four major TV series:  
**Friends**, **Game of Thrones**, **The Office**, and **The Big Bang Theory**.

Using natural language processing (NLP), we analyze patterns in scripts to uncover:
- Character similarities across shows
- Sentiment trends and emotional arcs
- Dialogue volume and use of language
- Script-based obscenity and tone

The goal is to identify whether perceived similarities between characters are supported by data, and how sentiment might correlate with broader show perception, including IMDB ratings.

---

## 🧠 Framework Overview

| Step | Description |
|------|-------------|
| **Raw Data** | Episode-level scripts sourced for all four shows |
| **Preprocessing** | Tokenization, lemmatization, stopword removal, dialogue extraction |
| **NLP Analysis** | Sentiment scoring (VADER, TextBlob, Bing Liu), obscenity detection, and TF-IDF similarity |
| **Output** | Sentiment, Dialogues Similarity, Dialogues Distribution visualization |

---

## 📘 Project Notebooks

### 🔹 [Preprocessing and Sentiment Analysis](Preprocessing%20and%20Sentiment%20Analysis.ipynb)

- Reading and cleaning script data
- Dialogue extraction and tokenization
- Sentiment analysis via:
  - Bing Liu Lexicon
  - LM Dictionary
  - TextBlob
  - Vader
  - Obscene word detection
- Export of consolidated datasets

---

### 🔹 [Similarity Indices](Similarity%20Indices.ipynb)

- Combining dialogues across shows
- TF-IDF vectorization and cosine similarity:
  - Overall inter-character similarity
  - Pairwise character comparisons
- Dataset export for visualization and interpretation

> <sub>⚠️ Note: Section links within notebooks may not work directly on GitHub. For full navigation, open notebooks in Jupyter or [nbviewer.org](https://nbviewer.org).</sub>

---

## 📊 Outputs

- Visual summaries created in **Tableau**
- Script-based similarity heatmaps and sentiment trends

---

## 🛠️ Tech Stack

- Python: `pandas`, `nltk`, `textblob`, `vaderSentiment`
- Jupyter Notebooks, .py scripts
- Tableau (for visuals)
