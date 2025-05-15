from pathlib import Path
import pandas as pd
import re    
import nltk
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords


def load_data(path):
    path = Path(path)  # Ensures compatibility on all OS
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_csv(path)


def analyze_character_dialogues(df, show_overall=True, season=None, episode=None):

    # Clean and prepare
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Normalize season value (e.g. 1 → '01')
    if season:
        season = str(season).zfill(2)
        df['season'] = df['season'].astype(str).str.zfill(2)
    if episode:
        season = str(season).zfill(2)
        df['season'] = df['season'].astype(str).str.zfill(2)
        episode = str(episode).zfill(2)
        df['episode'] = df['episode'].astype(str).str.zfill(2)
        season_episode = season + 'x' + episode
        
    df['character'] = df['character'].astype(str).str.strip().str.title()
    df['season_episode'] = df['season'].astype(str) + 'x' + df['episode'].astype(str)
    df['dialogue_length'] = df['dialogue'].astype(str).apply(len)

    results = {}

    def top_10_by_sum(df_subset):
        total = df_subset['dialogue_length'].sum()
        top10 = df_subset.groupby('character')['dialogue_length'] \
                         .sum().nlargest(10).reset_index().rename(columns={'dialogue_length': 'total_characters'})
        top10['percent_of_total'] = (top10['total_characters'] / total * 100).round(2)
        return top10

    def top_10_by_count(df_subset):
        total = len(df_subset)
        top10 = df_subset['character'].value_counts().head(10).reset_index()
        top10.columns = ['character', 'dialogue_lines']
        top10['percent_of_total'] = (top10['dialogue_lines'] / total * 100).round(2)
        return top10


    # Episode-specific (only if provided)
    if episode:
        df_ep = df[df['season_episode'] == season_episode]
        results[f'Top 10 by Dialogue Length - Episode {season_episode}'] = top_10_by_sum(df_ep)
        results[f'Top 10 by Dialogue Rows - Episode {season_episode}'] = top_10_by_count(df_ep)

    # Season-specific (even if episode is also passed)
    if season:
        df_season = df[df['season'] == season]
        results[f'Top 10 by Dialogue Length - Season {season}'] = top_10_by_sum(df_season)
        results[f'Top 10 by Dialogue Rows - Season {season}'] = top_10_by_count(df_season)

    # Overall (only if nothing else or explicitly allowed)
    if show_overall:
        results['Top 10 by Dialogue Length - Overall'] = top_10_by_sum(df)
        results['Top 10 by Dialogue Rows - Overall'] = top_10_by_count(df)

    return results


def preprocess_dialogue(dialogue):

    dialogue = str(dialogue) if pd.notnull(dialogue) else ''
    
    #Tokenize
    tokens1 = word_tokenize(dialogue)

    #Changing all tokens into lower case 
    words1 = [w.lower() for w in tokens1]

    #Only keep text words, no numbers 
    words2 = [w for w in words1 if w.isalpha()]
    
    #Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in words2 if word not in stop_words]
    
    #Stemming and Lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    #stemmed = [stemmer.stem(word) for word in tokens]
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    
    #Reconstruct the message from lemmatized tokens
    return ' '.join(lemmatized)


def write_data(df, filename, file_format='csv'):
    """
    Save a DataFrame to the data/processed/ folder.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): Output file name (e.g., 'cleaned_data.csv').
        file_format (str): Format to save ('csv' or 'xlsx').

    Returns:
        str: Full path to the saved file.
    """
    processed_path = Path('data/processed')
    processed_path.mkdir(parents=True, exist_ok=True)  # Ensure folder exists

    full_path = processed_path / filename

    if file_format == 'csv':
        df.to_csv(full_path, index=False)
    elif file_format == 'xlsx':
        df.to_excel(full_path, index=False)
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'xlsx'.")

    print(f"Data written to: {full_path.resolve()}")
    return str(full_path)

