import os
import re
import spacy
import pandas as pd
from scipy.stats import false_discovery_control


def tag_paragraph_with_spacy(doc):
    """
    Cleans unwanted spacy tags from a document and returns lemmas.
    :param doc: doc Spacy object
    :return: list of lemmas
    """

    def custom_stopwords(token):
        return token.lemma_.lower() not in {'and'}

    clean_tokens = []

    for token in doc:
        if not (
                token.is_stop or
                token.is_punct or
                token.is_space or
                not token.is_alpha or
                len(token.text) <= 2) and custom_stopwords(token):
            clean_tokens.append(token.lemma_)

    return clean_tokens


def tag_csvfile_with_spacy(file_path: str, nlp: spacy.language.Language, column_name: str) -> pd.DataFrame:
    """
    Cleans unwanted spacy tags from a csv file and returns lemmas.
    :return: pandas.DataFrame
    """

    # check if file is empty
    try:
        assert os.path.getsize(file_path) != 0
    except AssertionError:
        print(f"File {file_path} is empty!")


    # open csv in pandas
    df = pd.read_csv(file_path, delimiter=';')

    # add clean text column for lemmas
    df[column_name +'_clean'] = None

    # iterate though rows and make a clean representation
    for index, row in df[column_name].iterrows():
        doc = nlp(row)
        clean_tokens = tag_paragraph_with_spacy(doc)
        df.at[index, 'Clean Text'] = clean_tokens
    return df


