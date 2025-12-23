import os
import re
import spacy
import pandas as pd
from scipy.stats import false_discovery_control


def tag_paragraph_with_spacy(doc) -> list:
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
            clean_tokens.append(token.lemma_.lower())

    return clean_tokens


def tag_df_with_spacy(df, nlp: spacy.language.Language, column_names: list) -> pd.DataFrame:
    """
    Cleans unwanted spacy tags from a pkl file and returns lemmas. Operates chosen on columns.
    :param df: pandas dataframe or path to pkl file
    :param nlp: spacy language object
    :param column_names: list of column names to clean
    :return: pandas dataframe
    """

    # in case you pass a file path
    if isinstance(df, str):
        # check if file is empty
        try:
            assert os.path.getsize(df) != 0
            df = pd.read_pickle(df)
        except AssertionError:
            print(f"File {df} is empty!")

    for column in column_names:
        print(f'Cleaning column "{column}"...')
        # add clean text column for lemmas
        new_name = column +'_clean'
        df[new_name] = None

        # iterate though rows and make a clean representation
        for index, text in df[column].items():
            doc = nlp(text)
            clean_tokens = tag_paragraph_with_spacy(doc)
            df.at[index, new_name] = [clean_tokens]
    return df


