import pandas as pd
import spacy
from source import tokenizator as tk

df = pd.read_pickle('dfs/preprocessed-df.pkl')
df.head()

nlp = spacy.load('en_core_web_sm')

df_clean = tk.tag_df_with_spacy(df=df, nlp=nlp, column_names=['professor', 'student', 'prompt'])
print(df_clean.info())
