import pandas as pd
import sqlite3
import csv


csv_path = './database.sqlite'
csv_path1 = './data/def_data.csv'


con = sqlite3.connect(csv_path)
df = pd.read_sql_query(""" SELECT * FROM Reviews""", con)

df_clean = df[df["Score"] != 3]

def assign_sentiment(score):
    return "positive" if score >= 4 else "negative"

df_clean["Sentiment"] = df_clean["Score"].apply(assign_sentiment)

min_count = df_clean["Sentiment"].value_counts().min()
df_positive = df_clean[df_clean["Sentiment"] == "positive"].sample(min_count, random_state=42)
df_negative = df_clean[df_clean["Sentiment"] == "negative"].sample(min_count, random_state=42)
df_balanced = pd.concat([df_positive, df_negative]).sample(frac=1, random_state=42)

df_end = df_balanced[["Text", "Sentiment"]].dropna()

df_end = df_end.reset_index(drop=True)

df_end.to_csv(csv_path1, index=False, sep="¬")