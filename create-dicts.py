from argparse import ArgumentParser
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import pickle

parser = ArgumentParser()
parser.add_argument("--input_csv", type=str, required=True)
parser.add_argument("--terms_csv", type=str, required=True)
parser.add_argument("--frequency_threshold", type=str, default=100)
parser.add_argument("--vectorizer_file", type=str, required=True)
parser.add_argument("--vocab_size", type=int, default=10000)

args = parser.parse_args()

df = pd.read_csv(args.input_csv)
print("Total no. of entries in input_csv", len(df))
df.mesh_terms = df.mesh_terms.map(eval)

print("Processing Terms:")
term_counts = defaultdict(int)
for i in tqdm(range(len(df))):
    for term in df.iloc[i].mesh_terms:
        term_counts[term] += 1
    
valid_terms = list(filter(lambda x: term_counts[x] >= args.frequency_threshold, term_counts))
print("Total Unique Terms:", len(term_counts))
print("Valid Unique Terms:", len(valid_terms))

df_terms = pd.DataFrame(zip(range(len(valid_terms)), valid_terms), columns=["id", "term"])
df_terms.to_csv(args.terms_csv, index=False)

vectorizer = CountVectorizer(max_features=args.vocab_size, stop_words="english", binary=True)
vectorizer.fit(df.title)

with open(args.vectorizer_file, 'wb') as f:
    pickle.dump(vectorizer, f)



