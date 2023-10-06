import re
from argparse import ArgumentParser
import pandas as pd

def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text) 
    text = re.sub(r'\[[0-9]*\]', ' ', text)  
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'\d', ' ', text)
    
    return text

def preprocess_mesh_terms(terms_text):
    terms = terms_text.strip().split(';') # Divide text into individual terms
    terms = map(lambda x: x.strip().lower(), terms) # Clean and normailze terms
    terms = map(lambda x: x.replace('*', '').replace('.', ''), terms)  # Continue cleaning
    terms = filter(lambda x: ('/' in x) and (x.index('/') < len(x)-1), terms) # Remove terms that don't have abbv.

    terms = map(lambda x: x.split('/')[0], terms)
    
    final_terms = list(terms)
    
    return final_terms

parser = ArgumentParser()
parser.add_argument("--input_csv", type=str, required=True)
parser.add_argument("--output_csv", type=str, required=True)

args = parser.parse_args()

df = pd.read_csv(args.input_csv)
print("Total no. of entries in input_csv", len(df))
df.title = df.title.map(preprocess_text)
df.mesh_terms = df.mesh_terms.map(preprocess_mesh_terms)

original_len = len(df)
df = df[df.mesh_terms.map(len) > 0]
df = df[df.title.map(len) > 0]
print(f"Dropped {original_len - len(df)} entries")
print("Total no. of entries in output_csv", len(df))


df.to_csv(args.output_csv, index=False)
