from argparse import ArgumentParser
import pandas as pd
import pickle
from tqdm import tqdm
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, recall_score

parser = ArgumentParser()
parser.add_argument("--train_csv", type=str, required=True)
parser.add_argument("--test_csv", type=str, required=True)
parser.add_argument("--terms_csv", type=str, required=True)
parser.add_argument("--results_csv", type=str, required=True)
parser.add_argument("--vectorizer_file", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--force_train", type=bool, default=False)

args = parser.parse_args()

df_train = pd.read_csv(args.train_csv)
df_test = pd.read_csv(args.test_csv)

df_train.mesh_terms = df_train.mesh_terms.map(eval)
df_test.mesh_terms = df_test.mesh_terms.map(eval)

with open(args.vectorizer_file, 'rb') as f:
    vectorizer = pickle.load(f)

X_train = vectorizer.transform(df_train.title)
X_test = vectorizer.transform(df_test.title) 

print("Train Size:", X_train.shape)
print("Test Size:", X_test.shape)

df_terms = pd.read_csv(args.terms_csv)
print("Valid Term Count:", len(df_terms))

int2term = df_terms["term"].tolist()
term2int = {x:i for (i, x) in enumerate(int2term)}

df_train.mesh_terms = df_train.mesh_terms.map(lambda x: list(map(lambda y: term2int.get(y, -1), x)))
df_test.mesh_terms = df_test.mesh_terms.map(lambda x: list(map(lambda y: term2int.get(y, -1), x)))

model_dir = Path(args.model_dir)
model_dir.mkdir(exist_ok=True)

results = []
for i in tqdm(range(len(int2term))):
    y_train = ([(i in df_train.iloc[j].mesh_terms) for j in range(len(df_train))])
    y_test = [(i in df_test.iloc[j].mesh_terms) for j in range(len(df_test))]

    if sum(y_train) == 0:
        print("Skipping", int2term[i])
        continue

    if (not args.force_train) and (model_dir / f"{i}.pkl").is_file():
        with open(model_dir / f"{i}.pkl", 'rb') as f:
            model = pickle.load(f)
    else:
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        with open(model_dir / f"{i}.pkl", 'wb') as f:
            pickle.dump(model, f)
    
    y_pred = model.predict(X_test)
    results.append((
        int2term[i], 
        accuracy_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
    ))

df_results = pd.DataFrame(results, columns=["term", "accuracy", "recall", "f1_score"])
df_results.to_csv(args.results_csv, index=False)
print(f"Saved results for {len(df_results)} terms")
print()
