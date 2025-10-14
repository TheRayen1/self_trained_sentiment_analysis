# sentiment_tool.py
import pandas as pd, joblib, os, re, string, json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

TRAIN_CSV = r"C:\Users\rayen\Desktop\kagggle\train.csv"
TEST_CSV  = r"C:\Users\rayen\Desktop\kagggle\test.csv"
MODEL_PKL = "tweet_sentiment.joblib"
THRESH_PKL= "neutral_threshold.json"

# ---------- helpers ----------
def clean(text: str) -> str:
    text = re.sub(r"http\S+|@\w+|#\w+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.lower().strip()

def get_label_map():
    return {"positive":"positive", "pos":"positive", "+":"positive",
            "negative":"negative", "neg":"negative", "-":"negative"}

# ---------- build / load ----------
def build_model():
    # 1. read only the columns we care about (Latin-1 to avoid UTF-8 errors)
    train_df = pd.read_csv(TRAIN_CSV, usecols=["textID","text","sentiment"],
                           encoding="latin1")
    train_df = train_df.dropna(subset=["text","sentiment"])
    train_df["text"] = train_df["text"].astype(str).apply(clean)
    label_map = get_label_map()
    train_df = train_df[train_df.sentiment.str.lower().isin(label_map)]
    train_df["label"] = train_df.sentiment.str.lower().map(label_map)

    # 2. split train / val (for threshold tuning)
    X_train, X_val, y_train, y_val = train_test_split(
        train_df.text, train_df.label, test_size=0.15, random_state=42,
        stratify=train_df.label)

    # 3. train pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english",
                                  ngram_range=(1,2), min_df=2, max_df=0.9)),
        ("clf",  LogisticRegression(max_iter=1000, C=2.0))
    ])
    model.fit(X_train, y_train)
    print("Validation accuracy (pos/neg):", model.score(X_val, y_val))

    # 4. pick best neutral threshold on validation set
    best_thresh, best_f1 = 0.0, 0.0
    for t in [i/100 for i in range(45, 76, 5)]:   # 0.45 → 0.75
        preds = thresh_predict(model, X_val, t)
        rep = classification_report(y_val, preds, output_dict=True)
        f1_neg, f1_pos = rep["negative"]["f1-score"], rep["positive"]["f1-score"]
        macro_f1 = (f1_neg + f1_pos) / 2
        if macro_f1 > best_f1:
            best_f1, best_thresh = macro_f1, t
    print("Chosen neutral threshold:", best_thresh, "macro-F1", round(best_f1, 3))
    with open(THRESH_PKL, "w") as f:
        json.dump({"threshold": best_thresh}, f)

    # 5. final re-train on FULL training data
    model.fit(train_df.text, train_df.label)
    joblib.dump(model, MODEL_PKL)
    return model, best_thresh

def load_model():
    if os.path.exists(MODEL_PKL) and os.path.exists(THRESH_PKL):
        model = joblib.load(MODEL_PKL)
        thresh = json.load(open(THRESH_PKL))["threshold"]
        return model, thresh
    return build_model()

# ---------- threshold-based predict ----------
def thresh_predict(model, X, threshold):
    """Return list of labels according to probability threshold."""
    probs = model.predict_proba(X)
    labs  = []
    for row in probs:
        best_prob = row.max()
        best_lab  = model.classes_[row.argmax()]
        labs.append("neutral" if best_prob < threshold else best_lab)
    return labs

# ---------- public API ----------
_model, _thresh = load_model()

def predict_sentiment(text: str):
    cleaned = clean(text)
    prob = _model.predict_proba([cleaned])[0]
    best_prob = prob.max()
    best_lab  = _model.classes_[prob.argmax()]
    label = "neutral" if best_prob < _thresh else best_lab
    return {"label": label, "confidence": float(best_prob)}

# ---------- evaluate on official test.csv ----------
def evaluate_test():
    test_df = pd.read_csv(TEST_CSV, usecols=["textID","text","sentiment"],
                          encoding="latin1")
    test_df = test_df.dropna(subset=["text","sentiment"])
    test_df["text"] = test_df["text"].astype(str).apply(clean)
    label_map = get_label_map()
    test_df = test_df[test_df.sentiment.str.lower().isin(label_map)]
    test_df["correct"] = test_df.sentiment.str.lower().map(label_map)

    preds = thresh_predict(_model, test_df.text, _thresh)
    print("\n=====  FINAL TEST METRICS  =====")
    print(classification_report(test_df.correct, preds, digits=3))
    print("Accuracy:", accuracy_score(test_df.correct, preds))

# ---------- save predictions ----------
def save_predictions():
    """Create predictions.csv with columns:
       textID, text, correct_sentiment, predicted_sentiment
    """
    test_df = pd.read_csv(TEST_CSV, usecols=["textID","text","sentiment"],
                          encoding="latin1")
    test_df = test_df.dropna(subset=["text","sentiment"])
    test_df["text"] = test_df["text"].astype(str).apply(clean)
    label_map = get_label_map()
    test_df = test_df[test_df.sentiment.str.lower().isin(label_map)]
    test_df["correct_sentiment"] = test_df.sentiment.str.lower().map(label_map)

    preds = thresh_predict(_model, test_df.text, _thresh)

    out = pd.DataFrame({
        "textID": test_df.textID,
        "text":   test_df.text,
        "correct_sentiment": test_df.correct_sentiment,
        "predicted_sentiment": preds
    })
    out.to_csv("predictions.csv", index=False)
    print("\nSaved → predictions.csv  (head):")
    print(out.head())

# ---------- run everything ----------
if __name__ == "__main__":
    evaluate_test()
    save_predictions()
    print("\n----- interactive -----\n")
    while True:
        try:
            txt = input("Text > ").strip()
            if not txt: break
            print(predict_sentiment(txt))
        except KeyboardInterrupt:
            break