import math
import json


def load_faq_dataset(filepath):
    documents = []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()

        if not question or not answer:
            continue

        documents.append({
            "id": item.get("id"),
            "question": question,
            "answer": answer,
            "text": question + " " + answer
        })

    return documents


def preprocess_text(text):
    """
    Lowercasing, punctuation removal, tokenization
    """
    text = text.lower()

    cleaned_text = ""
    for char in text:
        if char.isalnum() or char == " ":
            cleaned_text += char
        else:
            cleaned_text += " "

    tokens = cleaned_text.split()
    return tokens


def preprocess_documents(documents):
    for doc in documents:
        doc["tokens"] = preprocess_text(doc["text"])
    return documents


def build_vocabulary(documents):
    vocab = set()
    for doc in documents:
        vocab.update(doc["tokens"])
    return sorted(list(vocab))


def compute_tf(tokens):
    tf = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1

    for token in tf:
        tf[token] = math.log10(tf[token] + 1)

    return tf


def compute_all_tf(documents):
    for doc in documents:
        doc["tf"] = compute_tf(doc["tokens"])
    return documents


def compute_idf(documents, vocabulary):
    N = len(documents)
    df = {term: 0 for term in vocabulary}

    for term in vocabulary:
        for doc in documents:
            if term in doc["tf"]:
                df[term] += 1

    idf = {}
    for term, freq in df.items():
        idf[term] = math.log10(N / freq) if freq > 0 else 0.0

    return idf


def compute_tfidf(tf, idf):
    return {term: tf[term] * idf.get(term, 0.0) for term in tf}


def compute_all_tfidf(documents, idf):
    for doc in documents:
        doc["tfidf"] = compute_tfidf(doc["tf"], idf)
    return documents


def vector_length(vector):
    return math.sqrt(sum(value * value for value in vector.values()))


def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1[t] * vec2[t] for t in vec1 if t in vec2)

    norm1 = vector_length(vec1)
    norm2 = vector_length(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def rank_documents(query, documents, idf):
    query_tokens = preprocess_text(query)
    query_tf = compute_tf(query_tokens)
    query_tfidf = compute_tfidf(query_tf, idf)

    scores = []
    for idx, doc in enumerate(documents):
        score = cosine_similarity(query_tfidf, doc["tfidf"])
        scores.append((score, idx))

    scores.sort(reverse=True)
    return scores


def get_top_answers(query, documents, idf, top_n=3):
    ranked_docs = rank_documents(query, documents, idf)

    results = []
    for score, idx in ranked_docs[:top_n]:
        results.append({
            "score": score,
            "id": documents[idx]["id"],
            "question": documents[idx]["question"],
            "answer": documents[idx]["answer"]
        })

    return results


if __name__ == "__main__":

    DATASET_PATH = "dataset/AI_ML_FAQ_Dataset.json"

    documents = load_faq_dataset(DATASET_PATH)
    documents = preprocess_documents(documents)

    vocabulary = build_vocabulary(documents)

    documents = compute_all_tf(documents)
    idf = compute_idf(documents, vocabulary)

    documents = compute_all_tfidf(documents, idf)

    query = "what is neural networks"

    results = get_top_answers(query, documents, idf, top_n=3)

    print("\nTop Retrieved Answers:\n")
    for res in results:
        print("Score:", round(res["score"], 4))
        print("Q:", res["question"])
        print("A:", res["answer"])
        print("-" * 50)
