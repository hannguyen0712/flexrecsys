# rank only movies the user has NOT rated in their history (train, or train+val at test time).
# users with no relevant held-out movies are skipped for Recall/NDCG.

import numpy as np
import pandas as pd

RELEVANT_RATING = 4.0


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def recall_at_k(ranked, relevant, k):
    hits = sum(item in relevant for item in ranked[:k])
    return hits / min(len(relevant), k)


def ndcg_at_k(ranked, relevant, k):
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal


def per_user_scores(recommendations, heldout, ks=(10, 20)):
    # recommendations: {userId: ranked movieIds}, already excluding movies seen in history
    # users with no relevant held-out movies are skipped
    relevant = (heldout[heldout["rating"] >= RELEVANT_RATING]
                .groupby("userId")["movieId"].apply(set))
    rows = []
    for user, items in relevant.items():
        ranked = list(recommendations.get(user, []))
        row = {"userId": user}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked, items, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, items, k)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_ranking(recommendations, heldout, ks=(10, 20)):
    scores = per_user_scores(recommendations, heldout, ks)
    out = scores.drop(columns="userId").mean().to_dict()
    out["n_users"] = len(scores)
    return out