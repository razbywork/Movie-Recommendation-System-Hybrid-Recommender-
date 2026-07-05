import pandas as pd


def split_train_test_by_user(ratings, test_ratio=0.2):
    train_list = []
    test_list = []

    for user_id, group in ratings.groupby("userId"):
        if len(group) < 5:
            train_list.append(group)
            continue

        n_test = max(1, int(len(group) * test_ratio))

        test = group.sample(n_test, random_state=42)
        train = group.drop(test.index)

        train_list.append(train)
        test_list.append(test)

    return pd.concat(train_list), pd.concat(test_list)


def precision_at_k(recommended, relevant, k=10):
    if len(recommended) == 0:
        return 0.0

    recommended = recommended[:k]

    hits = len(set(recommended) & set(relevant))

    return hits / k


def recall_at_k(recommended, relevant, k=10):
    if len(relevant) == 0:
        return 0.0

    recommended = recommended[:k]

    hits = len(set(recommended) & set(relevant))

    return hits / len(relevant)