import json
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

from config import config

stemmer = PorterStemmer()


def clean_text(text: str, lower: str = True, stem: bool =False, stopwords=config.STOPWORDS) -> str:
    """Clean raw text.

    Args:
        text (str): raw text to be cleaned.
        lower (str, optional): whether to lowercase the text. Defaults to True.
        stem (bool, optional): whether to stem the text. Defaults to False.

    Returns:
        str: cleaned text.
    """
    if lower:
        text = text.lower()

    if len(stopwords):
        pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub("", text)

    text = re.sub(
        r"([!\"#$%&'()*+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # add space around punctuation

    # remove non-alphanumeric characters
    text = re.sub("[^A-Za-z0-9]+", " ", text)
    text = re.sub(" +", " ", text)  # remove extra spaces
    text = text.strip()  # remove leading and trailing spaces

    text = re.sub(r"http\S+", "", text)  # remove urls

    if stem:
        text = " ".join([stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")])

    return text


def replace_oos_labels(df: pd.DataFrame, labels: List, label_col: str, oos_label: str = "other") -> pd.DataFrame:
    """Replace out of scope (OSS) labels.

    Args:
        df (pd.DataFrame): Pandas DataFrame with data.
        labels (List): list of accepted labels.
        label_col (str): name of the dataframe column that has the labels.
        oos_label (str, optional): nume of the new label for OOS labels. Defaults to "other".

    Returns:
        pd.DataFrame: Dataframe with replaced OOS labels.
    """
    oos_tags = [item for item in df[label_col].unique() if item not in labels]
    df[label_col] = df[label_col].replace(oos_tags, oos_label)
    return df


def replace_minority_labels(df: pd.DataFrame, label_col: str, min_freq: int, new_label: str = "other") -> pd.DataFrame:
    """Replace minority labels with another label.

    Args:
        df (pd.DataFrame): Pandas DataFrame with data.
        label_col (str): name of the dataframe column that has the labels.
        min_freq (int): minimum # of tdata points a label must have.
        new_label (str, optional): name of the new label to replace minority labels. Defaults to "other".

    Returns:
        pd.DataFrame: Dataframe with replaced minority labels.
    """
    labels = Counter(df[label_col].values)
    labels_above_freq = Counter(label for label in labels.elements() if labels[label] >= min_freq)
    df[label_col] = df[label_col].apply(lambda x: x if x in labels_above_freq else new_label)
    return df


def preprocess(df: pd.DataFrame, lower: bool, stem: bool, min_freq: int) -> pd.DataFrame:
    """Preprocess the data.

    Args:
        df (pd.DataFrame): Pandas DataFrame with data.
        lower (bool): whether to lowercase the text.
        stem (bool): whether to stem the text.
        min_freq (int): minimum # of tdata points a label must have.

    Returns:
        pd.DataFrame: Dataframe with preprocessed data.
    """
    df["text"] = df.title + " " + df.description
    df.text = df.text.apply(lambda x: clean_text(x, lower=lower, stem=stem))
    df = replace_oos_labels(df=df, labels=config.ACCEPTED_TAGS, label_col="tag", oos_label="other")
    df = replace_minority_labels(df=df, label_col="tag", min_freq=min_freq, new_label="other")
    return df


class LabelEncoder(object):
    """Encode labels into unique indices.

    ```python
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y = label_encoder.encode(labels)
    ```
    """

    def __init__(self, class_to_index: Dict = {}):
        """Initialize the LabelEncoder.

        Args:
            class_to_index (Dict, optional): mapping between classes and unique indices. Defaults to {}.
        """
        self.class_to_index = class_to_index or {}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y: List):
        """Fit a list of labels to the encoder.

        Args:
            y (List): raw labels.

        Returns:
            Fitted LabgelEncoder instance.
        """
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y: List) -> np.ndarray:
        """Encode a list of raw labels.

        Args:
            y (List): raw labels.

        Returns:
            np.ndarray: encoded labels as indices.
        """
        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded

    def decode(self, y: List) -> List:
        """Decode a list of indices.

        Args:
            y (List): indices.

        Returns:
            List: labels.
        """
        classes = []
        for i, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes

    def save(self, fp: str) -> None:
        """Save class instance to JSON file.

        Args:
            fp (str): filepath to save to.
        """
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp: str):
        """Load instance of LabelEncoder from file

        Args:
            fp (str): JSON filepath to load from.

        Returns:
            LabelEncoder instance.
        """
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


def get_data_splits(X: pd.Series, y: np.ndarray, train_size: float = 0.7) -> Tuple:
    """Generate balanced data splits.

    Args:
        X (pd.Series): input features.
        y (np.ndarray): encoded labels.
        train_size (float, optional): proportion of data to use for training. Defaults to 0.7.

    Returns:
        Tuple: data splits as Numpy arrays.
    """
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test
