import pandas as pd


class Data:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def load_data(self):
        data = pd.read_csv(self.csv_file)
        texts = data["text"].tolist()
        labels = data["label"].tolist()
        return texts, labels


def load_data():
    data = Data("dataset/dataset.csv")
    return data.load_data()
