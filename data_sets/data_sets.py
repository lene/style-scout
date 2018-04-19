from data_sets.data_set_base import DataSetBase

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class DataSets:

    def __init__(self, train: DataSetBase, validation: DataSetBase, test: DataSetBase) -> None:
        self.train = train
        self.validation = validation
        self.test = test
