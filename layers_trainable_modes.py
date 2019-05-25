from enum import Enum


class LayersTrainableMode(Enum):
    ONLY_CLASSIF = 1
    FROM_LAST_CONV = 2
    ALL = 3
    SVM_REP = 4