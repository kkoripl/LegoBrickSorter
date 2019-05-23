from enum import Enum


class LayersTrainableMode(Enum):
    LAST = 1
    LAST_CONV = 2
    All = 3
    SVM_REP = 4
