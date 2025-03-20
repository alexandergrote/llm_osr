from enum import Enum

class OutlierValue(Enum):

    OUTLIER = 'outlier'
    INLIER = 'inlier'

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))