import pandas as pd

from typing import Optional, Dict
from pydantic import BaseModel

from src.util.constants import UnknownClassLabel, ErrorValues


class ClassMapper(BaseModel):

    mapping: Optional[Dict[str, int]] = None

    def fit(self, data: pd.Series):

        unique_values = data.unique()
        unique_values.sort()

        mapping = {
            UnknownClassLabel.UNKNOWN_STR.value: UnknownClassLabel.UNKNOWN_NUM.value,
            ErrorValues.PARSING_STR.value: ErrorValues.PARSING_NUM.value
        }

        for idx, value in enumerate(unique_values):

            mapping[value] = idx

        self.mapping = mapping

    def transform(self, data: pd.Series):

        assert self.mapping is not None, "Fit has not been called"

        # work on copy
        data_copy = data.copy()

        return data_copy.replace(self.mapping)
    
    def fit_transform(self, data: pd.Series):

        self.fit(data)

        return self.transform(data=data)


if __name__ == '__main__':

    data = pd.Series(['b', 'a', 'c'])

    mapper = ClassMapper()

    result = mapper.fit_transform(data)

    print(result)