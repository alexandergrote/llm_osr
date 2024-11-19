import json
import pandas as pd

from requests import Response  # type: ignore

from src.util.constants import DatasetColumn

data_train = pd.DataFrame({
    DatasetColumn.FEATURES: ["Ich heiße Alex", "Auf Wiedersehen!", "Hallo", "Wo kann ich Kekse kaufen?", "Die Apfelsaftschorle finde ich wo?", "Das Essen ist sehr lecker", "München ist eine große Stadt."],
    DatasetColumn.LABEL: ['Greeting', 'Farewell', "Greeting", "Food", "Food", "Food", "City"]
})

data_valid = pd.DataFrame({
    DatasetColumn.FEATURES: ["Ich bin ein Mensch", "Ich war noch nie in Berlin"],
    DatasetColumn.LABEL: ['Mensch', 'City']
})

# Helper function to create mock responses
def mock_response(status_code=200, json_data=None, text_data=None):
    response = Response()
    response.status_code = status_code
    if json_data:
        response._content = json.dumps(json_data).encode('utf-8')
    elif text_data:
        response._content = text_data.encode('utf-8')
    return response
