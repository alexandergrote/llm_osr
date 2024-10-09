from pydantic import BaseModel

from src.io.data_export.base import BaseExporter


class DummyExport(BaseModel, BaseExporter):

    experiment_name: str

    def export(self, **kwargs):
        return