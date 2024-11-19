import unittest
import unittest.mock
from pydantic import BaseModel

from src.util.logger import console
from src.ml.util.backoff import BackoffMixin

class DummyModel(BaseModel, BackoffMixin):

    _counter: int = 0

    def my_function(self) -> bool:

        self._counter += 1

        console.log(f"my_function called {self._counter} times")

        if self._counter < 2:
            raise Exception("my_function failed")

        return True


class TestBackoffMixin(unittest.TestCase):

    def setUp(self) -> None:
        
        self.dummy_model = DummyModel()

    def test_completion_with_backoff(self):

        result = self.dummy_model.completion_with_backoff(
            function=self.dummy_model.my_function
        )

        self.assertTrue(result)


    @unittest.mock.patch("src.ml.util.backoff.Job.execute")
    def test_completion_with_backoff_and_queue(self, mock_job_execute):

        mock_job_execute.side_effect = [Exception("Job failed"), True]

        result = self.dummy_model.completion_with_backoff_and_queue(
            function=self.dummy_model.my_function,
            job_id="my_job_id",
            rest_model_name="my_rest_model_name",
        )

        mock_job_execute.assert_called()
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
