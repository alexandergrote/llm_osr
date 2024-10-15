import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

from src.ml.util.job_queue import Job, JobQueue, JobStatus, RequestFunction

TMP_DIR = TemporaryDirectory()
PATCH_JOB_DIR = Path(TMP_DIR.name)

class TestJobQueue(unittest.TestCase):

    def setUp(self):

        self.jobs = [
            Job(job_id='1', request_dict=dict(url='https://httpbin.org/post', data={'key': 'value'}), request_function=RequestFunction.post),
            Job(job_id='2', request_dict=dict(url='https://httpbin.org/post2', data={'key': 'value'}), request_function=RequestFunction.post),
            Job(job_id='3', request_dict=dict(url='https://httpbin.org/post3', data={'key': 'value'}), request_function=RequestFunction.post),
        ]

        PATCH_JOB_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):

        TMP_DIR.cleanup()


    @patch("src.ml.util.job_queue.get_job_dir")
    def test_execution(self, mock_job_dir):

        mock_job_dir.return_value = PATCH_JOB_DIR

        queue = JobQueue(jobs=iter(self.jobs))
        queue.run_failed_jobs()

        for job in queue.jobs:
            self.assertIn(job.status, [JobStatus.success, JobStatus.failed])

if __name__ == "__main__":
    unittest.main()
