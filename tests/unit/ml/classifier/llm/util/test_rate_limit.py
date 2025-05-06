import unittest

from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

from src.ml.classifier.llm.util.rate_limit import DatabaseManager, RateLimit, RateLimitManager

TMP_DIR = TemporaryDirectory()
RL_DIR = Path(TMP_DIR.name) / 'rate_limits'
RL_DB_PATH = RL_DIR / 'rate_limits.db'


class TestRLM(unittest.TestCase):

    def setUp(self):

        RL_DIR.mkdir(parents=True, exist_ok=True)

        self.rlm = RateLimitManager(
            name="test",
            db_manager=DatabaseManager(path=RL_DB_PATH),
            rate_limits= {
                "num_req_min": RateLimit(limit=5, agg_level="%Y-%m-%d %H:%M", increment_level="frequency", action="exit"),
                "num_token_min": RateLimit(limit=1000, agg_level="%Y-%m-%d %H:%M", increment_level="token"),
            }
        )

        # dummy query
        self.query = "hello world is wonderful"

    @patch("src.ml.classifier.llm.util.rate_limit.RateLimit._get_timestamp")
    def test_check_execution(self, mock_get_timestamp):

        mock_get_timestamp.return_value = "a"

        rl = self.rlm.rate_limits['num_req_min']
        rl.limit = 1
        self.rlm.rate_limits['num_req_min'] = rl

        # first check should be allowed
        num_tokens = 5
        
        result = self.rlm.check_execution(num_tokens)  # prior to execution, all rate limits will be updated, including the ones that refer to the request frequency per time unit
        self.assertIsNone(result)
        self.rlm.update(tokens=num_tokens, tokens_only=True)  # after execution, only tokens will be updated

        # second check should be blocked
        with self.assertRaises(SystemExit):
            self.rlm.check_execution(num_tokens)
 
    @patch("src.ml.classifier.llm.util.rate_limit.RateLimit._get_timestamp")
    def test_update(self, mock_get_timestamp):

        mock_get_timestamp.return_value = "a"

        # initial records
        num_req_min: RateLimit = self.rlm.rate_limits["num_req_min"]
        num_token_min: RateLimit = self.rlm.rate_limits["num_token_min"]
        self.assertTrue(num_req_min.records["a"] == 0)
        self.assertTrue(num_token_min.records["a"] == 0)

        # update
        num_tokens = 5
        self.rlm.update(tokens=num_tokens)
        num_req_min: RateLimit = self.rlm.rate_limits["num_req_min"]
        num_token_min: RateLimit = self.rlm.rate_limits["num_token_min"]
        self.assertTrue(num_req_min.records["a"] == 1)
        self.assertTrue(num_token_min.records["a"] == num_tokens)

        # new minute has begun
        mock_get_timestamp.return_value = "b"
        num_tokens = 6
        self.rlm.update(tokens=num_tokens)
        num_req_min: RateLimit = self.rlm.rate_limits["num_req_min"]
        num_token_min: RateLimit = self.rlm.rate_limits["num_token_min"]
        self.assertTrue(num_req_min.records["b"] == 1)
        self.assertTrue(num_token_min.records["b"] == num_tokens)

        # reset to 0 in default dict since entry for default dict has been cleared
        self.assertTrue(num_token_min.records["a"] == 0)  

    def test_load(self):
        self.assertIsInstance(self.rlm.load(), RateLimitManager)

    @patch("src.ml.classifier.llm.util.rate_limit.get_rate_limit_db_path")
    def test_save_and_load(self, mock_rate_limit_dir):

        mock_rate_limit_dir.return_value = RL_DB_PATH
        num_tokens = 5

        saved_rlm = RateLimitManager(
            name="test",
            db_manager=DatabaseManager(path=RL_DB_PATH),
            rate_limits= {
                "num_req_min": RateLimit(limit=5, agg_level="%Y-%m-%d %H:%M", increment_level="frequency"),
                "num_token_min": RateLimit(limit=1000, agg_level="%Y-%m-%d %H:%M", increment_level="token"),
            }
        )

        saved_rlm.update(tokens=num_tokens)
        saved_rlm.save()

        loaded_rlm = RateLimitManager(name="test", db_manager=DatabaseManager(path=RL_DB_PATH)).load()
        self.assertEqual(saved_rlm.dict(exclude=["db_manager"]), loaded_rlm.dict(exclude=["db_manager"]))

    def tearDown(self):

        # unlink all files in job and log dir
        # cleanup alone does not do the job 
        # todo: find better solution
        for file in RL_DIR.glob("*.db"):
            file.unlink()

        TMP_DIR.cleanup()

        
        

if __name__ == '__main__':
    unittest.main()

    
    