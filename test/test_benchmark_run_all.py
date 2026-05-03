import os
import tempfile
import unittest
from pathlib import Path

from benchmark.run_all import load_env_file


class BenchmarkRunAllEnvTests(unittest.TestCase):
    def test_load_env_file_sets_missing_variables_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = Path(temp_dir) / ".env"
            dotenv_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "OPENAI_API_KEY=test-openai-key",
                        "export ANTHROPIC_API_KEY='test-anthropic-key'",
                        'GEMINI_API_KEY="test-gemini-key"',
                    ]
                ),
                encoding="utf-8",
            )

            previous_openai = os.environ.pop("OPENAI_API_KEY", None)
            previous_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)
            previous_gemini = os.environ.get("GEMINI_API_KEY")
            os.environ["GEMINI_API_KEY"] = "already-set"

            try:
                loaded_keys = load_env_file(dotenv_path)
                self.assertIn("OPENAI_API_KEY", loaded_keys)
                self.assertIn("ANTHROPIC_API_KEY", loaded_keys)
                self.assertNotIn("GEMINI_API_KEY", loaded_keys)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "test-openai-key")
                self.assertEqual(os.environ["ANTHROPIC_API_KEY"], "test-anthropic-key")
                self.assertEqual(os.environ["GEMINI_API_KEY"], "already-set")
            finally:
                if previous_openai is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = previous_openai

                if previous_anthropic is None:
                    os.environ.pop("ANTHROPIC_API_KEY", None)
                else:
                    os.environ["ANTHROPIC_API_KEY"] = previous_anthropic

                if previous_gemini is None:
                    os.environ.pop("GEMINI_API_KEY", None)
                else:
                    os.environ["GEMINI_API_KEY"] = previous_gemini


if __name__ == "__main__":
    unittest.main()
