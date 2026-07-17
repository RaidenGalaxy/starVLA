import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.ffn_quant.outputs import atomic_json, write_report, write_summary_csv


def _result(name="reference"):
    identity = {
        "normalized_mse": 0.0,
        "cosine_similarity": 1.0,
    }
    return {
        "name": name,
        "method": "original",
        "scope": "none",
        "bits": "original",
        "metrics": {
            "ffn": {"overall": identity},
            "hidden_states": {"final": identity},
            "logits": {
                "kl_divergence": 0.0,
                "top1_agreement": 1.0,
                "target_nll": 1.25,
                "target_perplexity": 3.49,
            },
        },
    }


class OutputTest(unittest.TestCase):
    def test_atomic_json_csv_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "metrics.json", {"value": float("inf")})
            self.assertEqual(json.loads((root / "metrics.json").read_text())["value"], "Infinity")

            results = [_result()]
            write_summary_csv(root / "summary.csv", results)
            with (root / "summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["name"], "reference")

            inventory = {
                "decoder_layers": 36,
                "linear_modules": 108,
                "target_parameters": 2_434_793_472,
                "target_fraction": 0.8,
            }
            write_report(root / "report.md", results, inventory)
            report = (root / "report.md").read_text()
            self.assertIn("FP32 CPU compute", report)
            self.assertIn("2,434,793,472", report)


if __name__ == "__main__":
    unittest.main()
