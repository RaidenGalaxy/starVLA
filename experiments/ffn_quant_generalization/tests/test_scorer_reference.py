from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Optional

from experiments.ffn_quant_generalization.config import ROOT
from experiments.ffn_quant_generalization.scoring import score_prediction


def test_chartqa_matches_existing_public_vlmevalkit_reference():
    # Load only the pure scorer, avoiding unrelated API judges and optional packages.
    path = ROOT / "third_party/VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py"
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "relaxed_correctness")
    namespace = {"Optional": Optional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    reference = namespace["relaxed_correctness"]
    for target, prediction in [
        ("0", "0"),
        ("0", "0.0"),
        ("100", "105"),
        ("100", "106"),
        ("50%", "0.5"),
        ("-20", "-21"),
        ("Blue", "blue"),
        ("red blue", "red  blue"),
    ]:
        sample = SimpleNamespace(dataset="ChartQA_TEST", target=target)
        assert score_prediction(sample, prediction) == float(reference(target, prediction))
