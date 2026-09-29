import numpy as np
import pandas as pd

from experiments.vla_layer_token_sensitivity.bridge_confirmatory import select_balanced_episodes
from experiments.vla_layer_token_sensitivity.qwen_oft_horizon_data import (
    normalize_bridge_actions,
    normalize_bridge_state,
    persistence_baseline,
)
from experiments.vla_layer_token_sensitivity.qwen_oft_probe import _condition_instruction
from starVLA.model.framework.share_tools import add_discretized_state_to_instruction


def _statistics() -> dict:
    return {
        "state": {"q01": [0.0] * 8, "q99": [1.0] * 6 + [0.0, 1.0]},
        "action": {"q01": [0.0] * 7, "q99": [1.0] * 7},
    }


def test_bridge_state_inserts_pad_and_uses_q99_binary_rules() -> None:
    raw = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 0.4], dtype=np.float32)
    normalized = normalize_bridge_state(raw, _statistics())

    np.testing.assert_allclose(normalized, [-1.0, -0.5, 0.0, 0.5, 1.0, 2.2, 0.0, 0.0])
    assert persistence_baseline(normalized, 16).shape == (16, 7)
    assert np.all(persistence_baseline(normalized, 16)[:, 6] == 0.0)


def test_bridge_action_normalization_preserves_horizon_and_binarizes_gripper() -> None:
    raw = np.full((16, 7), 0.5, dtype=np.float32)
    raw[0, 6] = 0.49
    raw[1, 6] = 0.51
    normalized = normalize_bridge_actions(raw, _statistics())

    assert normalized.shape == (16, 7)
    np.testing.assert_allclose(normalized[:, :6], 0.0)
    assert normalized[0, 6] == 0.0
    assert normalized[1, 6] == 1.0


def test_condition_instruction_matches_native_qwen_oft_helper() -> None:
    class DummyModel:
        add_discretized_state_to_instruction = staticmethod(add_discretized_state_to_instruction)

    state = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    actual = _condition_instruction(DummyModel(), "move cup", state)
    expected = add_discretized_state_to_instruction(["move cup"], [state.reshape(1, -1)])[0]

    assert actual == expected
    assert "[STATE]" in actual and actual.endswith("[ACTION]")


def test_balanced_selector_excludes_discovery_and_requires_full_horizon() -> None:
    episodes = pd.DataFrame(
        {
            "episode_index": list(range(12)),
            "tasks": [[f"task {index % 3}"] for index in range(12)],
            "length": [23] * 11 + [19],
        }
    )
    selected = select_balanced_episodes(
        episodes,
        count=6,
        seed=20260917,
        excluded={0, 1},
        minimum_length=20,
    )

    assert len(selected) == 6
    assert not set(selected["episode_index"]) & {0, 1, 11}
    assert selected["instruction"].value_counts().max() - selected["instruction"].value_counts().min() <= 1
