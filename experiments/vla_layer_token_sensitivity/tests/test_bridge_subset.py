import pandas as pd
import pyarrow as pa

from experiments.vla_layer_token_sensitivity.bridge_subset import (
    _task_mapping,
    fraction_frame_indices,
    select_episodes,
)


def test_fraction_frame_indices_are_distinct_and_deterministic() -> None:
    assert fraction_frame_indices(11, [0.2, 0.4, 0.6, 0.8]) == [2, 4, 6, 8]


def test_select_episodes_prefers_distinct_tasks() -> None:
    episodes = pd.DataFrame(
        {
            "episode_index": list(range(8)),
            "tasks": [[f"task {index // 2}"] for index in range(8)],
            "length": [20] * 8,
        }
    )

    selected = select_episodes(episodes, count=4, seed=42)

    assert len(selected) == 4
    assert selected["instruction"].nunique() == 4
    assert selected["episode_index"].nunique() == 4


def test_task_mapping_uses_arrow_columns_independent_of_pandas_index_metadata() -> None:
    table = pa.table(
        {"task_index": [3, 7, 9, 11], "task": ["pick up pot", "open drawer", "", None]}
    )

    assert _task_mapping(table) == {3: "pick up pot", 7: "open drawer"}
