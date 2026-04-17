"""CRUXEval validation dataset for periodic eval during training.

Loads 800 CRUXEval problems and creates TWO sub-tasks per problem:
  - cruxeval_o: given code + input → predict output  (= code_o)
  - cruxeval_i: given code + output → predict input  (= code_i)

Total: 1600 validation samples (800 output + 800 input prediction).

Data source: https://huggingface.co/datasets/cruxeval-org/cruxeval
Requires: data/cruxeval.jsonl (pre-downloaded)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Prompts matching the training prompts exactly (same as _PROMPT_CODE_O/I)
_CRUX_PROMPT_O = (
    "You are given a Python function and an input. Predict `repr(f(input))` "
    "exactly. Respond with only the predicted repr string, no prose.\n\n"
    "```python\n{code}\n```\n"
    "Input: `f({inputs})`\n"
    "Answer:"
)

_CRUX_PROMPT_I = (
    "You are given a Python function and its output. Provide one possible "
    "input that produces this output. Format: comma-separated positional "
    "args (quote strings). Respond with only the input, no prose.\n\n"
    "```python\n{code}\n```\n"
    "Output: `{output}`\n"
    "Input:"
)


class CRUXEvalDataset:
    """Map-style dataset of CRUXEval problems for verl validation.

    Each CRUXEval problem yields TWO samples:
      - even index: cruxeval_o (output prediction)
      - odd index:  cruxeval_i (input prediction)

    ``data_source`` is set to ``"cruxeval_o"`` or ``"cruxeval_i"`` so
    verl's ``process_validation_metrics`` groups them separately in
    tensorboard.
    """

    def __init__(self, data_path: str | Path | None = None) -> None:
        if data_path is None:
            # Default: data/cruxeval.jsonl relative to project root
            data_path = Path(__file__).resolve().parent.parent.parent / "data" / "cruxeval.jsonl"
        else:
            data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(
                f"CRUXEval data not found at {data_path}. "
                f"Download with: python -c \"from datasets import load_dataset; "
                f"import json; ds=load_dataset('cruxeval-org/cruxeval')['test']; "
                f"[open('{data_path}','a').write(json.dumps(x)+'\\n') for x in ds]\""
            )

        self._problems: list[dict[str, Any]] = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._problems.append(json.loads(line))

        logger.info(
            "CRUXEvalDataset: loaded %d problems → %d samples "
            "(output + input prediction)",
            len(self._problems), len(self),
        )

    def __len__(self) -> int:
        # 2 samples per problem: output prediction + input prediction
        return len(self._problems) * 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        problem_idx = index // 2
        is_input_pred = (index % 2 == 1)
        problem = self._problems[problem_idx]

        code = problem["code"]
        inp = problem["input"]
        output = problem["output"]
        task_id = problem.get("id", f"sample_{problem_idx}")

        if is_input_pred:
            # cruxeval_i: given code + output → predict input
            content = _CRUX_PROMPT_I.format(code=code, output=output)
            data_source = "cruxeval_i"
            ground_truth = {
                "task_id": task_id,
                "task_type": "code_i",
                "code": code,
                "inputs": inp,
                "gold_output": output,
            }
        else:
            # cruxeval_o: given code + input → predict output
            content = _CRUX_PROMPT_O.format(code=code, inputs=inp)
            data_source = "cruxeval_o"
            ground_truth = {
                "task_id": task_id,
                "task_type": "code_o",
                "code": code,
                "inputs": inp,
                "gold_output": output,
            }

        messages = [{"role": "user", "content": content}]
        return {
            "prompt": messages,
            "raw_prompt": messages,
            "data_source": data_source,
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "extra_info": {"index": index},
            "index": index,
            "tools_kwargs": {},
            "interaction_kwargs": {},
        }
