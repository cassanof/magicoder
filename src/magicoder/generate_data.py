import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from vllm import LLM, SamplingParams

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

import magicoder

ERROR_MARGIN = 10


def make_starcoder2_prompt(code: str, lang: str) -> str:
    return f"""
<issue_start>username_0: You are exceptionally skilled at crafting high-quality programming problems and
offering precise solutions.
Please gain inspiration from the following random code snippet to create a
high-quality programming problem in {lang.title()}. Present your output in two distinct sections:
[Problem Description] and [Solution].

Code snippet for inspiration:
```{lang}
{code}
```
Guidelines for each section:
1. [Problem Description]: This should be **completely self-contained**, providing
all the contextual information one needs to understand and solve the problem.
Assume common programming knowledge, but ensure that any specific context,
variables, or code snippets pertinent to this problem are explicitly included.
2. [Solution]: Offer a comprehensive, **correct** solution that accurately
addresses the [Problem Description] you provided.<issue_comment>username_1: Sure, no problem. I will be able to help. I am  exceptionally skilled at crafting high-quality programming problems and
offering precise solutions.
I will use your provided code snippet as inspiration for the problem.

# [Problem Description]"""


@dataclass(frozen=True)
class Args:
    seed_code_start_index: int
    # `seed_code_start_index` + `max_new_data` is the last-to-end seed code index
    max_new_data: int
    continue_from: str | None = field(default=None)

    # Keep the following arguments unchanged for reproducibility
    seed: int = field(default=976)

    temperature: float = field(default=0.45)
    top_p: float = field(default=0.9)
    model: str = field(default="bigcode/starcoder2-15b")
    num_gpus: int = field(default=1)
    model_max_tokens: int = field(default=8192)
    max_new_tokens: int = field(default=2500)

    min_lines: int = field(default=1)
    max_lines: int = field(default=15)
    chunk_size: int = field(default=1000)

    dataset_name: str = field(default="bigcode/starcoderdata")
    data_dir: str | None = field(default="python")
    max_considered_data: int | None = field(default=150000)

    tag: str = field(
        default="",
        metadata={
            "help": "Custom tag as part of the output filename, not affecting the fingerprint"
        },
    )


def map_dataset(examples: dict, indices: list[int], args: Args) -> dict:
    random.seed(args.seed + indices[0])
    seed_snippets = [
        extract_seed_code(args, content) for content in examples["content"]
    ]
    return {
        "seed": seed_snippets,
        "raw_index": indices,
    }


def extract_seed_code(args: Args, document: str) -> str:
    lines = document.splitlines(keepends=True)
    start_index = random.choice(range(len(lines)))
    n_lines_to_consider = random.randint(args.min_lines, args.max_lines)
    code = "".join(lines[start_index: start_index + n_lines_to_consider])
    return code


def parse_problem_solution(response_text: str) -> tuple[str, str] | None:
    lines = response_text.splitlines(keepends=True)
    problem_start_index: int | None = None
    solution_start_index: int | None = None
    for idx, line in enumerate(lines):
        if "[problem description]" in line.lower() and problem_start_index is None:
            problem_start_index = idx
        if "[solution]" in line.lower() and solution_start_index is None:
            solution_start_index = idx
    if problem_start_index is None or solution_start_index is None:
        return None
    if problem_start_index >= solution_start_index:
        return None
    problem = "".join(lines[problem_start_index +
                      1: solution_start_index]).strip()
    solution = "".join(lines[solution_start_index + 1:]).strip()
    return problem, solution


def main():
    args, *_ = cast(
        tuple[Args, ...], HfArgumentParser(Args).parse_args_into_dataclasses()
    )
    split = (
        f"train[:{args.max_considered_data}]"
        if args.max_considered_data is not None
        else "train"
    )
    dataset: Dataset = load_dataset(
        args.dataset_name,
        data_dir=args.data_dir,
        split=split,
        num_proc=magicoder.utils.N_CORES,
    )
    random.seed(args.seed)
    # map_fn = get_map_dataset(args)
    dataset = dataset.map(
        function=map_dataset,
        fn_kwargs=dict(args=args),
        with_indices=True,
        batched=True,
        batch_size=args.chunk_size,
    )
    dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.map(lambda _, index: {"index": index}, with_indices=True)
    model = LLM(args.model, tensor_parallel_size=args.num_gpus)

    # Every run should produce the same data as long as the default params are not changed
    start_index = args.seed_code_start_index
    end_index = min(start_index + args.max_new_data, len(dataset))
    dataset = dataset.select(range(start_index, end_index))

    timestamp = magicoder.utils.timestamp()
    if args.continue_from is not None:
        assert f"{start_index}_{end_index}" in args.continue_from, "Index mismatch"
        old_path = Path(args.continue_from)
        assert old_path.exists()
        old_data = magicoder.utils.read_jsonl(old_path)
        assert len(old_data) > 0
        last_index = old_data[-1]["index"]
        n_skipped = last_index - start_index + 1
        print("Continuing from", old_path)
        f_out = old_path.open("a")
    else:
        tag = "" if args.tag == "" else f"-{args.tag}"
        path = Path(
            f"data{tag}-{start_index}_{end_index}-{timestamp}.jsonl"
        )
        assert not path.exists()
        f_out = path.open("w")
        print("Saving to", path)
        n_skipped = 0

    for index, example in enumerate(tqdm(dataset)):
        if index < n_skipped:
            continue
        assert index + start_index == example["index"]
        prompt = make_starcoder2_prompt(example["seed"], args.data_dir)
        # Make sure the generation is within the context size of the model
        max_new_tokens = min(
            args.max_new_tokens,
            args.model_max_tokens
            #  - magicoder.utils.num_tokens_from_string(prompt, args.model)
            - len(model.get_tokenizer().encode(prompt))
            # error margin (e.g., due to conversation tokens)
            - ERROR_MARGIN,
        )
        if max_new_tokens <= 0:
            continue
        response = model.generate(
            prompt,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=max_new_tokens,
                stop=["<issue_comment>", "<issue_start>"],
            )
        )
        choice = response[0].outputs[0]
        if choice.finish_reason != "stop":
            continue
        text = "# [Problem Description]\n" + choice.text
        parsing_result = parse_problem_solution(text)
        if parsing_result is None:
            continue
        problem, solution = parsing_result
        if len(problem) == 0 or len(solution) == 0:
            continue
        # In this dict seed means "seed code snippet" instead of "random seed"
        data = dict(
            raw_index=example["raw_index"],
            index=example["index"],
            seed=example["seed"],
            problem=problem,
            solution=solution,
        )

        print("[Problem Description]", problem, sep="\n", end="\n\n")
        print("[Solution]", solution, sep="\n")

        f_out.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
