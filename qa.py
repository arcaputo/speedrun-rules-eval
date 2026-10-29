"""QA for the modded-nanogpt speedrun: the candidate keeps the speedrun's rules, so its time counts.

Checks, on the subject (subjects/<name>/ with train_gpt.py) and on what the candidate changed since its
experiment started ($HARNESS_CHANGES_FILE; absent on the base node, where only the tree is checked):

  - train_gpt.py is still the entry point (the benchmark runs `torchrun ... train_gpt.py`).
  - val_tokens is 10485760 wherever it is set.
  - The data stays the eval's: train and val files are read from $DATA_PATH/data/fineweb10B/ with the
    speedrun's names, and nothing under data/ changed.
  - No torch.compile / inductor / dynamo tuning flags were added (the speedrun bans them).
  - The run still prints the speedrun's final line, `step:N/N val_loss:X train_time:Yms`.

Changes to how the model computes its validation loss (e.g. masking infeasible tokens) are the
benchmark's business, not a rule breach. Runs in seconds on a CPU.

Local check: HARNESS_SUBJECTS_DIR=/path/to/subjects [HARNESS_CHANGES_FILE=changes.diff] python qa.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

RESULT_FILE = os.environ.pop("HARNESS_RESULT_FILE", None)  # before anything reads the code under test


def report(result: dict) -> None:
    """Print the result, and write it to the file Kupanga reads it from (docs/evals.md)."""
    line = json.dumps(result)
    print(line, flush=True)
    if RESULT_FILE:
        with open(RESULT_FILE, "w") as f:
            f.write(line + "\n")


VAL_TOKENS = 10_485_760
VAL_TOKENS_SET = re.compile(r"\bval_tokens\s*(?::\s*int\s*)?=\s*([^#\n]+)")
DATA_FILES = ("data/fineweb10B/fineweb_train_*.bin", "data/fineweb10B/fineweb_val_*.bin")
FINAL_LINE = re.compile(r"val_loss:\{[^}]*\}\s*train_time:")
# Compiler tuning the speedrun bans: inductor/dynamo config, compile modes and options, inductor env vars.
TUNING = [
    (re.compile(r"\b(?:torch\.)?_inductor\.config\b|\binductor_config\b|\bfrom torch\._inductor import config\b"),
     "sets torch._inductor config"),
    (re.compile(r"\b(?:torch\.)?_dynamo\.config\.(?!cache_size_limit|recompile_limit)\w+\s*="), "sets torch._dynamo config"),
    (re.compile(r"\btorch\.compile\([^)]*\b(?:mode|options|backend)\s*="), "passes mode/options/backend to torch.compile"),
    (re.compile(r"\bTORCHINDUCTOR_\w+|\bmax_autotune\b|\bcoordinate_descent_tuning\b"), "sets inductor tuning"),
]


def find_subject() -> Path:
    root = Path(os.environ.get("HARNESS_SUBJECTS_DIR") or Path(__file__).resolve().parent.parent.parent / "subjects")
    found = sorted({p.parent for p in root.glob("*/train_gpt.py")} | {p.parent.parent for p in root.glob("*/*/train_gpt.py")})
    if not found:
        raise SystemExit(f"no modded-nanogpt checkout (train_gpt.py) under {root}/<subject>/ or {root}/<subject>/<record>/")
    return found[0]


def record_roots(subject: Path) -> list[str]:
    """The folders train_gpt.py runs from: the subject itself (""), or one folder per record when the subject
    holds several (subject/<record>/train_gpt.py)."""
    if (subject / "train_gpt.py").is_file():
        return [""]
    return sorted(p.parent.name for p in subject.glob("*/train_gpt.py"))


def sources(subject: Path) -> dict[str, str]:
    return {str(p.relative_to(subject)): p.read_text(errors="replace") for p in sorted(subject.rglob("*.py"))
            if "records" not in p.relative_to(subject).parts and p.name != "train_gpt_medium.py"}


def added_lines(diff: str) -> list[tuple[str, str]]:
    """(file, line) for every line the diff adds, file relative to its subject."""
    out, file = [], ""
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            file = path.split("/", 3)[-1] if path.startswith("b/subjects/") else path
        elif line.startswith("+") and not line.startswith("+++"):
            out.append((file, line[1:]))
    return out


def changed_files(diff: str) -> set[str]:
    files = set()
    for line in diff.splitlines():
        if line.startswith(("--- a/", "+++ b/")):
            files.add(line[6:].strip().split("/", 2)[-1])  # drop subjects/<name>/
    return files


def code_only(line: str) -> str:
    """A line without its comment (a commented-out flag isn't set)."""
    return line.split("#", 1)[0]


def main() -> None:
    subject = find_subject()
    src = sources(subject)
    changes_file = os.environ.get("HARNESS_CHANGES_FILE")
    diff = Path(changes_file).read_text(errors="replace") if changes_file and Path(changes_file).is_file() else ""
    problems: list[str] = []
    checks = 0

    roots = record_roots(subject)
    files = changed_files(diff)

    checks += 1
    for r in roots:
        if f"{r}/train_gpt.py".lstrip("/") not in src:
            problems.append(f"{(r + '/') if r else ''}train_gpt.py is gone: the benchmark runs `torchrun --standalone "
                            "--nproc_per_node=8 train_gpt.py`, so keep it as the entry point (it may import from other files).")
    if roots != [""]:
        checks += 1
        touched = sorted({f.split("/", 1)[0] for f in files} & set(roots))
        if len(touched) > 1:
            problems.append(f"The candidate changed {', '.join(touched)}; an experiment implements one change in one "
                            "record folder, so revert the others.")

    checks += 1
    sets = [(f, m.group(1).strip()) for f, text in src.items() for m in VAL_TOKENS_SET.finditer(text)]
    wrong = [(f, v) for f, v in sets if v.replace("_", "") != str(VAL_TOKENS)]
    if not sets:
        problems.append(f"val_tokens isn't set anywhere: keep `val_tokens: int = {VAL_TOKENS}` (the speedrun's "
                        "fixed validation size).")
    for f, v in wrong:
        problems.append(f"{f} sets val_tokens = {v}; the speedrun fixes it at {VAL_TOKENS}. Put it back.")

    checks += 1
    text = "\n".join(src.values())
    if "DATA_PATH" not in text or not all(d in text for d in DATA_FILES):
        problems.append("The data must come from the eval: read train and val shards from "
                        "os.environ['DATA_PATH'] + '/data/fineweb10B/fineweb_{train,val}_*.bin', as the base does.")

    checks += 1
    touched = sorted(f for f in files if f.startswith("data/") or "/data/" in f"/{f}" and f.split("/", 1)[0] in roots)
    if touched:
        problems.append(f"The candidate changed {', '.join(touched)}; the data pipeline under data/ is fixed. "
                        "Revert those files.")

    checks += 1
    for file, line in added_lines(diff):
        code = code_only(line)
        what = next((what for pattern, what in TUNING if pattern.search(code)), None)
        if what:
            problems.append(f"{file}: `{line.strip()[:160]}` {what}. The speedrun bans extra torch.compile "
                            "or inductor flags; remove it.")

    checks += 1
    if not FINAL_LINE.search(text):
        problems.append("No line prints `step:{step}/{total} val_loss:{val_loss:.4f} train_time:{ms:.0f}ms`; "
                        "the benchmark reads the final one, so keep printing it after the last validation.")

    print(f"subject {subject.name}: {len(src)} Python files, diff {len(diff)} chars, {len(problems)} problems",
          flush=True)
    report({"problems": len(problems), "checks": checks, "critique": problems})


if __name__ == "__main__":
    main()
