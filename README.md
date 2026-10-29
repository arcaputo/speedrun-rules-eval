# speedrun-rules

A Kupanga QA eval for the modded-nanogpt speedrun. It checks that a candidate keeps the rules that make
its time count, before `speedrun-8xh100` spends GPU time on it. It runs on a CPU in seconds.
Deterministic, `qa_cap = 3`. Each broken rule becomes one line of critique for the agent.

| Check | Fails when |
|---|---|
| Entry point | `train_gpt.py` is gone (the benchmark runs `torchrun … train_gpt.py`) |
| `val_tokens` | It is set to anything but 10485760, or isn't set at all |
| Data | Train or val files aren't read from `DATA_PATH/data/fineweb10B/fineweb_{train,val}_*.bin`, or a file under `data/` changed |
| Compiler flags | An added line sets `torch._inductor.config`, `torch._dynamo.config` (other than cache limits), `mode=`/`options=`/`backend=` on `torch.compile`, `TORCHINDUCTOR_*`, `max_autotune` or `coordinate_descent_tuning` |
| Final line | Nothing prints `step:{step}/{total} val_loss:… train_time:…ms` |

Flags are checked only on lines the candidate added (`$HARNESS_CHANGES_FILE`), with comments ignored.
On the base node only the tree is checked. Changing how the model computes its validation loss, such as
masking infeasible tokens as #91 does, is the benchmark's business and is allowed.

Checked against the real records: #87 as a base, and #88–#92 each as a candidate over the record
before it, all pass. A candidate that halves `val_tokens`, sets `max_autotune` and edits
`data/cached_fineweb10B.py` gets 4 problems.

Local check: `HARNESS_SUBJECTS_DIR=/path/to/subjects [HARNESS_CHANGES_FILE=changes.diff] python qa.py`
