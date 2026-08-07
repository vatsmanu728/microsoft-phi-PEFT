##### src/evaluate.py

"""
Loads the base Phi-4 model + saved LoRA adapters, generates summaries for
the test set using greedy decoding, and scores them against reference
summaries using ROUGE-1, ROUGE-2, and ROUGE-L.

Resilience notes:
- Results are saved incrementally, one JSON line per example, to
  <adapter_path>/eval_results.jsonl — progress survives an interrupted
  Colab runtime, and a later run resumes instead of starting over.
- device_map={"": 0}, not "auto" — matches the fix applied in model.py;
  "auto" can attempt CPU/disk offload on a near-full GPU, which 4-bit
  quantization doesn't support without extra flags.
- load_finetuned_model_and_tokenizer() accepts an optional adapter_path
  override, so a specific checkpoint-N subfolder can be evaluated
  directly — not just the final output_dir — for lightweight
  mid-training spot-checks without waiting for training to complete.
"""

import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer

from src.config import get_config
from src.model import _build_bnb_config


def load_finetuned_model_and_tokenizer(cfg: dict, adapter_path: str = None):
    """
    Loads the frozen base Phi-4 model in 4-bit and attaches LoRA adapters
    from adapter_path (defaults to cfg['output_dir'] — the final trained
    adapter — if not specified). Pass a specific checkpoint-N subfolder
    path here to evaluate a mid-training checkpoint instead.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. Evaluation requires CUDA — "
            "check Runtime > Change runtime type > GPU in Colab."
        )

    if adapter_path is None:
        adapter_path = cfg["output_dir"]

    bnb_config = _build_bnb_config(cfg)

    print(f"Loading base model: {cfg['base_model_id']}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model_id"],
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    print(f"Attaching LoRA adapters from: {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # Tokenizer comes from the BASE MODEL, not adapter_path — mid-training
    # checkpoint-N subfolders don't contain tokenizer files at all (only
    # written once, to the top-level output_dir, when training fully
    # completes). Loading from base_model_id works correctly whether
    # evaluating a checkpoint or the final adapter.
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_id"], trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(adapter_path) ########### This is bugged - load_finetuned_model_and_tokenizer() loads the tokenizer from adapter_path — the same path as the adapter itself. That's wrong: the tokenizer never changes across training, checkpoints, or fine-tuning at all — it should always come from the base model, independent of which checkpoint you're evaluating.
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_summary(model, tokenizer, prompt: str, cfg: dict) -> str:
    """Builds the chat-formatted prompt and generates a summary via greedy decoding."""
    messages = [
        {"role": "system", "content": cfg["system_prompt"]},
        {"role": "user", "content": prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg["generation_max_new_tokens"],
            do_sample=cfg["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _results_path(adapter_path: str) -> str:
    """Path to the incremental JSONL results file, saved alongside the adapter."""
    return os.path.join(adapter_path, "eval_results.jsonl")


def _load_existing_results(results_path: str) -> list:
    """Loads already-computed per-example results from a previous (possibly interrupted) run."""
    if not os.path.exists(results_path):
        return []
    records = []
    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _summarize(records: list, cfg: dict) -> dict:
    """Computes average ROUGE scores across all saved records and prints them."""
    if not records:
        return {rt: 0.0 for rt in cfg["rouge_types"]}

    totals = {rt: 0.0 for rt in cfg["rouge_types"]}
    for r in records:
        for rt in cfg["rouge_types"]:
            totals[rt] += r["scores"][rt]
    averages = {rt: v / len(records) for rt, v in totals.items()}

    print(f"\nROUGE scores over {len(records)} test examples:")
    for rt, score in averages.items():
        print(f"  {rt}: {score:.4f}")

    return averages


def check_eval_progress(adapter_path: str = None, cfg: dict = None) -> dict:
    """
    Reports how many test examples have been evaluated so far, based on
    the incremental results file. Safe to call any time.
    """
    if cfg is None:
        cfg = get_config()
    if adapter_path is None:
        adapter_path = cfg["output_dir"]

    results_path = _results_path(adapter_path)
    records = _load_existing_results(results_path)
    target = cfg["eval_num_samples"]

    if not records:
        print(f"No evaluation results found yet at {results_path}.")
        return {"completed": 0, "target": target}

    print(f"Evaluated results for {len(records)}/{target} examples "
          f"(saved incrementally — safe even if the runtime was interrupted).")

    _summarize(records, cfg)
    return {"completed": len(records), "target": target}


def evaluate_model(test_dataset, num_samples: int = None, model=None, tokenizer=None,
                    adapter_path: str = None, resume: bool = True):
    """
    Generates summaries for up to num_samples examples from test_dataset,
    saving each result incrementally so progress survives an interrupted
    runtime. If resume=True (default) and partial results already exist,
    continues from where it left off instead of starting over.

    adapter_path: which adapter to evaluate. Defaults to cfg['output_dir']
    (the final trained adapter) if not specified — pass a specific
    checkpoint-N subfolder to evaluate a mid-training checkpoint instead.
    """
    cfg = get_config()

    if adapter_path is None:
        adapter_path = cfg["output_dir"]

    if num_samples is None:
        num_samples = cfg["eval_num_samples"]
    num_samples = min(num_samples, len(test_dataset))

    results_path = _results_path(adapter_path)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    existing = _load_existing_results(results_path) if resume else []
    start_index = len(existing)

    if start_index >= num_samples:
        print(f"✅ Already have {start_index}/{num_samples} results saved — nothing to do.")
        return _summarize(existing, cfg)

    if start_index > 0:
        print(f"Resuming: found {start_index}/{num_samples} results already saved. "
              f"Continuing from example {start_index}.")

    if model is None or tokenizer is None:
        model, tokenizer = load_finetuned_model_and_tokenizer(cfg, adapter_path=adapter_path)

    scorer = rouge_scorer.RougeScorer(cfg["rouge_types"], use_stemmer=True)

    with open(results_path, "a") as f:
        for i in tqdm(range(start_index, num_samples), desc="Evaluating",
                      initial=start_index, total=num_samples):
            example = test_dataset[i]
            prediction = generate_summary(model, tokenizer, example["prompt"], cfg)
            reference = example["target"]

            scores = scorer.score(reference, prediction)
            record = {
                "index": i,
                "prediction": prediction,
                "reference": reference,
                "scores": {rt: scores[rt].fmeasure for rt in cfg["rouge_types"]},
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    all_results = _load_existing_results(results_path)
    print(f"Evaluated results for {len(all_results)}/{num_samples} examples — complete.")
    return _summarize(all_results, cfg)


def evaluate_checkpoint(checkpoint_name: str, num_samples: int = 30):
    """
    Convenience function for a quick, lightweight spot-check of a
    SPECIFIC mid-training checkpoint (e.g. "checkpoint-300"), on a small
    sample rather than the full test set. Use this instead of the full
    evaluate_model() call when training isn't complete yet.

    Example: evaluate_checkpoint("checkpoint-300", num_samples=30)
    """
    from src.data_loader import load_and_prepare_datasets
    from src.preprocess import preprocess_datasets

    cfg = get_config()
    adapter_path = os.path.join(cfg["output_dir"], checkpoint_name)

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"No checkpoint found at {adapter_path}. "
            f"Check the checkpoint name and that it hasn't been pruned "
            f"by save_total_limit."
        )

    datasets = load_and_prepare_datasets()
    processed = preprocess_datasets(datasets)

    return evaluate_model(
        test_dataset=processed["test"],
        num_samples=num_samples,
        adapter_path=adapter_path,
    )


if __name__ == "__main__":
    from src.data_loader import load_and_prepare_datasets
    from src.preprocess import preprocess_datasets

    datasets = load_and_prepare_datasets()
    processed_datasets = preprocess_datasets(datasets)

    evaluate_model(test_dataset=processed_datasets["test"], num_samples=500)
