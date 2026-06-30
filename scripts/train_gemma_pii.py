#!/usr/bin/env python3
"""
Self-contained training script for small Gemma (Gemma-2 2B-class or Gemma-4 small)
on PII redaction (or persona chat) synthetic data. Prototype / harness for iterating
toward a custom "Grimmy" personality (current default is qwen3b).

Usage (minimal):
  python scripts/train_gemma_pii.py --task pii --examples 10 --output-dir /tmp/grimalkin-train/gemma_train

Produces loadable artifact:
- If torch + transformers (+ peft for LoRA): does actual HF causal-LM training loop (SFT-style)
  on the synthetic jsonl using a small base (distilgpt2 for demo/env practicality, or --model
  for gemma-2-2b-it etc.); saves HF config/weights. GGUF is post-processing note (e.g. via llama.cpp).
  unsloth path noted for when that extra dep is present.
- Else (no torch etc.): writes synthetic data + honest stub marked "simulated":true with same layout.

This is the prototype harness for training a small Gemma (or equiv) for PII or new Grimmy personality.
Full runs use real deps + your data. Captures output + artifacts. Remote model
downloads are opt-in; by default real training requires a local model path.

Designed for iteration on grimalkin personality or redactor.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

SCRATCH = Path(os.environ.get("GRIM_TRAIN_SCRATCH", "/tmp/grimalkin-train"))
META_FILE = "grimalkin_training_meta.json"

def generate_synthetic_pii_data(n: int = 10):
    """Generate synthetic (instruction, input, output) for PII redaction task."""
    examples = []
    templates = [
        ("My name is {name} and SSN is {ssn}.", "My name is [NAME_1] and SSN is [SSN_1]."),
        ("Contact {name} at {email} living at {addr}.", "Contact [NAME_1] at [EMAIL_1] living at [ADDR_1]."),
        ("Pay {name} using card {cc} phone {phone}.", "Pay [NAME_1] using card [CC_1] phone [PHONE_1]."),
    ]
    names = ["Alice Chen", "Bob Rivera", "Sam Patel"]
    ssns = ["123-45-6789", "111-22-3333", "078-05-1120"]
    emails = ["alice@ex.com", "bob@corp.io"]
    addrs = ["123 Oak St, Springfield", "456 Pine Ave, Metropolis"]
    ccs = ["4111-1111-1111-1111", "5555 5555 5555 4444"]
    phones = ["555-123-4567", "(312) 555-0199"]

    for i in range(n):
        t = templates[i % len(templates)]
        name = names[i % len(names)]
        ssn = ssns[i % len(ssns)]
        email = emails[i % len(emails)]
        addr = addrs[i % len(addrs)]
        cc = ccs[i % len(ccs)]
        phone = phones[i % len(phones)]

        inp = t[0].format(name=name, ssn=ssn, email=email, addr=addr, cc=cc, phone=phone)
        out = t[1].format(name="[NAME_1]", ssn="[SSN_1]", email="[EMAIL_1]", addr="[ADDR_1]", cc="[CC_1]", phone="[PHONE_1]")
        examples.append({
            "instruction": "Redact all personal information using stable placeholders. Keep context.",
            "input": inp,
            "output": out,
        })
    return examples

def generate_persona_data(n: int = 10):
    """Simple synthetic for grimalkin-style personality fine-tune."""
    examples = []
    for i in range(n):
        user = f"User: Remember my cat is named Whiskers and I live in the hills."
        assistant = f"Grimalkin: Understood — Whiskers in the hills. How can I guard your vault today?"
        examples.append({"text": f"{user}\n{assistant}"})
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["pii", "persona"], default="pii")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=str(SCRATCH / "gemma_train"))
    parser.add_argument("--model", type=str, default="", help="local HF model directory; remote IDs require --allow-downloads")
    parser.add_argument("--allow-downloads", action="store_true", help="allow transformers to fetch a remote --model ID")
    parser.add_argument("--trust-remote-code", action="store_true", help="pass trust_remote_code=True to transformers")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now()}] Starting Gemma training prototype for task={args.task}")
    print(f"Target model: {args.model}")
    print(f"Output: {out_dir}")

    if args.task == "pii":
        data = generate_synthetic_pii_data(args.examples)
        data_path = out_dir / "pii_redact_train.jsonl"
    else:
        data = generate_persona_data(args.examples)
        data_path = out_dir / "persona_train.jsonl"

    with open(data_path, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(data)} synthetic examples to {data_path}")

    trained_path = None

    def train_real(data: list, out_dir: Path, task: str) -> Path:
        """Attempt real training when torch+transformers available.
        Uses small base (distilgpt2 for feasibility or --model) + manual SFT loop on synthetic data.
        Adds LoRA if peft present. Saves HF style. No linear proxy.
        If deps missing, raises so caller uses honest stub.
        """
        import hashlib
        model_dir = out_dir / "gemma_trained"
        model_dir.mkdir(parents=True, exist_ok=True)
        print("[REAL] entering real Gemma/personality training path")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        base = args.model
        if not base:
            raise RuntimeError("real training requires --model pointing at a local model directory")
        if not Path(base).exists() and not args.allow_downloads:
            raise RuntimeError("remote model IDs require --allow-downloads")
        print(f"[REAL] loading base {base} for task={task}")
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=args.trust_remote_code)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=args.trust_remote_code)
        # LoRA if peft
        try:
            from peft import LoraConfig, get_peft_model
            lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none")
            model = get_peft_model(model, lora)
            print("[REAL] LoRA (peft) enabled")
        except Exception:
            print("[REAL] peft not available, full fine-tune (or as loaded)")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        print("=== BEGIN REAL TORCH + HF TRAINING (Gemma/personality) ===")
        for step, ex in enumerate(data):
            text = ex.get("input", ex.get("text", "")) + "\n" + ex.get("output", ex.get("text", ""))
            enc = tok(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
            labels = enc["input_ids"].clone()
            labels[labels == tok.pad_token_id] = -100
            out = model(**enc, labels=labels)
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            print(f"  step {step+1}/{len(data)}: loss={loss.item():.4f}")
            if step >= 9:  # cap for speed
                break
        print("=== END REAL TRAINING ===")
        model.save_pretrained(model_dir)
        tok.save_pretrained(model_dir)
        with open(model_dir / META_FILE, "w") as f:
            json.dump({
                "model_type": "gemma-or-small",
                "task": task,
                "base": base,
                "real": True,
                "lora": "peft" if 'peft' in dir() else False,
                "grimalkin_use": "personality_or_redact",
            }, f)
        gguf = out_dir / "gemma_redact_demo.gguf"
        with open(gguf, "w") as f:
            f.write("# Run 'python -m llama.cpp.convert_hf_to_gguf " + str(model_dir) + "' for GGUF after training\n")
        print(f"[REAL] saved HF model to {model_dir}")
        return model_dir

    def train_stub(data: list, out_dir: Path, task: str) -> Path:
        """Honest stub when torch unavailable. Same artifact layout, marked simulated."""
        model_dir = out_dir / "gemma_trained"
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "config.json", "w") as f:
            json.dump({
                "model_type": "gemma",
                "task": task,
                "base": args.model,
                "simulated": True,
                "grimalkin_use": "personality_or_redact",
                "note": "torch not available; real branch (Gemma attempt) used when torch+transformers present"
            }, f)
        with open(model_dir / META_FILE, "w") as f:
            json.dump({
                "model_type": "gemma",
                "task": task,
                "base": args.model,
                "simulated": True,
                "grimalkin_use": "personality_or_redact",
            }, f)
        with open(model_dir / "pytorch_model.bin", "w") as f:
            f.write("SIMULATED_WEIGHTS_GEMMA_PII_OR_PERSONA_STUB")
        gguf = out_dir / "gemma_redact_demo.gguf"
        with open(gguf, "w") as f:
            f.write("# Gemma small model stub (BRANCH=stub)\n")
        print("BRANCH=stub")
        print(f"Artifact (loadable stub): {model_dir}")
        return model_dir

    try:
        artifact = train_real(data, out_dir, args.task)
    except Exception as e:
        print(f"real training not possible in this env ({type(e).__name__}); using honest stub")
        artifact = train_stub(data, out_dir, args.task)

    trained_path = artifact

    # Demo "load" for iteration (personality path)
    def demo_load_and_infer(stub_dir: Path, prompt: str) -> str:
        meta_path = stub_dir / META_FILE
        cfgf = stub_dir / "config.json"
        if meta_path.exists() or cfgf.exists():
            meta = json.load(open(meta_path if meta_path.exists() else cfgf))
            if meta.get("real"):
                return f"[Gemma-tiny-real] Inferred for: {prompt[:50]}..."
            task = meta.get("task", "pii")
            if task == "persona":
                return f"[Gemma-persona-stub] Understood vault query: {prompt[:60]}... (would guard with trained Grimmy weights)"
            return f"[Gemma-redact-stub] Redacted version of: {prompt[:50]}..."
        return "[Gemma-stub] " + prompt

    demo_out = demo_load_and_infer(artifact, "remember my ssn is hidden now train new grimmy")
    print(f"Load demo (for grimalkin gemma_personality_model wiring): {demo_out}")

    print(f"Training script completed. Artifact: {trained_path}")
    log_path = out_dir / "train.log"
    meta = json.load(open(trained_path / META_FILE)) if (trained_path / META_FILE).exists() else {}
    with open(log_path, "a") as f:
        f.write(
            f"[{datetime.now()}] task={args.task} examples={args.examples} model={args.model} "
            f"artifact={trained_path} branch={'real' if meta.get('real') else 'stub'}\n"
        )
    print(f"Log appended to {log_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
