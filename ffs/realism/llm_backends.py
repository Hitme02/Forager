"""LLM backends for the Tier-B agent harness -- a small, real model picking bounded actions.

These wire a *real* language model into the `agent_harness.LLMBrain` seam for the live demo.
The model's only job is to choose the next action from the fixed, non-weaponised menu
(probe / enumerate / attempt_access / collect) given a short textual observation of one
service -- so even a small instruct model suffices, and an invalid reply falls back to the
scripted tactic. Nothing here enlarges what an agent can do; it only chooses among the four
grounded steps the harness already exposes.

Backends share one interface (`__call__(prompt) -> str`, plus `batch(prompts) -> list[str]`):

- `OllamaBackend`   -- a local model served by Ollama over HTTP (Metal-accelerated on this
  Mac). The preferred local backend: no Python ML stack to build, works on any Python. Its
  `batch()` fires the W decisions as CONCURRENT requests, which Ollama continuous-batches on
  the GPU -- the PAGP point: data-parallel inference across the worker pool.
- `HFLocalBackend`  -- a small Hugging Face `transformers` model on MPS/CPU; `batch()` is one
  padded forward pass (true batched GPU inference). Used when `transformers` is installed.
- `HFApiBackend`    -- the Hugging Face serverless Inference API over HTTPS (needs HF_TOKEN).

`make_backend("auto", ...)` picks Ollama if it is serving, else a local transformers model,
else the API if a token is present, else returns None so the demo degrades to the scripted
brain and still runs. All heavy imports are lazy, so importing this module costs nothing.
"""

from __future__ import annotations

import os

# The only thing the model must do: emit one of the four bounded action words. The prompt
# gives the standard recon->access->loot methodology so a small model plays productively; it
# still reads the live state and chooses per service.
SYSTEM_PROMPT = (
    "You are an autonomous penetration-testing agent working ONE service on an authorised, "
    "isolated test range. Choose the single next action from exactly these four words: "
    "probe, enumerate, attempt_access, collect.\n"
    "Methodology: if the service is not yet reachable, choose probe; if it is reachable but "
    "not enumerated, choose enumerate; if it is enumerated but you do NOT yet have access, "
    "choose attempt_access; once you HAVE access, choose collect (repeatedly) to exfiltrate "
    "its artifacts and any flag.\n"
    "Reply with only one of the four words and nothing else."
)

_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_OLLAMA_HOST = "http://localhost:11434"


class OllamaBackend:
    """A local model served by Ollama (HTTP). Preferred local backend -- no ML stack to build.

    `batch(prompts)` issues the W decisions concurrently; with OLLAMA_NUM_PARALLEL > 1 Ollama
    continuous-batches them on the GPU, so the pool's decisions are served data-parallel."""

    def __init__(self, model: str | None = None, host: str = _OLLAMA_HOST,
                 max_new_tokens: int = 5) -> None:
        import requests
        self.host = host.rstrip("/")
        self.max_new_tokens = max_new_tokens
        tags = requests.get(f"{self.host}/api/tags", timeout=3).json()
        names = [m["name"] for m in tags.get("models", [])]
        if not names:
            raise RuntimeError("Ollama is serving but has no models pulled.")
        # Use the requested model if present; an HF-style id (contains '/') means "no
        # preference" -> take the first available local model.
        if model and "/" not in model and any(n == model or n.startswith(model) for n in names):
            self.model = model
        else:
            self.model = names[0]

    def _chat(self, prompt: str) -> str:
        import requests
        r = requests.post(
            f"{self.host}/api/chat",
            json={"model": self.model, "stream": False,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": prompt}],
                  "options": {"num_predict": self.max_new_tokens, "temperature": 0.0}},
            timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def __call__(self, prompt: str) -> str:
        return self._chat(prompt)

    def batch(self, prompts: list[str]) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor
        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=len(prompts)) as ex:
            return list(ex.map(self._chat, prompts))

    @property
    def device(self) -> str:
        return f"ollama:{self.model}"


class HFLocalBackend:
    """A small local instruct model (Hugging Face `transformers`) on MPS/CPU.

    `batch(prompts)` is a single padded `generate()` call -- one GPU inference serving every
    worker's decision this tick -- which is what the demo measures against sequential calls."""

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str | None = None,
                 max_new_tokens: int = 5) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"           # left-pad so generations align for a batch
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _format(self, user_prompt: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def batch(self, prompts: list[str]) -> list[str]:
        import torch
        if not prompts:
            return []
        texts = [self._format(p) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [self.tok.decode(g, skip_special_tokens=True).strip() for g in gen]

    def __call__(self, prompt: str) -> str:
        return self.batch([prompt])[0]


class HFApiBackend:
    """Hugging Face serverless Inference API (chat-completions router). Needs HF_TOKEN."""

    def __init__(self, model_name: str = _DEFAULT_MODEL, token: str | None = None,
                 max_new_tokens: int = 5) -> None:
        self.model_name = model_name
        self.token = token or os.environ.get("HF_TOKEN")
        if not self.token:
            raise RuntimeError("HFApiBackend needs an HF_TOKEN environment variable.")
        self.url = "https://router.huggingface.co/v1/chat/completions"
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        import requests
        r = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.token}"},
            json={"model": self.model_name, "max_tokens": self.max_new_tokens,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": prompt}]},
            timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def batch(self, prompts: list[str]) -> list[str]:
        # The serverless API has no batch endpoint; issue one request per prompt.
        return [self(p) for p in prompts]


def make_backend(kind: str = "auto", model_name: str = _DEFAULT_MODEL):
    """Return an LLM backend, or None if none is available (demo falls back to scripted).

    kind: "auto" (Ollama if serving, else local transformers, else API if HF_TOKEN, else None),
          "ollama", "hf-local", "hf-api", or "scripted"/"none" (-> None)."""
    kind = kind.lower()
    if kind in ("scripted", "none"):
        return None
    if kind in ("auto", "ollama"):
        try:
            return OllamaBackend(model_name)
        except Exception:
            if kind == "ollama":
                raise
    if kind in ("auto", "hf-local"):
        try:
            import transformers  # noqa: F401  (probe availability)
            return HFLocalBackend(model_name)
        except Exception:
            if kind == "hf-local":
                raise
    if kind in ("auto", "hf-api"):
        try:
            return HFApiBackend(model_name)
        except Exception:
            if kind == "hf-api":
                raise
    return None
