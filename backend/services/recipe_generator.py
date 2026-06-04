"""
FitNova - Recipe Instruction Generator (serving the fine-tuned DistilGPT-2)
===========================================================================
Serves the N2 model: given a recipe name + ingredients + calorie level, generate
the cooking instructions. This is the academic centerpiece (beats the paper on
BLEU/ROUGE/Distinct) made available at runtime.

Lazy, fault-tolerant: torch + transformers load only on first use, and if the
model directory is absent (it is gitignored — 312 MB, copied from Drive) the
service reports unavailable rather than crashing the server. CPU inference is
fine for single-recipe, on-demand generation.

Model dir: backend/models/recipe_model_tf/ (override via FITNOVA_RECIPE_MODEL_DIR).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("fitnova.recipe_gen")

_DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "recipe_model_tf")
_CAL = {0: "low", 1: "medium", 2: "high", "low": "low", "medium": "medium", "high": "high"}


class RecipeGenerator:
    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or os.environ.get(
            "FITNOVA_RECIPE_MODEL_DIR", _DEFAULT_DIR)
        self._model = None
        self._tok = None
        self._loaded = False
        self._load_error: str | None = None
        # Presence check only — defer the heavy import to first generate().
        self.available = os.path.isdir(self.model_dir) and os.path.exists(
            os.path.join(self.model_dir, "config.json"))
        if not self.available:
            logger.warning("Recipe model not found at %s — generator disabled "
                           "(copy from Drive FoodCom/model_tf to enable).", self.model_dir)

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if not self.available:
            return False
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._torch = torch
            self._tok = AutoTokenizer.from_pretrained(self.model_dir)
            self._tok.pad_token = self._tok.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(self.model_dir)
            self._model.eval()
            self._loaded = True
            logger.info("Recipe generator loaded from %s", self.model_dir)
            return True
        except Exception as exc:
            self._load_error = str(exc)
            logger.exception("Failed to load recipe generator")
            return False

    def generate(
        self,
        name: str,
        ingredients: list[str],
        calorie_level: str | int = "medium",
        max_new_tokens: int = 200,
    ) -> dict:
        """Generate instructions for one recipe. Returns
        {ok, instructions, prompt} or {ok: False, error}."""
        if not self._ensure_loaded():
            return {"ok": False,
                    "error": self._load_error or "Recipe generator unavailable "
                    "(model not installed)."}
        cal = _CAL.get(calorie_level, "medium")
        ing = ", ".join(i.strip() for i in ingredients if i and i.strip())
        prompt = (f"Recipe: {name}\nIngredients: {ing}\n"
                  f"Calorie level: {cal}\nInstructions:")
        torch = self._torch
        enc = self._tok(prompt, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                no_repeat_ngram_size=3,
                pad_token_id=self._tok.pad_token_id,
            )
        gen = out[0, enc["input_ids"].shape[1]:]
        text = self._tok.decode(gen, skip_special_tokens=True).strip()
        return {"ok": True, "instructions": text, "prompt": prompt}
