"""Run lm-eval with reliable assistant-prefix continuation for HumanEval."""

from __future__ import annotations

import runpy
from collections.abc import Mapping, Sequence
from typing import Any


def open_final_assistant_continuation(
    rendered: Any,
    chat_history: Any,
    *,
    add_generation_prompt: bool,
) -> Any:
    """Remove template suffixes that incorrectly close a HumanEval prefix."""
    if add_generation_prompt or not isinstance(rendered, str):
        return rendered
    if not isinstance(chat_history, Sequence) or not chat_history:
        return rendered

    final_message = chat_history[-1]
    if not isinstance(final_message, Mapping):
        return rendered
    if final_message.get("role") != "assistant":
        return rendered

    prefix = final_message.get("content")
    if not isinstance(prefix, str) or not prefix:
        return rendered

    prefix_start = rendered.rfind(prefix)
    if prefix_start < 0:
        return rendered

    # Some tokenizer templates ignore continue_final_message and append their
    # assistant/end-of-message markers. Preserve all model-native opening
    # tokens while leaving the provided function prefix open for generation.
    return rendered[: prefix_start + len(prefix)]


def install_humaneval_continuation_patch() -> None:
    from lm_eval.models.api_models import TemplateAPI

    original = TemplateAPI.apply_chat_template
    if getattr(original, "_magpie_humaneval_continuation", False):
        return

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> Any:
        rendered = original(
            self,
            chat_history,
            add_generation_prompt=add_generation_prompt,
        )
        return open_final_assistant_continuation(
            rendered,
            chat_history,
            add_generation_prompt=add_generation_prompt,
        )

    apply_chat_template._magpie_humaneval_continuation = True  # type: ignore[attr-defined]
    TemplateAPI.apply_chat_template = apply_chat_template


def main() -> None:
    install_humaneval_continuation_patch()
    runpy.run_module("lm_eval.__main__", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
