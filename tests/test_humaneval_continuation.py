from __future__ import annotations

from Magpie.scripts.benchmark.lm_eval_humaneval_compat import (
    open_final_assistant_continuation,
    render_humaneval_chat_template,
)


def test_removes_suffix_that_closes_assistant_prefix():
    prefix = "Here is the completed function:\n```python\ndef answer():\n"
    rendered = f'<assistant><response>{prefix}</response></assistant><eos>'
    chat = [
        {"role": "user", "content": "Complete this function"},
        {"role": "assistant", "content": prefix},
    ]

    assert open_final_assistant_continuation(
        rendered,
        chat,
        add_generation_prompt=False,
    ) == f"<assistant><response>{prefix}"


def test_keeps_template_that_already_leaves_prefix_open():
    prefix = "Here is the completed function:\n```python\ndef answer():\n"
    rendered = f"<assistant>{prefix}"
    chat = [{"role": "assistant", "content": prefix}]

    assert (
        open_final_assistant_continuation(
            rendered,
            chat,
            add_generation_prompt=False,
        )
        == rendered
    )


def test_keeps_non_continuation_and_unrecognized_templates_unchanged():
    chat = [{"role": "assistant", "content": "raw <prefix>"}]

    assert (
        open_final_assistant_continuation(
            "<assistant>raw <prefix></assistant>",
            chat,
            add_generation_prompt=True,
        )
        == "<assistant>raw <prefix></assistant>"
    )
    assert (
        open_final_assistant_continuation(
            "<assistant>raw &lt;prefix&gt;</assistant>",
            chat,
            add_generation_prompt=False,
        )
        == "<assistant>raw &lt;prefix&gt;</assistant>"
    )
    assert (
        open_final_assistant_continuation(
            [1, 2, 3],
            chat,
            add_generation_prompt=False,
        )
        == [1, 2, 3]
    )


def test_renders_huggingface_template_locally_for_string_requests():
    prefix = "```python\ndef answer():\n"
    chat = [{"role": "assistant", "content": prefix}]

    class Tokenizer:
        def apply_chat_template(self, history, **kwargs):
            assert history == chat
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": False,
                "continue_final_message": True,
            }
            return f"<assistant>{prefix}</assistant><eos>"

    class Model:
        tokenizer_backend = "huggingface"
        tokenized_requests = False
        tokenizer = Tokenizer()

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("local Hugging Face rendering must not use JSON chat")

    rendered = render_humaneval_chat_template(
        Model(),
        chat,
        add_generation_prompt=False,
        fallback=unexpected_fallback,
    )

    assert isinstance(rendered, str)
    assert rendered == f"<assistant>{prefix}"


def test_uses_lm_eval_fallback_without_huggingface_tokenizer():
    chat = [{"role": "user", "content": "hello"}]
    sentinel = object()

    def fallback(model, history, *, add_generation_prompt):
        assert model.tokenizer_backend == "remote"
        assert history == chat
        assert add_generation_prompt is True
        return sentinel

    model = type("Model", (), {"tokenizer_backend": "remote", "tokenizer": None})()
    assert (
        render_humaneval_chat_template(
            model,
            chat,
            add_generation_prompt=True,
            fallback=fallback,
        )
        is sentinel
    )
