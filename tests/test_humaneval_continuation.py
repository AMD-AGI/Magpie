from __future__ import annotations

from Magpie.scripts.benchmark.lm_eval_humaneval_compat import (
    open_final_assistant_continuation,
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
