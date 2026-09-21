"""Runtime coverage that actually starts `python -m lm_eval`.

Magpie CI does not install lm-eval; these tests skip unless it is importable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

lm_eval = pytest.importorskip("lm_eval")

ROOT = Path(__file__).parents[1]
COMPAT = ROOT / "Magpie" / "scripts" / "benchmark" / "magpie_bench_remote_compat.sh"
TOKENIZER_ID = "hf-internal-testing/tiny-random-gpt2"


def _choice(n_tokens: int, *, text: str = " 2", index: int = 0) -> dict:
    n = max(int(n_tokens) + 8, 64)
    tokens = [f"t{i}" for i in range(n)]
    token_logprobs = [-0.1] * n
    top_logprobs = [{tokens[i]: -0.1} for i in range(n)]
    return {
        "index": index,
        "text": text,
        "logprobs": {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
        },
    }


def _prompt_widths(prompt) -> list[int]:
    if isinstance(prompt, str):
        return [max(len(prompt.split()), 1)]
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
        return [len(prompt)]
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
        return [max(len(item.split()), 1) for item in prompt]
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
        return [max(len(item), 1) for item in prompt]
    return [8]


class CompletionsHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        widths = _prompt_widths(payload.get("prompt", ""))
        echo = bool(payload.get("echo"))
        choices = [
            _choice(width, text=" 2", index=index)
            for index, width in enumerate(widths)
        ]
        if not echo:
            for choice in choices:
                choice["text"] = "2"
        body = json.dumps({"id": "cmpl-magpie", "choices": choices}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def completions_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), CompletionsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _write_tiny_tasks(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    include = tmp_path / "tasks"
    include.mkdir()
    mc_samples = tmp_path / "tiny_mc.jsonl"
    mc_samples.write_text(
        json.dumps({"question": "1+1=", "choices": ["1", "2"], "answer": 1}) + "\n",
        encoding="utf-8",
    )
    gen_samples = tmp_path / "tiny_gen.jsonl"
    gen_samples.write_text(
        json.dumps({"question": "say two", "answer": "2"}) + "\n",
        encoding="utf-8",
    )
    (include / "magpie_tiny_mc.yaml").write_text(
        "\n".join(
            [
                "task: magpie_tiny_mc",
                "dataset_path: json",
                "dataset_kwargs:",
                f"  data_files: {{test: {mc_samples}}}",
                "output_type: multiple_choice",
                "test_split: test",
                'doc_to_text: "{{question}}"',
                'doc_to_choice: "{{choices}}"',
                'doc_to_target: "{{answer}}"',
                "metric_list:",
                "  - metric: acc",
                "    aggregation: mean",
                "    higher_is_better: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (include / "magpie_tiny_unsafe.yaml").write_text(
        "\n".join(
            [
                "task: magpie_tiny_unsafe",
                "unsafe_code: true",
                "dataset_path: json",
                "dataset_kwargs:",
                f"  data_files: {{test: {gen_samples}}}",
                "output_type: generate_until",
                "test_split: test",
                'doc_to_text: "{{question}}"',
                'doc_to_target: "{{answer}}"',
                "generation_kwargs:",
                "  until: ['\\n']",
                "  max_gen_toks: 8",
                "metric_list:",
                "  - metric: exact_match",
                "    aggregation: mean",
                "    higher_is_better: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return include


def _run_magpie_eval(
    tmp_path: Path,
    *,
    base_url: str,
    tasks: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    include = _write_tiny_tasks(tmp_path)
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(COMPAT),
        "RESULT_DIR": str(tmp_path / "result"),
        "BENCHMARK_BASE_URL": base_url,
        "MAGPIE_EVAL_PYTHON": sys.executable,
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": tasks,
        "MAGPIE_EVAL_INCLUDE_PATH": str(include),
        "MAGPIE_EVAL_LIMIT": "1",
        "MAGPIE_EVAL_BATCH_SIZE": "1",
        "MAGPIE_EVAL_CONCURRENCY": "1",
        "MAGPIE_EVAL_TOKENIZED_REQUESTS": "false",
        "MODEL": TOKENIZER_ID,
        "HF_HOME": str(tmp_path / "hf"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "OPENAI_API_KEY": "dummy",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if extra_env:
        env.update(extra_env)
    env.pop("HF_ALLOW_CODE_EVAL", None)
    (tmp_path / "result").mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        ["bash", "-c", 'source "$MAGPIE_COMPAT" && magpie_run_eval_remote_direct'],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


def test_chat_backend_cannot_score_multiple_choice():
    from lm_eval.models.openai_completions import LocalChatCompletion

    model = LocalChatCompletion(
        model=TOKENIZER_ID,
        base_url="http://127.0.0.1:9/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
    )
    with pytest.raises(NotImplementedError, match="Loglikelihood is not supported"):
        model.loglikelihood([])


def test_magpie_local_completions_runs_multiple_choice(tmp_path, completions_server):
    completed = _run_magpie_eval(tmp_path, base_url=completions_server, tasks="magpie_tiny_mc")
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert "--model local-completions" in completed.stderr
    result = tmp_path / "result" / "lm_eval" / "results_conc1.json"
    assert result.is_file(), completed.stderr[-4000:]
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert "lm_eval_version" in payload
    assert "magpie_tiny_mc" in payload["results"]
    assert payload["results"]["magpie_tiny_mc"].get("acc,none") is not None


def test_magpie_passes_confirm_unsafe_to_real_lm_eval(tmp_path, completions_server):
    blocked = _run_magpie_eval(tmp_path, base_url=completions_server, tasks="magpie_tiny_unsafe")
    assert blocked.returncode != 0
    assert "unsafe" in (blocked.stderr + blocked.stdout).lower()

    allowed = _run_magpie_eval(
        tmp_path / "ok",
        base_url=completions_server,
        tasks="magpie_tiny_unsafe",
        extra_env={"MAGPIE_EVAL_CONFIRM_UNSAFE_CODE": "true"},
    )
    assert allowed.returncode == 0, allowed.stderr[-4000:]
    assert "--confirm_run_unsafe_code" in allowed.stderr
    payload = json.loads(
        (tmp_path / "ok" / "result" / "lm_eval" / "results_conc1.json").read_text(
            encoding="utf-8"
        )
    )
    assert "magpie_tiny_unsafe" in payload["results"]


def test_magpie_sets_hf_allow_code_eval_for_code_eval_metric(tmp_path):
    evaluate = pytest.importorskip("evaluate")
    shell = r"""
source "$MAGPIE_COMPAT"
export MAGPIE_EVAL_TASKS=humaneval_instruct
unset HF_ALLOW_CODE_EVAL
magpie_eval_apply_code_eval_env
"$PYTHON" - <<'PY'
import os
from evaluate import load

assert os.environ.get("HF_ALLOW_CODE_EVAL") == "1"
metric = load("code_eval")
os.environ.pop("HF_ALLOW_CODE_EVAL", None)
try:
    metric.compute(
        predictions=[["def add(a, b):\n    return a + b\n"]],
        references=["assert add(1, 2) == 3"],
        k=[1],
    )
except ValueError as exc:
    assert "HF_ALLOW_CODE_EVAL" in str(exc)
else:
    raise AssertionError("code_eval must refuse to run without HF_ALLOW_CODE_EVAL")
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
PY
"""
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(COMPAT),
        "PYTHON": sys.executable,
    }
    env.pop("HF_ALLOW_CODE_EVAL", None)
    completed = subprocess.run(
        ["bash", "-c", shell],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        cwd=str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
