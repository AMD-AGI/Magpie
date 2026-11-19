"""Opt-in contract tests against an unmodified real InferenceX checkout.

Set MAGPIE_TEST_INFERENCEX_PATH to the checkout or its inferencex-e2e project.
These tests need no GPU, Slurm, model download, or AgentX harness installation.
"""

import json
import os
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from Magpie.modes.benchmark.agentx import (
    ensure_agentx_dependencies,
    resolve_agentx_recipe,
)
from Magpie.modes.benchmark.agentx_runtime import (
    _client_environment,
    _preflight_modern_client,
)
from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.inferencex import resolve_inferencex_root
from Magpie.modes.benchmark.result import ResultParser


@pytest.fixture
def upstream():
    path = os.environ.get("MAGPIE_TEST_INFERENCEX_PATH")
    if not path:
        pytest.skip("set MAGPIE_TEST_INFERENCEX_PATH to run upstream contract tests")
    return Path(resolve_inferencex_root(path))


@pytest.mark.parametrize(
    "recipe,runner,conc",
    [
        ("qwen3.5-fp8-mi355x-sglang-agentic-mtp", "mi355x", 4),
        ("glm5.2-fp8-mi325x-sglang-agentic-mtp", "mi325x", 2),
        ("dsv41flash-fp4-mi355x-vllm-agentic-dspark", "mi355x", 1),
    ],
)
def test_real_native_recipe_and_client_plan(upstream, tmp_path, recipe, runner, conc):
    row = yaml.safe_load((upstream / "configs/amd-master.yaml").read_text())[recipe]
    config = BenchmarkConfig.from_dict(
        {
            "model": row["model"],
            "framework": row["framework"],
            "precision": row["precision"],
            "agentx": {
                "recipe": recipe,
                "concurrency": conc,
                "selector": {
                    "tp": row["scenarios"]["agentic-coding"][0]["search-space"][0]["tp"]
                },
            },
            "inferencex_path": str(upstream),
        }
    )
    launch = resolve_agentx_recipe(config, str(upstream.parent), runner)
    ensure_agentx_dependencies(str(upstream))
    env = _client_environment(
        {"client_env": config.get_env_vars()}, launch.server, upstream, tmp_path
    )
    _preflight_modern_client(upstream, env)
    assert int(config.envs["CONC"]) == conc
    assert launch.server["framework"] == row["framework"]
    assert config.model in launch.server["argv"]


def test_real_aggregation_then_magpie_parsing(upstream, tmp_path):
    artifacts = tmp_path / "aiperf_artifacts"
    artifacts.mkdir()
    records = []
    for i in range(2):
        records.append(
            {
                "metadata": {
                    "conversation_id": "trace-A",
                    "turn_index": i,
                    "request_start_ns": (i + 1) * 1_000_000_000,
                    "request_end_ns": (i + 2) * 1_000_000_000,
                    "benchmark_phase": "profiling",
                    "was_cancelled": False,
                },
                "metrics": {
                    name: {"value": value, "unit": unit}
                    for name, value, unit in [
                        ("input_sequence_length", 100, "tokens"),
                        ("output_sequence_length", 50, "tokens"),
                        ("time_to_first_token", 20, "ms"),
                        ("request_latency", 1000, "ms"),
                        ("inter_token_latency", 10, "ms"),
                    ]
                },
                "error": None,
            }
        )
    (artifacts / "profile_export.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    (artifacts / "profile_export_aiperf.json").write_text(
        json.dumps(
            {
                "was_cancelled": False,
                "request_count": {"avg": 2},
                "benchmark_duration": 2,
                "metadata": {},
            }
        )
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(upstream),
        "PYTHONDONTWRITEBYTECODE": "1",
        "RESULT_DIR": str(tmp_path),
        "AGENTIC_OUTPUT_DIR": str(tmp_path),
        "RESULT_FILENAME": "result",
        "MODEL": "test-model",
        "MODEL_PREFIX": "test-model",
        "FRAMEWORK": "sglang",
        "PRECISION": "fp8",
        "TP": "1",
        "CONC": "1",
        "KV_OFFLOADING": "none",
        "RUNNER_TYPE": "mi355x",
        "IMAGE": "test/image",
        "RECIPE_FINGERPRINT": "b" * 64,
        "SPEC_DECODING": "none",
        "DISAGG": "false",
        "IS_MULTINODE": "false",
    }
    completed = subprocess.run(
        [sys.executable, "-m", "infx.results.agentic.process_agentic_result"],
        cwd=upstream,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    parsed = ResultParser.parse_inferencex_result(
        tmp_path / "result.json",
        framework="sglang",
        model="test-model",
        scenario="agentx",
    )
    assert parsed.success, parsed.errors
    assert parsed.throughput.completed_requests == 2
    assert parsed.throughput.output_throughput == pytest.approx(50)
    assert parsed.throughput.total_token_throughput == pytest.approx(150)
    assert parsed.latency.ttft_mean == pytest.approx(20)


def test_real_fixed_client_over_http(upstream, tmp_path):
    python = os.environ.get("MAGPIE_TEST_CLIENT_PYTHON")
    if not python:
        pytest.skip(
            "set MAGPIE_TEST_CLIENT_PYTHON to an isolated InferenceX client interpreter"
        )
    tokenizer = tmp_path / "tokenizer"
    prepare = subprocess.run(
        [
            python,
            "-c",
            """
import sys
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
vocab = {'[UNK]':0, '[PAD]':1, **{'word'+str(i):i+2 for i in range(200)}}
t = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
t.pre_tokenizer = pre_tokenizers.Whitespace()
PreTrainedTokenizerFast(tokenizer_object=t, unk_token='[UNK]', pad_token='[PAD]').save_pretrained(sys.argv[1])
""",
            str(tokenizer),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert prepare.returncode == 0, prepare.stderr
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, request["model"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for _ in range(2):
                self.wfile.write(b'data: {"choices":[{"text":" word2","index":0}]}\n\n')
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        support = Path(__file__).parents[2] / "Magpie/scripts/benchmark"
        body = f"source {shlex.quote(str(support / 'magpie_bench_common.sh'))}\n"
        body += (
            f"source {shlex.quote(str(support / 'magpie_bench_remote_compat.sh'))}\n"
        )
        body += shlex.join(
            [
                "magpie_run_benchmark_serving",
                "--model",
                "served-fixture",
                "--backend",
                "vllm",
                "--input-len",
                "8",
                "--output-len",
                "2",
                "--random-range-ratio",
                "1",
                "--num-prompts",
                "4",
                "--max-concurrency",
                "2",
                "--result-filename",
                "http-result",
                "--result-dir",
                str(tmp_path),
                "--tokenizer",
                str(tokenizer),
            ]
        )
        env = {
            **os.environ,
            "MAGPIE_INFERENCEX_ROOT": str(upstream),
            "MAGPIE_BENCHMARK_PYTHON": python,
            "BENCHMARK_BASE_URL": f"http://127.0.0.1:{server.server_port}",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PROFILE": "0",
        }
        completed = subprocess.run(
            ["bash", "-c", body],
            cwd=upstream,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        result = json.loads((tmp_path / "http-result.json").read_text())
        assert result["completed"] == 4
        assert result["total_output_tokens"] == 8
        assert requests and all(
            p == "/v1/completions" and m == "served-fixture" for p, m in requests
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
