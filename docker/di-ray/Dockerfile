# Build from the Magpie repository root with --platform linux/amd64.
FROM ghcr.io/astral-sh/uv:0.9.9@sha256:2fdbcf58c1339b1cbc6deba77bd0540df11a20dbba1df75f21c0bf959afccc4e AS uv
FROM lmsysorg/sglang:v0.5.18-rocm720-mi35x@sha256:6d68cd19206716cb3f1e31e2ad89cd0852d7ae614a792773c30a4277f8955c72
COPY --from=uv /uv /uvx /usr/local/bin/
ENV PATH="/opt/venv/bin:${PATH}" UV_NATIVE_TLS=true PYTHONUNBUFFERED=1 \
    RAY_USAGE_STATS_ENABLED=0 HF_HOME=/shared/huggingface \
    MAGPIE_BENCHMARK_PYTHON=/opt/magpie-client/bin/python

# The vendor ships unowned Mooncake binaries; replace them with the ROCm wheel.
ADD --checksum=sha256:f5a3725bae80917ea50a78fe45931605587e2c94946844d6ee2f2c3094d56d22 https://files.pythonhosted.org/packages/2e/04/814fa24102e94fe25c3f551561afe233f4ef847b4ada8d38d0ebf72b356c/mooncake_transfer_engine_rocm-0.3.14-cp310-cp310-manylinux_2_35_x86_64.whl /tmp/mooncake_transfer_engine_rocm-0.3.14-cp310-cp310-manylinux_2_35_x86_64.whl
COPY docker/di-ray/install-mooncake.py /tmp/install-mooncake.py
RUN python /tmp/install-mooncake.py && rm /tmp/install-mooncake.py /tmp/mooncake_transfer_engine_rocm-0.3.14-cp310-cp310-manylinux_2_35_x86_64.whl
COPY docker/di-ray/engine-requirements.txt /tmp/engine-requirements.txt
RUN uv pip install --python /opt/venv/bin/python -r /tmp/engine-requirements.txt

# InferenceX uses Python 3.12 independently of the serving image's Python 3.10.
ADD --checksum=sha256:1ca5a1a0f6505cfd16c37285ecf0b60040bfb1e4edcff6e719f385b34952ebb9 https://github.com/astral-sh/python-build-standalone/releases/download/20251031/cpython-3.12.12%2B20251031-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz /tmp/python312.tar.gz
COPY docker/di-ray/client-requirements.txt /tmp/client-requirements.txt
RUN mkdir -p /opt/python \
 && tar -xzf /tmp/python312.tar.gz --strip-components=1 -C /opt/python \
 && uv venv --python /opt/python/bin/python3.12 /opt/magpie-client \
 && uv pip install --python /opt/magpie-client/bin/python -r /tmp/client-requirements.txt \
 && rm /tmp/python312.tar.gz
ADD --checksum=sha256:99dcba44a2214622da629babd29b427b42c71b46cbda6b6a7c06eccf0a5d8b78 https://codeload.github.com/SemiAnalysisAI/InferenceX/tar.gz/408c015be4b22d14c69518643609669405507077 /tmp/inferencex.tar.gz
RUN mkdir -p /opt/InferenceX \
 && tar -xzf /tmp/inferencex.tar.gz --strip-components=2 -C /opt/InferenceX InferenceX-408c015be4b22d14c69518643609669405507077/inferencex-e2e \
 && rm /tmp/inferencex.tar.gz
COPY pyproject.toml README.md LICENSE /opt/Magpie/
COPY Magpie /opt/Magpie/Magpie
RUN uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation /opt/Magpie \
 && cp /opt/Magpie/Magpie/scripts/benchmark/*.sh /opt/InferenceX/benchmarks/ \
 && python -c "import ray, torch, mooncake.engine, sglang_router.launch_router; from Magpie.di import DISession; assert ray.__version__ == '2.55.1'; assert torch.__version__ == '2.9.1+rocm7.2.0.git7e1940d4'; import lm_eval" \
 && cd /opt/InferenceX && /opt/magpie-client/bin/python -c "from infx.bench_serving import benchmark_serving"
WORKDIR /shared
ENTRYPOINT []
CMD ["bash"]
