---
myst:
    html_meta:
        "description": "Drive Magpie from AI agents using the MCP server or the Magpie agent skill for Cursor, Claude Code, and Codex. Covers installation, configuration, and verification."
        "keywords": "Magpie, MCP server, agent skills, Claude Code, Cursor, Codex, Model Context Protocol, AI agents, GPU evaluation"
---

# Run MCP server and agent skills with Magpie

Magpie can be driven by AI agents in two ways: the Model Context Protocol (MCP) server, which exposes the full Magpie toolset to any MCP-compatible client, or the Magpie agent skill, which gives editors without MCP support access to the same capabilities through documented CLI patterns. The MCP server is the preferred integration for programmatic use, remote execution, and Ray-based workflows, while the skill is a drop-in option for Cursor, Claude Code, Codex, and similar editors. Both approaches let agents analyze kernels, run benchmarks, query GPU hardware, and trigger gap analysis without leaving the IDE.

## Run the MCP server

The MCP server exposes Magpie's evaluation capabilities (analyze, compare,
benchmark, gap analysis, hardware queries, and more) as MCP tools. Start it
with:

```bash
python -m Magpie.mcp
```

A sample client configuration is provided at `Magpie/mcp/config.json`. For the
full list of tools and their parameters, see the
[API reference](../reference/api-reference.md).

## Verify the MCP server

After starting the server, run these checks to confirm it is reachable and functional before connecting a client.

### Health endpoint

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status": "ok"}
```

If the request times out or is refused, the server is not running. Check that `python -m Magpie.mcp` is still running in another terminal, or that `MAGPIE_HOST`/`MAGPIE_PORT` environment variables match what you are querying.

### First tool call

Use any MCP client to call the `hardware_spec` tool with no arguments. It queries the local GPU and requires no kernel files, making it the simplest smoke test:

```json
{"tool": "hardware_spec", "arguments": {}}
```

A successful response includes `vendor`, `device_count`, and per-device `name` and `memory_total_gb` fields. An error here usually means ROCm or CUDA is not available in the server's environment—check that `rocminfo` or `nvidia-smi` runs in the same shell.

## Install the Magpie skill

The Magpie skill lets AI agents (Cursor, Claude Code, Codex, and similar editors) drive Magpie through documented CLI patterns when MCP is not available. The skill lives in this repo under **`skills/magpie/`** (IDE-neutral). Install it into each editor’s skills directory with the script below, or follow the manual copy steps.

## Install script (recommended)

From the Magpie repository root:

```bash
# Optional: make the script executable once
chmod +x skills/install-skill.sh

# Global install (default): user home, e.g. ~/.cursor/skills/magpie
./skills/install-skill.sh cursor
./skills/install-skill.sh claude
./skills/install-skill.sh codex
./skills/install-skill.sh all

# Explicit global scope (same as omitting the flag)
./skills/install-skill.sh cursor --global

# Project scope: current working directory
./skills/install-skill.sh cursor --project

# Project scope: specific directory
./skills/install-skill.sh claude --project /path/to/your/project

# Help
./skills/install-skill.sh -h
```

The script removes any existing destination folder named `magpie` and copies `skills/magpie` there. No editor restart is usually required.

Review these topics for more information:
- [Analyze and compare kernels with Magpie](analyze-compare.md)
- [Benchmark frameworks with Magpie](benchmarking/benchmark.md)
- [README](https://github.com/AMD-AGI/Magpie#readme) (MCP vs skill).

## Manual install

Source folder: **`skills/magpie/`** in this repo (contains `SKILL.md`, `reference.md`, `examples.md`).

### Cursor

- **Global**: Copy the skill into your Cursor skills folder:
  ```bash
  cp -r /path/to/Magpie/skills/magpie ~/.cursor/skills/magpie
  ```
- **Project**: Copy into your project’s Cursor skills folder so only that project uses it:
  ```bash
  mkdir -p /path/to/your/project/.cursor/skills
  cp -r /path/to/Magpie/skills/magpie /path/to/your/project/.cursor/skills/magpie
  ```
  (Replace `/path/to/Magpie` and `/path/to/your/project` with the actual paths.)

### Claude Code

Same `SKILL.md` format and layout.

- **Global**:
  ```bash
  mkdir -p ~/.claude/skills
  cp -r /path/to/Magpie/skills/magpie ~/.claude/skills/magpie
  ```
- **Project**:
  ```bash
  mkdir -p /path/to/your/project/.claude/skills
  cp -r /path/to/Magpie/skills/magpie /path/to/your/project/.claude/skills/magpie
  ```

### Codex and other IDEs

- If the IDE has a `skills` or `custom instructions` directory (for example, `~/.codex/skills/` or a project `.codex/skills/`), copy the **`magpie`** directory there:
  ```bash
  mkdir -p ~/.codex/skills
  cp -r /path/to/Magpie/skills/magpie ~/.codex/skills/magpie
  ```
  Use the path your IDE documents for custom skills.
- If the IDE only has a single `custom instructions` or `rules` field, paste the body of [skills/magpie/SKILL.md](https://github.com/AMD-AGI/Magpie/blob/main/skills/magpie/SKILL.md) (the markdown after the YAML frontmatter) into that field, and add a note that these instructions apply when working with Magpie, GPU kernel analysis and comparison, or vLLM and SGLang benchmarks.

## Verify the skill

1. In the target IDE, ask: “What can you do with Magpie?” or “I want to analyze a HIP kernel with Magpie.” The agent should use the Magpie skill (reference its instructions or run Magpie commands).
2. Ask “Show GPU info using Magpie.” The agent should run `magpie --gpu-info` or `python -m Magpie --gpu-info` from the Magpie repo (or from a directory where Magpie is installed).
3. From the Magpie repo root, run `magpie --gpu-info` (or `python -m Magpie --gpu-info`) to confirm the CLI and environment work.
