"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    return re.sub(r"-+", "-", re.sub(r"[^\w]+", "-", topic.strip().lower(), flags=re.UNICODE)).strip("-")[:60] or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (f"Research topic: {topic}\n\nFollow your system instructions. Produce a sourced English survey report in "
            "the required workspace paths. Use parallel researchers, verify claims, then finalize and validate citations.")


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    counts, tokens, subagent_calls = Counter(), {"input": 0, "output": 0}, 0
    for message in messages:
        calls = getattr(message, "tool_calls", None)
        if calls is None and isinstance(message, dict):
            calls = message.get("tool_calls", [])
        for call in calls or []:
            name = call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
            if name:
                counts[name] += 1
                subagent_calls += name == "task"
        usage = getattr(message, "usage_metadata", None)
        if usage is None and isinstance(message, dict):
            usage = message.get("usage_metadata")
        if usage:
            tokens["input"] += usage.get("input_tokens", 0)
            tokens["output"] += usage.get("output_tokens", 0)
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": subagent_calls,
            "tool_calls": dict(counts), "tokens": tokens}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes, sources_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("sandbox report is missing or empty")
    if not sources_bytes:
        raise RuntimeError("sandbox sources.json is missing")
    try:
        sources = json.loads(sources_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"sandbox sources.json is invalid: {exc}") from exc
    if not isinstance(sources, list) or not sources or not all(isinstance(row, dict) for row in sources):
        raise RuntimeError("sandbox sources.json must be a non-empty list of objects")
    stem = slugify(topic)
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(messages, elapsed, model_name)
    meta = {"topic": topic, **summary, "n_sources": len(sources),
            "source_families": sorted({str(row.get("source")) for row in sources if row.get("source")})}
    outputs = {reports_dir / f"{stem}.md": report_bytes,
               reports_dir / f"{stem}.sources.json": json.dumps(sources, ensure_ascii=False, indent=2).encode(),
               reports_dir / f"{stem}.meta.json": json.dumps(meta, ensure_ascii=False, indent=2).encode()}
    for path, content in outputs.items():
        path.write_bytes(content)
    return reports_dir / f"{stem}.md"


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    if not topic.strip():
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    try:
        model = make_model()
        model_name = getattr(model, "model", None) or getattr(model, "model_name", None) or os.getenv("LAB_MODEL", "unknown")
        start = time.monotonic()
        with open_sandbox() as backend:
            response = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if getattr(response, "exit_code", 0) != 0:
                raise RuntimeError(f"cannot prepare sandbox workspace: {getattr(response, 'output', '')}")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
            agent = build_lead_agent(backend, model)
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                  config={"recursion_limit": 1000})
            elapsed = time.monotonic() - start
            report_path = save_outputs(backend, topic, result.get("messages", []), elapsed, model_name)
        print(f"Saved report: {report_path}")
        return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
