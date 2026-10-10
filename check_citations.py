"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]
    problems, by_number, urls = [], {}, set()
    for source in sources:
        if not isinstance(source, dict):
            problems.append("source entry must be an object")
            continue
        number, url = source.get("n"), source.get("url")
        if not isinstance(number, int) or isinstance(number, bool):
            problems.append(f"source n={number!r} is not an integer")
        elif number in by_number:
            problems.append(f"source number [{number}] is duplicated")
        else:
            by_number[number] = source
        if not isinstance(url, str) or not re.match(r"^https?://", url):
            problems.append(f"source [{number}] has an invalid URL")
        elif url in urls:
            problems.append(f"duplicate URL: {url}")
        else:
            urls.add(url)
        family, paper_id = source.get("source"), source.get("id")
        expected = (f"https://arxiv.org/abs/{paper_id}" if family == "arxiv" else
                    f"https://huggingface.co/papers/{paper_id}" if family in {"hf-daily", "hf-search"} else None)
        if family not in {"arxiv", "hf-daily", "hf-search", "web"}:
            problems.append(f"source [{number}] has an invalid source family")
        elif expected and url != expected:
            problems.append(f"source [{number}] URL does not match its {family} family and id")

    heading = re.search(r"(?m)^##[ \t]+References[ \t]*$", report_text)
    if not heading:
        return problems + ["missing ## References section"]
    body, reference_text = report_text[:heading.start()], report_text[heading.end():]
    code_or_link = re.compile(r"(```.*?```|`[^`\n]*`|\[[^\]]*\]\([^)]*\))", re.DOTALL)
    body = code_or_link.sub("", body)
    citation = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")
    cited = set()
    for match in citation.finditer(body):
        for part in re.split(r"\s*,\s*", match.group(1)):
            span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
            cited.update(range(int(span[1]), int(span[2]) + 1) if span and int(span[2]) >= int(span[1]) and int(span[2]) - int(span[1]) <= 200 else ([int(span[1]), int(span[2])] if span else [int(part)]))
    for number in sorted(cited - by_number.keys()):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(by_number.keys() - cited):
        problems.append(f"source [{number}] never cited")

    refs = {}
    for line in reference_text.splitlines():
        match = re.match(r"^\[(\d+)\](.*)$", line.strip())
        if not match:
            continue
        number, tail = int(match[1]), match[2]
        refs.setdefault(number, []).append(tail)
    for number, entries in refs.items():
        if len(entries) != 1:
            problems.append(f"reference [{number}] appears more than once")
    for number in sorted(by_number.keys() - refs.keys()):
        problems.append(f"reference [{number}] is missing")
    for number in sorted(refs.keys() - by_number.keys()):
        problems.append(f"reference [{number}] has no matching source")
    for number, entries in refs.items():
        if len(entries) != 1 or number not in by_number:
            continue
        found = re.findall(r"https?://[^\s<>]+", entries[0])
        if len(found) != 1:
            problems.append(f"reference [{number}] must contain exactly one URL")
        elif found[0].rstrip(".,;)") != by_number[number].get("url"):
            problems.append(f"reference [{number}] URL does not match sources.json")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
