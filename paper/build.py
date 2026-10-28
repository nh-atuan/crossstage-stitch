"""Build and audit the anonymous working paper using installed TeX tools."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
TEMPLATE = REPO / "PARMA-DITAM_2027_Workshop"
TEMPLATE_FILES = ("oasics-v2021.cls", "cc-by.pdf", "oasics-logo-bw.pdf",
                  "orcid.pdf", "LICENSE.md")


def run(command, cwd, log, env=None):
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            encoding="utf-8", errors="replace", capture_output=True,
                            timeout=120)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command}; see {log}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "build" / "paper")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == PAPER or not output.is_relative_to((REPO / "build").resolve()):
        parser.error("output must be inside this repository's build/ directory")
    for tool in ("pdflatex", "bibtex", "kpsewhich", "pdfinfo", "pdftotext", "pdffonts"):
        if shutil.which(tool) is None:
            parser.error(f"required installed tool unavailable: {tool}")
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "verification.json"
    # A failed rerun must not leave a previous successful receipt in place.
    receipt = {"schema_version": "anonymous-paper-build-v1", "status": "failed"}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    try:
        env = os.environ.copy()
        # Reuse publisher sources verbatim; the trailing separator retains TeX defaults.
        env["TEXINPUTS"] = str(TEMPLATE) + os.pathsep + env.get("TEXINPUTS", "")
        env["BIBINPUTS"] = str(PAPER) + os.pathsep + env.get("BIBINPUTS", "")
        provenance = {}
        for name in TEMPLATE_FILES:
            original = TEMPLATE / name
            provenance[str(original.relative_to(REPO))] = hashlib.sha256(original.read_bytes()).hexdigest()
            if name != "LICENSE.md":
                resolved = run(["kpsewhich", name], PAPER, output / f"lookup-{name}.txt", env).strip()
                if Path(resolved).resolve() != original.resolve():
                    raise RuntimeError(f"TeX resolved {name} outside the repository template: {resolved}")
        command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                   "-file-line-error", "-no-shell-escape", f"-output-directory={output}", "main.tex"]
        run(command, PAPER, output / "pdflatex-1.txt", env)
        run(["bibtex", "main"], output, output / "bibtex.txt", env)
        run(command, PAPER, output / "pdflatex-2.txt", env)
        run(command, PAPER, output / "pdflatex-3.txt", env)
        log = (output / "main.log").read_text(encoding="utf-8", errors="replace")
        warnings = [line for line in log.splitlines()
                    if re.search(r"(?:LaTeX|Package .*?) Warning:|Overfull|Underfull|"
                                 r"(?:Citation|Reference).*undefined|undefined references", line, re.I)]
        if warnings:
            raise RuntimeError("final TeX log has unresolved diagnostics: " + " | ".join(warnings))
        pdf = output / "main.pdf"
        metadata = run(["pdfinfo", str(pdf)], PAPER, output / "pdfinfo.txt")
        text = run(["pdftotext", "-layout", str(pdf), "-"], PAPER, output / "paper-text.txt")
        fonts = run(["pdffonts", str(pdf)], PAPER, output / "pdffonts.txt")
        author = re.search(r"^Author:\s*(.*)$", metadata, re.M)
        if author is None or author.group(1).strip() != "Anonymous Authors":
            raise RuntimeError("PDF author metadata is not the expected anonymous value")
        if re.search(r"HCMUS|crossstage-stitch[/\\]|nh-atuan|@gmail|@hcmus", metadata + text, re.I):
            raise RuntimeError("PDF text/metadata contains a known identifying token")
        font_lines = [line for line in fonts.splitlines()[2:] if line.strip()]
        if not font_lines or any(re.search(r"\sno\s+(?:yes|no)\s+(?:yes|no)\s+\d+", line)
                                 for line in font_lines):
            raise RuntimeError("at least one PDF font is not embedded")
        receipt.update(status="passed", pages=int(re.search(r"^Pages:\s*(\d+)", metadata, re.M)[1]),
                       author=author.group(1).strip(), warnings=warnings,
                       template_provenance_sha256=provenance,
                       pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                       source_sha256={str(p.relative_to(PAPER)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(PAPER.rglob("*")) if p.is_file()
                                      and p.suffix in (".tex", ".bib", ".cls", ".py")},
                       limits="Build/anonymity audit only; no completed experimental claims or visual acceptance implied")
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        receipt["error"] = str(exc)
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "pages": receipt["pages"],
                      "author": receipt["author"], "output": str(pdf)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
