"""Print one version's section of CHANGELOG.md, for the GitHub release notes, or with
--top print the version of the top entry. Fails when the section is missing or empty."""

from __future__ import annotations

import re
import sys
from pathlib import Path

HEADING = re.compile(r"^## \[([^\]]+)\]")


def main() -> None:
    lines = Path("CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    if sys.argv[1:] == ["--top"]:
        for line in lines:
            m = HEADING.match(line)
            if m:
                print(m.group(1))
                return
        sys.exit("CHANGELOG.md has no version entry")
    version = sys.argv[1]
    out, inside = [], False
    for line in lines:
        m = HEADING.match(line)
        if m:
            if inside:
                break
            inside = m.group(1) == version
            continue
        if inside:
            out.append(line)
    text = "\n".join(out).strip()
    if not text:
        sys.exit("CHANGELOG.md has no section for %s" % version)
    print(text)


if __name__ == "__main__":
    main()
