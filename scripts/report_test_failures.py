"""Expose all failed case names within GitHub's per-step annotation budget."""

import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path


def annotation(text):
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main():
    path = Path(sys.argv[1])
    if not path.is_file():
        print("No pytest XML was produced; inspect the failed preceding step.")
        return
    groups = OrderedDict()
    names = []
    for case in ET.parse(path).iter("testcase"):
        for kind in ("failure", "error"):
            failure = case.find(kind)
            if failure is None:
                continue
            group = case.get("classname", "")
            name = group + "." + case.get("name", "")
            detail = failure.text or failure.get("message", "")
            names.append(name)
            groups.setdefault(group, (name, detail))
            print(name + "\n" + detail)
    if not names:
        return
    summary = f"{len(names)} failed/error cases:\n" + "\n".join(names)
    if len(summary) > 56000:
        summary = summary[:56000] + "\n[case list truncated; full XML/log retained]"
    print("::error::" + annotation(summary))
    # One summary plus at most six distinct-module examples stays below the
    # ten-error-per-step limit. Previously early parameter cases hid all later
    # modules from the anonymous annotations API.
    for name, detail in list(groups.values())[:6]:
        bounded = detail if len(detail) <= 12000 else detail[:6000] + "\n...\n" + detail[-6000:]
        print("::error::" + annotation(name + "\n" + bounded))


if __name__ == "__main__":
    main()
