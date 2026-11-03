"""Expose public CI failures as annotations when raw Actions logs require login."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def annotation(text):
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main():
    path = Path(sys.argv[1])
    if not path.is_file():
        print("No pytest XML was produced; inspect the failed preceding step.")
        return
    tree = ET.parse(path)
    for case in tree.iter("testcase"):
        for kind in ("failure", "error"):
            failure = case.find(kind)
            if failure is not None:
                name = case.get("classname", "") + "." + case.get("name", "")
                print(
                    "::error::"
                    + annotation(name + "\n" + (failure.text or failure.get("message", "")))
                )


if __name__ == "__main__":
    main()
