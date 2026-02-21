from __future__ import annotations

import sys

from huweibot.main import main as _new_main


_DEPRECATION_MSG = "[deprecation] xbot is deprecated, use huweibot."


def main(argv: list[str] | None = None) -> int:
    print(_DEPRECATION_MSG, file=sys.stderr)
    return int(_new_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
