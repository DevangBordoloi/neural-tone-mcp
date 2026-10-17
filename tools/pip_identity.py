"""Run pip with Accept-Encoding: identity (works around Norton's mangled gzip/chunked PyPI index replies).

  python tools/pip_identity.py --python .venv/Scripts/python.exe install <packages>
"""
import sys

import pip._vendor.requests.sessions as _s
from pip._internal.cli.main import main

_orig = _s.default_headers
_s.default_headers = lambda: {**_orig(), "Accept-Encoding": "identity"}
sys.exit(main(sys.argv[1:]))
