"""Make Python's TLS use the operating-system certificate store.

Needed on machines where antivirus (e.g. Norton Web Shield) re-signs HTTPS traffic with a root certificate
that Windows trusts but Python's bundled CA list does not. Verification stays ON.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)
_done = False


def use_system_trust_store() -> None:
    global _done
    if _done:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception as e:  # truststore missing: fall back to Python's default CA bundle
        log.debug("truststore unavailable: %s", e)
    _done = True
