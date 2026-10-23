"""Check a deployed API without printing or storing its credential.

NEXUS_BASE_URL=https://your-service NEXUS_API_KEY=... python scripts/check_production.py
"""

import json
import os
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def main():
    base = os.environ.get("NEXUS_BASE_URL", "").rstrip("/")
    parsed = urlsplit(base)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path
    ):
        raise SystemExit("Set NEXUS_BASE_URL to the HTTPS service origin")
    key = os.environ.get("NEXUS_API_KEY")
    if not key:
        raise SystemExit("Set NEXUS_API_KEY to a client credential")

    def request(path, payload=None, authenticated=False):
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {key}"
        req = Request(
            base + path,
            headers=headers,
            data=json.dumps(payload).encode() if payload is not None else None,
        )
        # Refuse redirects so Authorization never reaches a different host.
        from urllib.request import HTTPRedirectHandler, build_opener

        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        with build_opener(NoRedirect).open(req, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError(f"{path}: HTTP {response.status}")
            result = json.load(response)
        print(f"PASS {path}", flush=True)
        return result

    from urllib.error import HTTPError

    request("/health")
    request("/ready")
    request("/openapi.json")
    try:
        urlopen(
            Request(
                base + "/chat",
                data=b'{"message":"hi"}',
                headers={"Content-Type": "application/json"},
            ),
            timeout=20,
        )
    except HTTPError as error:
        if error.code != 401:
            raise RuntimeError("Unauthenticated chat must return 401") from None
    else:
        raise RuntimeError("Unauthenticated chat unexpectedly succeeded")
    print("PASS unauthorized chat blocked", flush=True)
    result = request(
        "/chat", {"message": "What is 2 + 2? Answer briefly.", "max_tokens": 32}, authenticated=True
    )
    if not isinstance(result.get("response"), str) or not result["response"].strip():
        raise RuntimeError("Chat returned an empty response")
    print("PASS real model response")


if __name__ == "__main__":
    main()
