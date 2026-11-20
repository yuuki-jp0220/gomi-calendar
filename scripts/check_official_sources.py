import hashlib
import json
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "official-sources.json"
OUTPUT = ROOT / "data" / "source-status.json"

def fetch(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "IbarakiGarbageCalendarBot/1.0 (official-source change monitor)"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
        return {
            "status": response.status,
            "final_url": response.geturl(),
            "content_type": response.headers.get("Content-Type", ""),
            "last_modified": response.headers.get("Last-Modified"),
            "etag": response.headers.get("ETag"),
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
        }

def main():
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    previous = {}
    if OUTPUT.exists():
        try:
            previous = {x["url"]: x for x in json.loads(OUTPUT.read_text(encoding="utf-8")).get("sources", [])}
        except Exception:
            previous = {}

    checked_at = datetime.now(timezone.utc).isoformat()
    results = []
    for source in sources:
        item = dict(source)
        try:
            current = fetch(source["url"])
            old = previous.get(source["url"], {})
            item.update(current)
            item["check"] = "changed" if old.get("sha256") and old.get("sha256") != current["sha256"] else "ok"
        except Exception as exc:
            item["check"] = "error"
            item["error"] = str(exc)[:300]
        item["checked_at"] = checked_at
        results.append(item)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps({"checked_at": checked_at, "sources": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Checked {len(results)} official source(s)")

if __name__ == "__main__":
    main()
