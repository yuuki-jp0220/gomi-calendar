import hashlib, json, os, re, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/official-sources.json"
STATUS = ROOT / "data/pdf-source-status.json"
OUTPUT = ROOT / "data/imported-collection-dates.json"
ASSET_DIR = ROOT / "temp_assets"
HEADERS = {"User-Agent": "IbarakiGarbageCalendar/1.0 (official municipal calendar importer)"}
KEYWORDS = ("ごみ", "ゴミ", "収集", "カレンダー", "日程", "分別", "地区", "資源")
EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".webp")
MAX_ASSET_BYTES = 20 * 1024 * 1024
MAX_ASSETS_PER_PAGE = 8

class GarbageSchedule(BaseModel):
    date: str = Field(description='YYYY-MM-DD または「毎週月・木」「第2・4水曜」など原文どおりのルール')
    garbage_type: str
    note: str = ""

class MunicipalityGarbageData(BaseModel):
    municipality: str
    area_name: str
    schedules: list[GarbageSchedule]

class ExtractionResult(BaseModel):
    areas: list[MunicipalityGarbageData]

def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def clean(value):
    return re.sub(r"\s+", " ", value or "").strip()

def fetch_page_assets(url):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=45000)
        except Exception:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass
        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    page_text = clean(soup.get_text(separator="\n", strip=True))
    links = {}
    for tag in soup.find_all(["a", "img"]):
        raw = tag.get("href") or tag.get("src") or tag.get("data-src") or ""
        if not raw:
            continue
        full = urljoin(url, raw)
        path = urlparse(full).path.lower()
        label = clean(tag.get_text(" ", strip=True)) if tag.name == "a" else clean(tag.get("alt", ""))
        relevant = any(k in (label + " " + unquote(path)) for k in KEYWORDS)
        if path.endswith(EXTENSIONS) and (relevant or path.endswith((".pdf",))):
            links[full] = label or Path(path).name

    downloaded = []
    session = requests.Session()
    session.headers.update(HEADERS)
    for asset_url, label in list(links.items())[:MAX_ASSETS_PER_PAGE]:
        try:
            response = session.get(asset_url, timeout=30, stream=True)
            response.raise_for_status()
            content = response.content
            if len(content) > MAX_ASSET_BYTES:
                continue
            suffix = Path(urlparse(asset_url).path).suffix.lower()
            if suffix not in EXTENSIONS:
                continue
            if suffix == ".pdf" and not content.startswith(b"%PDF"):
                continue
            if suffix in (".png", ".jpg", ".jpeg", ".webp") and not response.headers.get("Content-Type", "").startswith("image/"):
                continue
            digest = hashlib.sha256(content).hexdigest()[:16]
            path = ASSET_DIR / (digest + suffix)
            path.write_bytes(content)
            downloaded.append({"path": path, "url": asset_url, "label": label, "sha256": hashlib.sha256(content).hexdigest()})
            time.sleep(0.3)
        except Exception as exc:
            print("Asset download failed:", asset_url, str(exc)[:160])
    return page_text[:20000], downloaded

def parse_with_gemini(client, municipality, page_url, page_text, assets):
    uploads = []
    contents = []
    try:
        for asset in assets:
            uploaded = client.files.upload(file=str(asset["path"]))
            uploads.append(uploaded)
            contents.append(uploaded)
        prompt = f"""
あなたは自治体公式のごみ収集情報を正確に構造化する担当です。
自治体: {municipality}
公式ページ: {page_url}
以下のページ本文と添付PDF・画像から、明記されている地区ごとの収集予定を抽出してください。

出力スキーマは areas 配列です。各要素は municipality, area_name, schedules を持ち、
schedules の各要素は date, garbage_type, note を持ちます。

重要なルール:
- 年月日が明記されている予定だけ YYYY-MM-DD で出す。
- 「毎週月・木」「第1・3水曜」などの規則は、date欄に原文どおり保存する。日付に展開しない。
- 地区別情報は地区ごとに area_name を分ける。地区が不明なら「地区不明」。
- 年が分からない日付は推測しない。
- ごみ種別の表記は原資料を尊重する。
- 休止日・祝日・注意書きを収集予定として扱わない。
- 読み取れない情報は出力しない。重複させない。
- ページに関係のないリンク先や情報は採用しない。

ページ本文:
{page_text}

添付ファイル:
{json.dumps([{"url":a["url"],"label":a["label"]} for a in assets],ensure_ascii=False)}
"""
        contents.append(prompt)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult,
                temperature=0.1,
            ),
        )
        parsed = ExtractionResult.model_validate_json(response.text)
        return parsed.areas
    finally:
        for uploaded in uploads:
            try:
                client.files.delete(name=uploaded.name)
            except Exception as exc:
                print("Temporary Gemini file cleanup warning:", str(exc)[:120])

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is missing; extraction skipped.")
        return
    sources = read_json(SOURCES, [])
    old_status = read_json(STATUS, {"pdfs": []})
    old_by_url = {x.get("page_url"): x for x in old_status.get("pdfs", [])}
    existing = read_json(OUTPUT, {"events": [], "rules": []})
    events = existing.get("events", [])
    rules = existing.get("rules", [])
    client = genai.Client(api_key=api_key)
    status_rows = []

    for source in sources:
        municipality = source.get("municipality", "")
        page_url = source.get("url", "")
        row = {"municipality": municipality, "page_url": page_url}
        assets = []
        try:
            page_text, assets = fetch_page_assets(page_url)
            digest = hashlib.sha256((page_text + "".join(a["sha256"] for a in assets)).encode("utf-8")).hexdigest()
            row.update({"sha256": digest, "asset_count": len(assets)})
            old = old_by_url.get(page_url, {})
            if old.get("sha256") == digest and old.get("status") == "processed":
                row["status"] = "processed_unchanged"
                for asset in assets:
                    asset["path"].unlink(missing_ok=True)
                status_rows.append(row)
                continue
            extracted = parse_with_gemini(client, municipality, page_url, page_text, assets)
            # この公式ページ由来の旧データを置き換える
            events = [e for e in events if e.get("source_page") != page_url]
            rules = [e for e in rules if e.get("source_page") != page_url]
            added_dates, added_rules = 0, 0
            for area in extracted:
                for schedule in area.schedules:
                    date_value = clean(schedule.date)
                    kind = clean(schedule.garbage_type)
                    if not date_value or not kind:
                        continue
                    common = {
                        "municipality": municipality,
                        "district": clean(area.area_name) or "地区不明",
                        "type": kind,
                        "note": clean(schedule.note),
                        "source_page": page_url,
                        "verification": "要確認",
                    }
                    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_value):
                        try:
                            datetime.strptime(date_value, "%Y-%m-%d")
                        except ValueError:
                            continue
                        events.append({**common, "date": date_value})
                        added_dates += 1
                    else:
                        # ルール文はそのまま保持し、具体的な日付には変換しない
                        rules.append({**common, "rule": date_value})
                        added_rules += 1
            row.update({"status": "processed", "date_events": added_dates, "rule_events": added_rules,
                        "processed_at": datetime.now(timezone.utc).isoformat()})
            for asset in assets:
                try: asset["path"].unlink(missing_ok=True)
                except Exception: pass
        except Exception as exc:
            for asset in assets:
                try: asset["path"].unlink(missing_ok=True)
                except Exception: pass
            row.update({"status": "error", "error": str(exc)[:300]})
        status_rows.append(row)

    timestamp = datetime.now(timezone.utc).isoformat()
    STATUS.write_text(json.dumps({"checked_at": timestamp, "pdfs": status_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT.write_text(json.dumps({"updated_at": timestamp, "count": len(events), "events": events,
                                  "rules_count": len(rules), "rules": rules}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Imported date events:", len(events), "recurring rules:", len(rules))

if __name__ == "__main__":
    main()
