import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import fitz
import pdfplumber
import requests
from bs4 import BeautifulSoup
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/official-sources.json"
STATUS = ROOT / "data/pdf-source-status.json"
OUTPUT = ROOT / "data/imported-collection-dates.json"
ASSET_DIR = ROOT / "temp_assets"
HEADERS = {"User-Agent": "IbarakiGarbageCalendar/1.0 (official municipal calendar importer)"}
EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".webp")
MAX_ASSET_BYTES = 20 * 1024 * 1024
MAX_ASSETS_PER_PAGE = 8

# 小規模な文字分類モデル。日程の抽出自体は原文の行・表記を優先する。
TRAINING = [
    ("燃えるごみ", "可燃ごみ"), ("燃やすごみ", "可燃ごみ"), ("可燃ごみ", "可燃ごみ"),
    ("燃えないごみ", "不燃ごみ"), ("燃やさないごみ", "不燃ごみ"), ("不燃ごみ", "不燃ごみ"), ("不燃ごみ（プラスチック類）", "不燃ごみ"), ("不燃ごみ（金属類・割れ物）", "不燃ごみ"),
    ("資源ごみ", "資源ごみ"), ("資源物", "資源ごみ"), ("びん", "資源ごみ"), ("ビン", "資源ごみ"),
    ("かん", "資源ごみ"), ("缶", "資源ごみ"), ("ペットボトル", "資源ごみ"),
    ("古紙", "資源ごみ"), ("紙", "資源ごみ"), ("布類", "資源ごみ"), ("新聞紙", "資源ごみ"), ("段ボール", "資源ごみ"),
    ("プラスチック", "プラスチック"), ("容器包装プラ", "プラスチック"),
    ("粗大ごみ", "粗大ごみ"), ("有害ごみ", "有害ごみ"), ("危険ごみ", "有害ごみ"),
]
class GarbageBuiltInAI:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
        texts, labels = zip(*TRAINING)
        self.matrix = self.vectorizer.fit_transform(texts)
        self.model = MultinomialNB().fit(self.matrix, labels)
        self.examples = [x for x, _ in TRAINING]

    def classify(self, text):
        text = clean(text)
        if not text:
            return text, 0.0, True
        # 短い・未知の語は無理にカテゴリ変更しない
        if not any(term in text for term in self.examples):
            return text, 0.0, True
        vec = self.vectorizer.transform([text])
        probs = self.model.predict_proba(vec)[0]
        best = int(probs.argmax())
        confidence = float(probs[best])
        if confidence < 0.72:
            return text, confidence, True
        return str(self.model.classes_[best]), confidence, False

def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()

def fetch_page_assets(url):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    response = session.get(url, timeout=35)
    response.raise_for_status()
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    page_text = "\n".join(clean(x) for x in soup.stripped_strings if clean(x))
    links = {}
    for tag in soup.find_all(["a", "img"]):
        raw = tag.get("href") or tag.get("src") or tag.get("data-src") or ""
        if not raw:
            continue
        full = urljoin(url, raw)
        path = urlparse(full).path.lower()
        label = clean(tag.get_text(" ", strip=True)) if tag.name == "a" else clean(tag.get("alt", ""))
        if path.endswith(EXTENSIONS):
            links[full] = label or Path(path).name

    downloaded = []
    for asset_url, label in list(links.items())[:MAX_ASSETS_PER_PAGE]:
        try:
            r = session.get(asset_url, timeout=40)
            r.raise_for_status()
            content = r.content
            if len(content) > MAX_ASSET_BYTES:
                continue
            suffix = Path(urlparse(asset_url).path).suffix.lower()
            if suffix == ".pdf" and not content.startswith(b"%PDF"):
                continue
            if suffix in (".png", ".jpg", ".jpeg", ".webp") and not r.headers.get("Content-Type", "").startswith("image/"):
                continue
            digest = hashlib.sha256(content).hexdigest()
            path = ASSET_DIR / (digest[:16] + suffix)
            path.write_bytes(content)
            downloaded.append({"path": path, "url": asset_url, "label": label, "sha256": digest})
            time.sleep(0.15)
        except Exception as exc:
            print("Asset download failed:", asset_url, str(exc)[:120])
    # JavaScript描画型ページだけPlaywrightで補完。通常ページはrequestsを優先して高速化。
    if len(page_text) < 80:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=25000)
                page_text = "\n".join(x.strip() for x in page.locator("body").inner_text(timeout=10000).splitlines() if x.strip())
                browser.close()
        except Exception as exc:
            print("Dynamic page fallback:", str(exc)[:120])
    return page_text[:50000], downloaded


def fetch_linked_calendar_text(url):
    """守谷市などの公式テキスト版カレンダーから月別日程を取得する。"""
    session = requests.Session()
    session.headers.update(HEADERS)
    response = session.get(url, timeout=35)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    text_link = ""
    for link in soup.find_all("a", href=True):
        label = clean(link.get_text(" ", strip=True))
        if "テキスト版" in label and "クリーンカレンダー" in label:
            text_link = urljoin(url, link["href"])
            break
    if not text_link:
        return ""

    index_response = session.get(text_link, timeout=35)
    index_response.raise_for_status()
    index_soup = BeautifulSoup(index_response.text, "html.parser")
    index_text = clean(index_soup.get_text(" ", strip=True))
    year_match = re.search(r"(20\d{2})年度", index_text)
    if not year_match:
        return ""
    fiscal_year = int(year_match.group(1))

    lines = []
    for link in index_soup.find_all("a", href=True):
        label = clean(link.get_text(" ", strip=True))
        month_match = re.search(r"(1[0-2]|[1-9])月クリーンカレンダー", label)
        if not month_match:
            continue
        month = int(month_match.group(1))
        month_url = urljoin(text_link, link["href"])
        try:
            month_response = session.get(month_url, timeout=30)
            month_response.raise_for_status()
            month_soup = BeautifulSoup(month_response.text, "html.parser")
            body = month_soup.find("main") or month_soup.find(id="contents") or month_soup.body or month_soup
            body_text = body.get_text("\n", strip=True)
            calendar_year = fiscal_year + (1 if month <= 3 else 0)
            lines.append("【公式月別ページ " + month_url + "】")
            for raw_line in body_text.splitlines():
                line = clean(raw_line)
                match = re.match(r"^(\d{1,2})日[（(]([^）)]+)[）)]\s*(.*)$", line)
                if not match:
                    continue
                day, weekday, details = match.groups()
                try:
                    date_text = datetime(calendar_year, month, int(day)).strftime("%Y年%m月%d日")
                except ValueError:
                    continue
                lines.append(f"{date_text}（{weekday}）{details}")
        except Exception as exc:
            print("Monthly text page failed:", month_url, str(exc)[:120])
    return "\n".join(lines)

def extract_asset_text(asset, ocr):
    path = asset["path"]
    suffix = path.suffix.lower()
    chunks = []
    if suffix == ".pdf":
        try:
            with pdfplumber.open(str(path)) as pdf:
                for page_no, page in enumerate(pdf.pages, 1):
                    text = page.extract_text(layout=False) or ""
                    if text.strip():
                        chunks.append((page_no, text))
        except Exception as exc:
            print("pdfplumber warning:", str(exc)[:120])
        try:
            doc = fitz.open(str(path))
            text_page_numbers = {page_no for page_no, _ in chunks}
            for i, page in enumerate(doc):
                if i + 1 in text_page_numbers:
                    continue
                pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
                img_path = ASSET_DIR / ("ocr_" + str(i) + ".png")
                pix.save(str(img_path))
                result, _ = ocr(str(img_path))
                text = "\n".join(x[1] for x in result) if result else ""
                img_path.unlink(missing_ok=True)
                if text.strip():
                    chunks.append((i + 1, text))
            doc.close()
        except Exception as exc:
            print("PDF OCR warning:", str(exc)[:160])
    else:
        try:
            result, _ = ocr(str(path))
            text = "\n".join(x[1] for x in result) if result else ""
            if text.strip():
                chunks.append((1, text))
        except Exception as exc:
            print("Image OCR warning:", str(exc)[:120])
    return chunks

DATE_PATTERNS = [
    re.compile(r"(?<!\d)(20\d{2})[年./-](\d{1,2})[月./-](\d{1,2})日?"),
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"),
]
WEEKDAY_RULE = re.compile(r"(?:毎週|週\d回|第[1-5一二三四五]?[・、,と及び]*[1-5一二三四五]?)(?:月|火|水|木|金|土|日)(?:曜(?:日)?)?(?:[・、,／/と](?:月|火|水|木|金|土|日)(?:曜(?:日)?)?)*")
GARBAGE_TERMS = [
    "燃えるごみ", "燃やすごみ", "可燃ごみ", "燃えないごみ", "燃やさないごみ", "不燃ごみ",
    "資源ごみ", "資源物", "びん", "かん", "缶", "ペットボトル", "古紙", "新聞紙", "段ボール",
    "プラスチック", "容器包装プラ", "粗大ごみ", "有害ごみ", "危険ごみ"
]

def explicit_dates(line):
    found = []
    for pat in DATE_PATTERNS:
        for m in pat.finditer(line):
            try:
                if len(m.groups()) == 3:
                    y, mo, d = map(int, m.groups())
                else:
                    raw = m.group(1)
                    y, mo, d = int(raw[:4]), int(raw[4:6]), int(raw[6:8])
                value = datetime(y, mo, d).strftime("%Y-%m-%d")
                if value not in found:
                    found.append(value)
            except Exception:
                pass
    return found

def terms_in(line):
    # 公式テキスト版の「ビン、ペットボトル」のような列挙を品目ごとに分ける。
    # 括弧内の補足は残し、分類モデルにはその品目全体を渡す。
    found = []
    for part in re.split(r"[、,，／/]", clean(line)):
        part = clean(part)
        if not part:
            continue
        matches = [term for term in GARBAGE_TERMS if term in part]
        if matches:
            candidate = part[:100]
            if candidate not in found:
                found.append(candidate)
    return found

def parse_text(municipality, page_url, page_text, asset_texts, ai):
    events, rules, review = [], [], []
    source_units = [("ページ本文", 0, page_text)]
    for asset in asset_texts:
        for page_no, txt in asset["pages"]:
            source_units.append((asset["url"], page_no, txt))
    seen = set()
    for source_name, page_no, text in source_units:
        lines = [clean(x) for x in text.splitlines() if clean(x)]
        for i, line in enumerate(lines):
            # ひとつの行の中だけで日付と品目を結び付ける。前後行の安易な結合はしない。
            dates = explicit_dates(line)
            terms = terms_in(line)
            rule_match = WEEKDAY_RULE.search(line)
            if not dates and not rule_match:
                continue
            # 「無し」はごみの品目ではなく「収集なし」として日程に記録する。
            # 同じ行に実際の品目があれば、その品目を優先して通常どおり抽出する。
            no_collection = re.search(r"(?:収集なし|無し|なし)", line)
            if not terms and no_collection:
                if len(dates) == 1:
                    key = (dates[0], "収集なし", source_name, page_no, line)
                    if key not in seen:
                        events.append({
                            "municipality": municipality,
                            "district": "地区不明",
                            "type": "収集なし",
                            "raw_type": no_collection.group(0),
                            "date": dates[0],
                            "note": line[:300],
                            "source_page": page_url,
                            "source_asset": source_name if source_name != "ページ本文" else "",
                            "source_page_number": page_no or None,
                            "verification": "要確認",
                            "engine": "BuiltIn-Lightweight-AI (v1.0)",
                            "confidence": 1.0,
                        })
                        seen.add(key)
                continue
            if len(terms) > 1 and len(dates) == 1:
                for raw_type in terms:
                    normalized, confidence, uncertain = ai.classify(raw_type)
                    common_multi = {
                        "municipality": municipality,
                        "district": "地区不明",
                        "type": normalized or raw_type,
                        "note": line[:300],
                        "source_page": page_url,
                        "source_asset": "",
                        "source_page_number": None,
                        "verification": "要確認",
                        "engine": "BuiltIn-Lightweight-AI (v1.0)",
                    }
                    if not uncertain:
                        key = (dates[0], normalized, source_name, page_no, line, raw_type)
                        if key not in seen:
                            events.append({**common_multi, "date": dates[0], "raw_type": raw_type, "confidence": round(confidence, 3)})
                            seen.add(key)
                    else:
                        review.append({**common_multi, "raw_text": line[:500], "candidate_dates": dates, "candidate_types": [raw_type], "reason": "品目分類の確信度が低いため"})
                continue
            if len(terms) == 1:
                raw_type = terms[0]
                normalized, confidence, uncertain = ai.classify(raw_type)
            else:
                raw_type, normalized, confidence, uncertain = "", "", 0.0, True
            common = {
                "municipality": municipality,
                "district": "地区不明",
                "type": normalized or raw_type or "品目未判定",
                "note": line[:300],
                "source_page": page_url,
                "source_asset": source_name if source_name != "ページ本文" else "",
                "source_page_number": page_no or None,
                "verification": "要確認",
                "engine": "BuiltIn-Lightweight-AI (v1.0)",
            }
            if len(terms) == 1 and not uncertain and dates and len(dates) == 1:
                key = (dates[0], normalized, source_name, page_no, line)
                if key not in seen:
                    events.append({**common, "date": dates[0], "raw_type": raw_type, "confidence": round(confidence, 3)})
                    seen.add(key)
            elif len(terms) == 1 and not uncertain and rule_match and not dates:
                key = (rule_match.group(0), normalized, source_name, page_no, line)
                if key not in seen:
                    rules.append({**common, "rule": rule_match.group(0), "raw_type": raw_type, "confidence": round(confidence, 3)})
                    seen.add(key)
            else:
                review.append({
                    **common,
                    "raw_text": line[:500],
                    "candidate_dates": dates,
                    "candidate_types": terms,
                    "reason": "日付・品目・地区の対応を一意に確定できないため",
                })
    return events, rules, review

def main():
    sources = read_json(SOURCES, [])
    old_status = read_json(STATUS, {"pdfs": []})
    old_by_url = {x.get("page_url"): x for x in old_status.get("pdfs", [])}
    existing = read_json(OUTPUT, {"events": [], "rules": [], "review_items": []})
    events = list(existing.get("events", []))
    rules = list(existing.get("rules", []))
    review_items = list(existing.get("review_items", []))
    status_rows = []
    ocr = RapidOCR()
    ai = GarbageBuiltInAI()

    for source in sources:
        municipality = source.get("municipality", "")
        page_url = source.get("url", "")
        if not municipality or not page_url:
            continue
        row = {"municipality": municipality, "page_url": page_url}
        assets = []
        try:
            page_text, assets = fetch_page_assets(page_url)
            linked_calendar_text = fetch_linked_calendar_text(page_url)
            if linked_calendar_text:
                page_text = page_text + "\n" + linked_calendar_text
            asset_texts = []
            for asset in assets:
                pages = extract_asset_text(asset, ocr)
                asset_texts.append({"url": asset["url"], "pages": pages})
            if not page_text.strip() and not assets:
                raise RuntimeError("公式ページから本文・添付資料を取得できませんでした")
            digest = hashlib.sha256((page_text + "".join(a["sha256"] for a in assets)).encode("utf-8")).hexdigest()
            row.update({"sha256": digest, "asset_count": len(assets)})
            old = old_by_url.get(page_url, {})
            if old.get("sha256") == digest and old.get("status") in ("processed", "processed_unchanged"):
                row["status"] = "processed_unchanged"
                status_rows.append(row)
                for asset in assets:
                    asset["path"].unlink(missing_ok=True)
                continue

            new_events, new_rules, new_review = parse_text(municipality, page_url, page_text, asset_texts, ai)
            # 解析が正常終了した公式ページ分だけ差し替える。失敗時は既存データを保持。
            events = [x for x in events if x.get("source_page") != page_url] + new_events
            rules = [x for x in rules if x.get("source_page") != page_url] + new_rules
            review_items = [x for x in review_items if x.get("source_page") != page_url] + new_review
            row.update({
                "status": "processed",
                "date_events": len(new_events),
                "rule_events": len(new_rules),
                "review_items": len(new_review),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            row.update({"status": "error", "error": str(exc)[:300]})
        finally:
            for asset in assets:
                try:
                    asset["path"].unlink(missing_ok=True)
                except Exception:
                    pass
        status_rows.append(row)

    timestamp = datetime.now(timezone.utc).isoformat()
    STATUS.write_text(json.dumps({"checked_at": timestamp, "pdfs": status_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT.write_text(json.dumps({
        "updated_at": timestamp,
        "count": len(events),
        "events": events,
        "rules_count": len(rules),
        "rules": rules,
        "review_count": len(review_items),
        "review_items": review_items,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Imported events:", len(events), "rules:", len(rules), "review:", len(review_items))

if __name__ == "__main__":
    main()
