import hashlib, json, os, re, tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
import pdfplumber, requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/official-sources.json"
STATUS = ROOT / "data/pdf-source-status.json"
OUTPUT = ROOT / "data/imported-collection-dates.json"
HEADERS = {"User-Agent": "IbarakiGarbageCalendar/1.0"}
TOKENS = ("カレンダー", "収集", "ごみ", "ゴミ", "家庭ごみ", "分別", "地区", "曜日")

def read_json(path, default):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return default

def clean(value):
    return re.sub(r"\s+", " ", value or "").strip()

def pdf_links(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")
    found = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        absolute = urljoin(r.url, href)
        label = clean(a.get_text(" ", strip=True))
        if not urlparse(absolute).path.lower().endswith(".pdf"): continue
        if any(t in (label + " " + unquote(urlparse(absolute).path)) for t in TOKENS):
            found.append({"url": absolute, "label": label or "公式PDF"})
    return found

def main():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY is missing; import skipped.")
        return
    sources = read_json(SOURCES, [])
    old = read_json(STATUS, {"pdfs": []})
    old_map = {x.get("url"): x for x in old.get("pdfs", [])}
    existing = read_json(OUTPUT, {"events": []}).get("events", [])
    keys = {(e.get("municipality"),e.get("district"),e.get("date"),e.get("type"),e.get("source_pdf")) for e in existing}
    client = genai.Client(api_key=key)
    rows = []
    for source in sources:
        municipality, page_url = source.get("municipality",""), source.get("url","")
        try: links = pdf_links(page_url)
        except Exception as exc:
            rows.append({"municipality":municipality,"page_url":page_url,"status":"page_error","error":str(exc)[:250]})
            continue
        for link in links:
            url = link["url"]
            row = {"municipality":municipality,"page_url":page_url,"url":url,"label":link["label"]}
            temp_path = None
            try:
                r = requests.get(url, headers=HEADERS, timeout=45)
                r.raise_for_status()
                data = r.content
                if len(data) > 20*1024*1024 or not data.startswith(b"%PDF"): raise ValueError("Invalid PDF or over 20 MB")
                digest = hashlib.sha256(data).hexdigest()
                row.update({"sha256":digest,"bytes":len(data)})
                if old_map.get(url,{}).get("sha256")==digest and old_map.get(url,{}).get("status")=="processed":
                    row["status"]="processed"; rows.append(row); continue
                with tempfile.NamedTemporaryFile(suffix=".pdf",delete=False) as tmp:
                    tmp.write(data); temp_path=Path(tmp.name)
                page_data=[]
                with pdfplumber.open(temp_path) as pdf:
                    for n,page in enumerate(pdf.pages,1):
                        txt=clean(page.extract_text() or "")
                        tables=[]
                        for table in page.extract_tables() or []:
                            tables.extend([" | ".join(clean(c or "") for c in (line or [])) for line in table])
                        page_data.append({"page":n,"text":txt,"tables":tables})
                uploaded=client.files.upload(file=str(temp_path))
                prompt = (
                    "添付PDFだけを根拠に自治体ごみ収集予定を抽出。自治体: "+municipality+
                    "\nPDF URL: "+url+
                    '\nJSON配列のみ。各要素: {"date":"YYYY-MM-DD","type":"燃やすごみ","district":"地区名または不明","note":""}'+
                    "\nPDFに明記された日付だけ。年が不明なら出力しない。曜日・規則から推測しない。地区別は分ける。"+
                    "収集日以外の休止日等は出力しない。判読不能は除外。重複禁止。"+
                    "\npdfplumber抽出テキスト: "+json.dumps(page_data,ensure_ascii=False)[:12000]
                )
                response=client.models.generate_content(model="gemini-2.5-flash",contents=[uploaded,prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json"))
                raw=(response.text or "").strip()
                raw=re.sub(r"^\x60{3}(?:json)?\s*","",raw,flags=re.I)
                raw=re.sub(r"\s*\x60{3}$","",raw)
                parsed=json.loads(raw)
                if not isinstance(parsed,list): raise ValueError("Gemini response must be a JSON array")
                # 公式PDFが更新された場合は、そのPDF由来の旧データを置き換える
                existing=[e for e in existing if e.get("source_pdf") != url]
                keys={(e.get("municipality"),e.get("district"),e.get("date"),e.get("type"),e.get("source_pdf")) for e in existing}
                added=0
                for item in parsed:
                    if not isinstance(item,dict): continue
                    date=str(item.get("date","")); kind=clean(str(item.get("type","")))
                    district=clean(str(item.get("district","不明"))) or "不明"
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}",date) or not kind: continue
                    try: datetime.strptime(date,"%Y-%m-%d")
                    except ValueError: continue
                    event={"municipality":municipality,"district":district,"date":date,"type":kind,
                           "note":clean(str(item.get("note",""))),"source_pdf":url,"source_page":page_url,
                           "verification":"要確認"}
                    k=(municipality,district,date,kind,url)
                    if k not in keys: existing.append(event); keys.add(k); added+=1
                row.update({"status":"processed","events_added":added,"processed_at":datetime.now(timezone.utc).isoformat()})
            except Exception as exc:
                row.update({"status":"error","error":str(exc)[:300]})
            finally:
                if temp_path: temp_path.unlink(missing_ok=True)
            rows.append(row)
    STATUS.write_text(json.dumps({"checked_at":datetime.now(timezone.utc).isoformat(),"pdfs":rows},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    OUTPUT.write_text(json.dumps({"updated_at":datetime.now(timezone.utc).isoformat(),"count":len(existing),"events":existing},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Imported events:",len(existing))

if __name__=="__main__": main()
