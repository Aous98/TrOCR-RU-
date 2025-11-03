import json, re, glob
from pathlib import Path
import argparse, csv

def clean(s: str) -> str:
    s = re.sub(r'\s+', ' ', str(s)).strip()
    return s

def flatten_ru_invoice(d: dict) -> str:
    """Turn your Russian TORG/счёт/накладная JSON into one text string.
    Adjust this if your keys differ."""
    parts = []
    # Common top-level blocks (use .get to be robust)
    doc = d.get("Документ", d.get("document", {}))
    parts += [
        f"Тип документа: {doc.get('Тип','') or doc.get('type','')}",
        f"Номер: {doc.get('Номер документа','') or doc.get('number','')}",
        f"Дата: {doc.get('Дата составления','') or doc.get('date','')}",
    ]

    supp = d.get("Поставщик", d.get("supplier", {}))
    parts += [f"Поставщик: {supp.get('Наименование','') or supp.get('name','')}, ИНН {supp.get('ИНН','') or supp.get('inn','')}, КПП {supp.get('КПП','') or supp.get('kpp','')}"]

    cons = d.get("Грузополучатель", d.get("consignee", {}))
    parts += [f"Грузополучатель: {cons.get('Наименование','') or cons.get('name','')}, ИНН {cons.get('ИНН','') or cons.get('inn','')}"]

    payer = d.get("Плательщик", d.get("payer", {}))
    parts += [f"Плательщик: {payer.get('Наименование','') or payer.get('name','')}, ИНН {payer.get('ИНН','') or payer.get('inn','')}"]

    if d.get("Основание") or d.get("basis"):
        parts += [f"Основание: {d.get('Основание') or d.get('basis')}"]

    # items: accept several possible keys
    items = d.get("Товары") or d.get("Items") or d.get("items") or []
    if isinstance(items, dict):  # sometimes stored as {idx: {...}}
        items = list(items.values())
    for i, it in enumerate(items, 1):
        name = it.get("Наименование, характеристика, сорт, артикул товара") or it.get("name","")
        unit = (it.get("Единица измерения") or {}).get("Наименование","") if isinstance(it.get("Единица измерения"), dict) else it.get("unit","")
        qty  = (it.get("Количество (масса, объем)") or {}).get("Мест, штук","") if isinstance(it.get("Количество (масса, объем)"), dict) else it.get("quantity","")
        price = it.get("Цена, руб. коп.") or it.get("price","")
        nds = (it.get("НДС") or {}).get("Ставка, %","") if isinstance(it.get("НДС"), dict) else it.get("vat","")
        parts += [f"Товар {i}: {name}, ед. {unit}, кол-во {qty}, цена {price}, НДС {nds}%"]

    total = d.get("Итого") or d.get("totals") or {}
    if isinstance(total, dict) and total:
        parts += [
            f"Масса брутто: {total.get('Масса груза (брутто), кг','') or total.get('gross_kg','')}",
            f"Масса нетто: {total.get('Масса груза (нетто), кг','') or total.get('net_kg','')}",
            f"Всего к оплате (прописью): {total.get('Всего отпущено на сумму','') or total.get('amount_words','')}",
        ]

    text = ". ".join(clean(p) for p in parts if p)
    return text

def make_split(root: Path, split: str, out_tsv: Path):
    img_dir = root / split / "imgs"
    js_dir  = root / split / "jsons"
    rows = []
    for jp in sorted(js_dir.glob("*.json")):
        with jp.open("r", encoding="utf-8") as f:
            d = json.load(f)
        text = flatten_ru_invoice(d)
        stem = jp.stem
        # try jpg/png
        img = img_dir / f"{stem}.jpg"
        if not img.exists():
            img = img_dir / f"{stem}.png"
        if not img.exists():
            print(f"[WARN] image not found for {jp.name}")
            continue
        rows.append((str(img), text))
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["image_path","text"])
        w.writerows(rows)
    print(f"{split}: wrote {len(rows)} rows to {out_tsv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="/home/aous/Desktop/work/lab/data")
    ap.add_argument("--out_dir", default="manifests")
    args = ap.parse_args()
    root = Path(args.data_root)
    out = Path(args.out_dir)
    make_split(root, "train", out / "train.tsv")
    make_split(root, "test",  out / "test.tsv")

