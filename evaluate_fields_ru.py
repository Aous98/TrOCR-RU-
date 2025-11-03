# evaluate_fields_ru_v2.py
import csv, regex as re, argparse
from datetime import datetime
from rapidfuzz import fuzz

FIELDS = ["ИНН", "КПП", "Дата", "Номер", "Тип документа", "Поставщик", "Плательщик"]

def norm_space(s):  # collapse whitespace, strip quotes & special quotes
    s = re.sub(r"[«»\"ʼ’`´']", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def digits_only(s):  # keep only 0-9
    return re.sub(r"\D+", "", s)

def norm_date(s):
    s = s.strip()
    # extract dd mm yyyy (accept ., /, -, spaces)
    m = re.search(r"(\d{1,2})[.\-/\s](\d{1,2})[.\-/\s](\d{2,4})", s)
    if not m: return ""
    d, mth, y = m.group(1), m.group(2), m.group(3)
    if len(y)==2: y = "20"+y
    try:
        return datetime(int(y), int(mth), int(d)).strftime("%d.%m.%Y")
    except ValueError:
        return ""

def grab_after_label(text, label):
    # tolerant capture after label, stop at ., newline, or another label-like token
    pat = fr"{label}\s*[:№\-]?\s*([A-Za-zА-Яа-я0-9/\.\,\-\s]+)"
    m = re.search(pat, text)
    return m.group(1).strip() if m else ""

def extract_all(text):
    out = {}
    out["ИНН"] = grab_after_label(text, "ИНН")
    out["КПП"] = grab_after_label(text, "КПП")
    out["Номер"] = grab_after_label(text, "Номер")
    out["Дата"] = grab_after_label(text, "Дата")
    out["Тип документа"] = grab_after_label(text, "Тип документа")
    out["Поставщик"] = grab_after_label(text, "Поставщик")
    out["Плательщик"] = grab_after_label(text, "Плательщик")
    return out

def equal_inn(gt, pr):
    return digits_only(gt) in (d:=digits_only(pr)) and len(digits_only(gt)) in (10,12)

def equal_kpp(gt, pr):
    return digits_only(gt) == digits_only(pr) and len(digits_only(gt))==9

def equal_number(gt, pr):
    # compare only digits (invoice numbers often have prefixes)
    g, p = digits_only(gt), digits_only(pr)
    return (g != "" and g == p)

def equal_date(gt, pr):
    return norm_date(gt) != "" and norm_date(gt) == norm_date(pr)

def equal_label(gt, pr):
    # for "Тип документа": require that gt token (e.g., "ТОРГ-12") appears in pred
    g = norm_space(gt).lower()
    p = norm_space(pr).lower()
    return len(g)>0 and (g in p or fuzz.partial_ratio(g, p) >= 90)

def equal_name(gt, pr):
    # fuzzy compare supplier/payer names
    g = norm_space(gt).lower()
    p = norm_space(pr).lower()
    # strip long tails like addresses if present
    g = re.split(r",\s*(инн|кпп|бик|р/с|к/с)\b", g, flags=re.I)[0]
    p = re.split(r",\s*(инн|кпп|бик|р/с|к/с)\b", p, flags=re.I)[0]
    return (len(g)>0 and len(p)>0 and fuzz.token_set_ratio(g, p) >= 90)

def eval_file(tsv):
    total = 0
    ok = {f:0 for f in FIELDS}
    present = {f:0 for f in FIELDS}
    with open(tsv, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            total += 1
            gtF = extract_all(row["gt"])
            prF = extract_all(row["pred"])
            # count presence if GT has a value
            for k in FIELDS:
                if gtF[k]:
                    present[k] += 1
            # compare with tolerant rules
            if gtF["ИНН"]:         ok["ИНН"]         += int(equal_inn(gtF["ИНН"], prF["ИНН"]))
            if gtF["КПП"]:         ok["КПП"]         += int(equal_kpp(gtF["КПП"], prF["КПП"]))
            if gtF["Номер"]:       ok["Номер"]       += int(equal_number(gtF["Номер"], prF["Номер"]))
            if gtF["Дата"]:        ok["Дата"]        += int(equal_date(gtF["Дата"], prF["Дата"]))
            if gtF["Тип документа"]: ok["Тип документа"] += int(equal_label(gtF["Тип документа"], prF["Тип документа"]))
            if gtF["Поставщик"]:   ok["Поставщик"]   += int(equal_name(gtF["Поставщик"], prF["Поставщик"]))
            if gtF["Плательщик"]:  ok["Плательщик"]  += int(equal_name(gtF["Плательщик"], prF["Плательщик"]))
    print(f"Samples: {total}\n")
    for k in FIELDS:
        den = max(1, present[k])
        print(f"{k:15s} accuracy: {ok[k]/den:.3f}  (matched {ok[k]}/{den})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    args = ap.parse_args()
    eval_file(args.preds)

