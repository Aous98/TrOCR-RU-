import csv, argparse
import editdistance
from jiwer import wer

def cer(gt, pred):
    """Character Error Rate"""
    return editdistance.eval(list(gt), list(pred)) / max(1, len(gt))

def evaluate(tsv_path):
    refs, hyps = [], []
    with open(tsv_path, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            gt, pr = row["gt"].strip(), row["pred"].strip()
            if not gt:
                continue
            refs.append(gt)
            hyps.append(pr)

    total_cer = sum(cer(g, p) for g, p in zip(refs, hyps)) / max(1, len(refs))
    total_wer = wer(refs, hyps)
    print(f"Samples evaluated: {len(refs)}")
    print(f"Character Error Rate (CER): {total_cer:.4f}")
    print(f"Word Error Rate (WER): {total_wer:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    args = ap.parse_args()
    evaluate(args.preds)

