import torch, csv, argparse, time, os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, GenerationConfig
from tqdm import tqdm

def read_manifest(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append((row["image_path"], row["text"]))
    return rows

def main(model_name, manifest_path, out_path,
         batch_size=2, num_beams=1, max_new_tokens=384, save_every=50, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device).eval()

    # Greedy by default (fast). You can still pass --num_beams 4 if you want.
    gen_cfg = GenerationConfig(
        do_sample=False,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    data = read_manifest(manifest_path)
    n = len(data)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # open output for incremental writes
    written = 0
    fout = open(out_path, "w", encoding="utf-8", newline="")
    w = csv.writer(fout, delimiter="\t")
    w.writerow(["image_path", "gt", "pred"])

    batch_imgs, batch_meta = [], []
    t0 = time.time()
    pbar = tqdm(total=n, desc="OCR inference", unit="img")

    def flush_batch():
        nonlocal written, batch_imgs, batch_meta
        if not batch_imgs:
            return
        pixel_values = processor(images=batch_imgs, return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            out = model.generate(pixel_values, generation_config=gen_cfg)
        preds = processor.batch_decode(out, skip_special_tokens=True)
        for (ip, gt), pr in zip(batch_meta, preds):
            w.writerow([ip, gt, pr])
        fout.flush()
        written += len(batch_imgs)
        batch_imgs, batch_meta = [], []

    for i, (ip, gt) in enumerate(data, 1):
        try:
            img = Image.open(ip).convert("RGB")
        except Exception as e:
            # write an empty pred with the error note; don't stall
            w.writerow([ip, gt, f"[ERROR opening image: {e}]"])
            fout.flush()
            pbar.update(1)
            continue

        batch_imgs.append(img)
        batch_meta.append((ip, gt))

        if len(batch_imgs) == batch_size:
            flush_batch()
            pbar.update(batch_size)

        # periodic ETA hint + force save
        if i % save_every == 0:
            elapsed = time.time() - t0
            ips = max(1e-9, written) / elapsed
            pbar.set_postfix(speed=f"{ips:.2f} img/s", elapsed=f"{elapsed/60:.1f}m")

    # tail
    flush_batch()
    pbar.update(len(batch_imgs))  # if any left (usually 0)
    pbar.close()
    fout.close()
    print(f"Saved predictions: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--save_every", type=int, default=50)
    args = ap.parse_args()
    main(args.model, args.manifest, args.out,
         batch_size=args.batch_size,
         num_beams=args.num_beams,
         max_new_tokens=args.max_new_tokens,
         save_every=args.save_every)

