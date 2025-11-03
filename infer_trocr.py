import torch, csv, argparse
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

def main(model_name, manifest_path, out_path, max_length=1024, num_beams=4, batch_size=1, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device).eval()

    rows_out, batch, metas = [], [], []
    with open(manifest_path, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            img = Image.open(row["image_path"]).convert("RGB")
            batch.append(img); metas.append((row["image_path"], row["text"]))
            if len(batch) == batch_size:
                pixel_values = processor(images=batch, return_tensors="pt").pixel_values.to(device)
                with torch.inference_mode():
                    out = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)
                preds = processor.batch_decode(out, skip_special_tokens=True)
                for (ip, gt), pr in zip(metas, preds):
                    rows_out.append((ip, gt, pr))
                batch, metas = [], []
        if batch:
            pixel_values = processor(images=batch, return_tensors="pt").pixel_values.to(device)
            with torch.inference_mode():
                out = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)
            preds = processor.batch_decode(out, skip_special_tokens=True)
            for (ip, gt), pr in zip(metas, preds):
                rows_out.append((ip, gt, pr))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["image_path","gt","pred"])
        w.writerows(rows_out)
    print(f"Saved predictions: {out_path} ({len(rows_out)} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/trocr-base-printed")  # try trocr-large-printed if VRAM allows
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_beams", type=int, default=4)
    args = ap.parse_args()
    main(args.model, args.manifest, args.out, num_beams=args.num_beams, batch_size=args.batch_size)

