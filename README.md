# 🇷🇺 TrOCR-RU — Russian Invoice OCR using Microsoft TrOCR

This repository provides a complete end-to-end pipeline for **optical character recognition (OCR)** on Russian invoices such as *ТОРГ-12* and *счёт-фактура* using **Microsoft TrOCR**.  
It includes dataset preparation, fine-tuning, inference, and structured evaluation for key invoice fields.

---

## 🧩 Pipeline Overview

1. **Data Preparation** — `build_manifest_ru.py`  
   Converts invoice JSON annotations + images into training/test TSV manifests containing `image_path` and Russian `text`.

2. **Model Fine-Tuning** — `finetune_trocr_ru.py`  
   Fine-tunes the pretrained model `microsoft/trocr-base-stage1` for Russian text recognition.  
   Includes GPU-friendly options (fp16, gradient accumulation, encoder freeze warm-up).

3. **Inference**  
   - `infer_trocr.py` – standard beam-search inference (higher quality, slower).  
   - `infer_trocr_fast.py` – fast greedy inference with progress bar and periodic saving.

4. **Evaluation**  
   - `evaluate_levenshtein.py` – computes Character Error Rate (CER) and Word Error Rate (WER).  
   - `evaluate_fields_ru.py` – measures field-level accuracy for attributes (ИНН, КПП, Дата, Номер, Тип документа, Поставщик, Плательщик).

---

