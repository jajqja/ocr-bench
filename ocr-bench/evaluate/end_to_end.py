"""End-to-end OCR benchmark for VLMs and API models.

These models receive an image and return text (with optional bboxes) in a single
forward pass, unlike the two-stage pipeline (detection → recognition).

Evaluation uses page-level CER/WER: all text lines for a page are joined and
compared as a single string, which avoids line-count mismatch issues common
with VLMs that don't predict the exact same number of lines as the GT.
"""
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import click
from tabulate import tabulate

from utils.metrics import calculate_recognition_metrics
from utils.datasets import (
    extract_text_from_pdf,
    load_doclaynet_recognition,
    load_nvidia_recognition,
    load_pdfa_recognition,
    load_recognition_folder,
)


def _load_model(model_type: str, model_name: str, **kwargs):
    """Load a VLM or API model by type and name from its registry."""
    if model_type == "vlm":
        from models.vlm import load
    elif model_type == "api":
        from models.api import load
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'vlm' or 'api'.")
    return load(model_name, **kwargs)


def _group_texts_by_page(
    ground_truth_texts: List[str], bboxes: List[List]
) -> List[str]:
    """Join GT text lines into one string per page using bboxes to find page boundaries."""
    page_texts = []
    text_idx = 0
    for page_bboxes in bboxes:
        n = len(page_bboxes)
        lines = ground_truth_texts[text_idx : text_idx + n]
        page_texts.append("\n".join(lines))
        text_idx += n
    return page_texts


@click.command(help="Benchmark an end-to-end OCR model (VLM or API) on a dataset.")
@click.option("--pdf_path", type=str, default=None, help="Path to PDF file.")
@click.option("--dataset_name", type=str, default=None, help="HuggingFace dataset name.")
@click.option(
    "--data_dir", type=str, default=None,
    help="Local dataset folder containing images/ and labels.txt.",
)
@click.option("--image_folder", type=str, default="images", help="Image subfolder name.")
@click.option("--label_file", type=str, default="labels.txt", help="Labels filename.")
@click.option(
    "--results_dir", type=str, default="./results/end_to_end_benchmark",
    help="Directory to write results.",
)
@click.option("--max_rows", type=int, default=100, help="Max samples to evaluate.")
@click.option("--batch_size", type=int, default=4, help="Inference batch size.")
@click.option(
    "--model_type", type=click.Choice(["vlm", "api"]), required=True,
    help="Model category: 'vlm' for local VLMs, 'api' for cloud API models.",
)
@click.option(
    "--model", type=str, required=True,
    help="Model name within the chosen category (must be registered in its __init__.py).",
)
@click.option(
    "--model_path", type=str, default=None,
    help="Checkpoint path (required for local VLMs, omit for API models).",
)
@click.option(
    "--language", type=str, default="en",
    help="Language for NVIDIA dataset (en, ja, ko, ru, zh_hans, zh_hant).",
)
@click.option(
    "--h5_files", type=str, default="train_000",
    help="Comma-separated H5 filenames for the NVIDIA dataset.",
)
@click.option(
    "--max_size_limit", type=int, default=None,
    help="Resize images so the longest side does not exceed this value.",
)
def main(
    pdf_path: Optional[str],
    dataset_name: Optional[str],
    data_dir: Optional[str],
    image_folder: str,
    label_file: str,
    results_dir: str,
    max_rows: int,
    batch_size: int,
    model_type: str,
    model: str,
    model_path: Optional[str],
    language: str,
    h5_files: str,
    max_size_limit: Optional[int],
):
    kwargs = {}
    if model_path:
        kwargs["checkpoint"] = model_path

    print(f"Loading {model_type} model '{model}'...")
    ocr_model = _load_model(model_type, model, **kwargs)

    # --- Load data ---
    if pdf_path is not None:
        pathname = Path(pdf_path).stem
        print(f"Loading PDF: {pdf_path}")
        images, ground_truth_texts, bboxes = extract_text_from_pdf(pdf_path, max_rows)
    elif dataset_name == "vikp/doclaynet_bench":
        pathname = dataset_name.replace("/", "_")
        print(f"Loading dataset: {dataset_name}")
        images, ground_truth_texts, bboxes = load_doclaynet_recognition(dataset_name, max_rows)
    elif dataset_name == "pixparse/pdfa-eng-wds":
        pathname = dataset_name.replace("/", "_")
        print(f"Loading dataset: {dataset_name}")
        images, ground_truth_texts, bboxes = load_pdfa_recognition(dataset_name, max_rows)
    elif dataset_name == "nvidia/OCR-Synthetic-Multilingual-v1":
        pathname = f"nvidia_ocr_{language}"
        print(f"Loading dataset: {dataset_name}")
        h5_file_list = [f.strip() for f in h5_files.split(",")]
        images, ground_truth_texts, bboxes = load_nvidia_recognition(
            h5_file_list, max_rows, language, max_size_limit=max_size_limit
        )
    elif data_dir is not None:
        pathname = Path(data_dir).name
        print(f"Loading local dataset: {data_dir}")
        images, ground_truth_texts, bboxes = load_recognition_folder(
            data_dir, image_folder, label_file, max_rows
        )
    else:
        raise ValueError("Either --pdf_path, --dataset_name, or --data_dir must be provided")

    # Group GT text lines into page-level strings for evaluation
    page_gt_texts = _group_texts_by_page(ground_truth_texts, bboxes)
    print(f"Loaded {len(images)} images ({len(ground_truth_texts)} GT text lines)")

    # --- Inference ---
    print("Running end-to-end inference...")
    start_time = time.time()
    per_image_results = ocr_model.predict(images, batch_size=batch_size)
    inference_time = time.time() - start_time

    # Join predicted regions into page-level text for comparison
    page_pred_texts = [
        "\n".join(region["text"] for region in image_regions)
        for image_regions in per_image_results
    ]

    print(f"Inference completed in {inference_time:.2f}s")

    # --- Metrics (page-level) ---
    print("Calculating page-level metrics...")
    metrics = calculate_recognition_metrics(page_gt_texts, page_pred_texts, show_progress=True)

    # --- Save results ---
    os.makedirs(results_dir, exist_ok=True)
    result_filename = f"{pathname}_{model_type}_{model}_results.json"
    result_path = os.path.join(results_dir, result_filename)

    output_data = {
        "model": model,
        "model_type": model_type,
        "model_path": model_path,
        "dataset": pathname,
        "num_images": len(images),
        "num_gt_lines": len(ground_truth_texts),
        "inference_time_total": inference_time,
        "inference_time_per_image": inference_time / len(images) if images else 0,
        "metrics": {
            "cer": metrics["cer"],
            "wer": metrics["wer"],
            "accuracy": metrics["accuracy"],
        },
        "page_predictions": [
            {
                "ground_truth": gt,
                "prediction": pred,
                "cer": metrics["cer_scores"][i],
                "wer": metrics["wer_scores"][i],
            }
            for i, (gt, pred) in enumerate(zip(page_gt_texts, page_pred_texts))
        ],
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    table_data = [
        ["Character Error Rate (CER)", f"{metrics['cer']:.4f}"],
        ["Word Error Rate (WER)", f"{metrics['wer']:.4f}"],
        ["Accuracy (exact match)", f"{metrics['accuracy']:.4f}"],
        ["Inference Time (total)", f"{inference_time:.2f}s"],
        [
            "Inference Time (per image)",
            f"{inference_time / len(images) if images else 0:.4f}s",
        ],
    ]

    print("\n" + "=" * 50)
    print("END-TO-END OCR METRICS  (page level)")
    print("=" * 50)
    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="github"))

    print("\nSAMPLE PREDICTIONS (first 3 pages)")
    print("=" * 50)
    for i in range(min(3, len(page_pred_texts))):
        print(f"\nPage {i + 1}:")
        print(f"  GT:   {page_gt_texts[i][:120]}...")
        print(f"  Pred: {page_pred_texts[i][:120]}...")
        print(f"  CER: {metrics['cer_scores'][i]:.4f}, WER: {metrics['wer_scores'][i]:.4f}")

    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
