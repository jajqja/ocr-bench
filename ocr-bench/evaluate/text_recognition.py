import json
import os
import time
from pathlib import Path
from typing import Optional

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
from models.recognition import load as load_recognition_model


@click.command(help="Benchmark a recognition model on a dataset.")
@click.option("--pdf_path", type=str, default=None, help="Path to PDF file.")
@click.option("--dataset_name", type=str, default=None, help="HuggingFace dataset name.")
@click.option(
    "--data_dir", type=str, default=None,
    help="Local dataset folder containing images/ and labels.txt.",
)
@click.option("--image_folder", type=str, default="images", help="Image subfolder name.")
@click.option("--label_file", type=str, default="labels.txt", help="Labels filename.")
@click.option(
    "--results_dir", type=str, default="./results/recognition_benchmark",
    help="Directory to write results.",
)
@click.option("--max_rows", type=int, default=100, help="Max samples to evaluate.")
@click.option("--batch_size", type=int, default=8, help="Inference batch size.")
@click.option(
    "--model", type=str, default="surya",
    help="Recognition model name (must be registered in models/recognition/__init__.py).",
)
@click.option("--model_path", type=str, required=True, help="Path to model checkpoint.")
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
    model: str,
    model_path: str,
    language: str,
    h5_files: str,
    max_size_limit: Optional[int],
):
    print(f"Loading recognition model '{model}' from {model_path}...")
    rec_model = load_recognition_model(model, checkpoint=model_path)

    if max_size_limit:
        print(f"Max image dimension: {max_size_limit}px")

    # --- Load data ---
    if pdf_path is not None:
        pathname = Path(pdf_path).stem
        print(f"Loading data from PDF: {pdf_path}")
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

    flat_bboxes = [bbox for page_bboxes in bboxes for bbox in page_bboxes]
    sample_count = len(ground_truth_texts)
    print(f"Loaded {len(images)} images and {sample_count} text lines")

    # --- Inference ---
    print("Running inference...")
    start_time = time.time()
    predictions = rec_model.predict(images, bboxes, batch_size)
    inference_time = time.time() - start_time

    print(
        f"Inference completed in {inference_time:.2f}s "
        f"({inference_time / len(images):.4f}s per image)"
    )

    # --- Metrics ---
    print("Calculating metrics...")
    metrics = calculate_recognition_metrics(ground_truth_texts, predictions, show_progress=True)

    # --- Save results ---
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{pathname}_results.json")

    output_data = {
        "model": model_path,
        "model_name": model,
        "dataset": pathname,
        "num_samples": sample_count,
        "num_images": len(images),
        "inference_time_total": inference_time,
        "inference_time_per_sample": inference_time / sample_count if sample_count else 0,
        "metrics": {
            "cer": metrics["cer"],
            "wer": metrics["wer"],
            "accuracy": metrics["accuracy"],
        },
        "predictions": [
            {
                "ground_truth": gt,
                "prediction": pred,
                "bbox": flat_bboxes[i],
                "cer": metrics["cer_scores"][i],
                "wer": metrics["wer_scores"][i],
            }
            for i, (gt, pred) in enumerate(zip(ground_truth_texts, predictions))
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
            "Inference Time (per text line)",
            f"{inference_time / sample_count if sample_count else 0:.4f}s",
        ],
    ]

    print("\n" + "=" * 50)
    print("RECOGNITION METRICS")
    print("=" * 50)
    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="github"))
    print(f"\nResults saved to: {result_path}")

    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS (first 5)")
    print("=" * 50)
    for i in range(min(5, len(predictions))):
        print(f"\nSample {i + 1}:")
        print(f"  Ground Truth: {ground_truth_texts[i][:100]}...")
        print(f"  Prediction:   {predictions[i][:100]}...")
        print(f"  CER: {metrics['cer_scores'][i]:.4f}, WER: {metrics['wer_scores'][i]:.4f}")


if __name__ == "__main__":
    main()
