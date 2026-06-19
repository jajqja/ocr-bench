import json
import os
import time

import click
from tabulate import tabulate

from utils.metrics import calculate_recognition_metrics
from utils.datasets import load_dataset, parse_opts
from models.recognition import load as load_recognition_model


@click.command(help="Benchmark a recognition model on a dataset.")
@click.option(
    "--dataset",
    type=str,
    required=True,
    help="Dataset name (pdf, doclaynet, pdfa, nvidia, folder).",
)
@click.option(
    "--opt",
    "opts",
    multiple=True,
    help="Dataset-specific option key=value (repeatable). E.g. --opt path=a.pdf",
)
@click.option(
    "--results_dir",
    type=str,
    default="./results/recognition_benchmark",
    help="Directory to write results.",
)
@click.option("--max_rows", type=int, default=100, help="Max samples to evaluate.")
@click.option("--batch_size", type=int, default=8, help="Inference batch size.")
@click.option(
    "--model",
    type=str,
    default="surya",
    help="Recognition model name (must be registered in models/recognition/__init__.py).",
)
@click.option("--model_path", type=str, required=True, help="Path to model checkpoint.")
def main(
    dataset: str,
    opts: tuple,
    results_dir: str,
    max_rows: int,
    batch_size: int,
    model: str,
    model_path: str,
):
    print(f"Loading recognition model '{model}' from {model_path}...")
    rec_model = load_recognition_model(model, checkpoint=model_path)

    # --- Load data ---
    ds = load_dataset(dataset)
    opt_dict = parse_opts(opts)
    pathname = ds.pathname(opt_dict)
    print(f"Loading dataset '{dataset}' (opts: {opt_dict or 'none'})")
    images, ground_truth_texts, bboxes = ds.recognition(max_rows, opt_dict)

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
    metrics = calculate_recognition_metrics(
        ground_truth_texts, predictions, show_progress=True
    )

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
        "inference_time_per_sample": (
            inference_time / sample_count if sample_count else 0
        ),
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
        print(
            f"  CER: {metrics['cer_scores'][i]:.4f}, WER: {metrics['wer_scores'][i]:.4f}"
        )


if __name__ == "__main__":
    main()
