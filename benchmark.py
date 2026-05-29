#!/usr/bin/env python3
"""
Main benchmark script for evaluating OCR models (detection + recognition) on PDF datasets.

This script provides a unified interface to:
1. Load models (from local path or HuggingFace)
2. Evaluate detection model with IOU, precision, recall metrics
3. Evaluate recognition model with WER, CER, accuracy metrics
4. Generate comprehensive benchmark reports
"""

import click
import os
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from tabulate import tabulate


@click.group()
def cli():
    """OCR Benchmark Suite - Evaluate detection and recognition models."""
    pass


@cli.command()
@click.option(
    "--pdf_path", type=str, required=True, help="Path to PDF file for evaluation"
)
@click.option(
    "--detection_model",
    type=str,
    required=True,
    help="Path to detection model or HuggingFace ID",
)
@click.option(
    "--recognition_model",
    type=str,
    required=True,
    help="Path to recognition model or HuggingFace ID",
)
@click.option(
    "--results_dir",
    type=str,
    default="./benchmark_results",
    help="Directory to save results",
)
@click.option("--max_pages", type=int, default=100, help="Maximum pages to evaluate")
@click.option(
    "--hf_token", type=str, default=None, help="HuggingFace token for private models"
)
@click.option(
    "--debug", is_flag=True, help="Enable debug mode (saves visualization images)"
)
def benchmark_pdf(
    pdf_path: str,
    detection_model: str,
    recognition_model: str,
    results_dir: str,
    max_pages: int,
    hf_token: Optional[str],
    debug: bool,
):
    """Run full benchmark on a PDF file."""

    # Validate PDF exists
    if not os.path.exists(pdf_path):
        click.echo(f"❌ PDF not found: {pdf_path}", err=True)
        return

    click.echo(f"\n{'='*70}")
    click.echo("OCR BENCHMARK - PDF Evaluation")
    click.echo(f"{'='*70}")
    click.echo(f"PDF Path: {pdf_path}")
    click.echo(f"Detection Model: {detection_model}")
    click.echo(f"Recognition Model: {recognition_model}")
    click.echo(f"Max Pages: {max_pages}")
    click.echo(f"Results Dir: {results_dir}")
    click.echo(f"{'='*70}\n")

    os.makedirs(results_dir, exist_ok=True)

    # Run detection benchmark
    click.echo("📊 Running detection benchmark...")
    from text_detection import main as detection_benchmark

    try:
        detection_benchmark(
            [
                "--pdf_path",
                pdf_path,
                "--results_dir",
                os.path.join(results_dir, "detection"),
                "--max_rows",
                str(max_pages),
                "--model_path",
                detection_model,
                "--debug" if debug else "",
            ]
        )
    except Exception as e:
        click.echo(f"❌ Detection benchmark failed: {e}", err=True)
        return

    # Run recognition benchmark
    click.echo("\n📊 Running recognition benchmark...")
    from text_recognition import main as recognition_benchmark

    try:
        recognition_benchmark(
            [
                "--pdf_path",
                pdf_path,
                "--results_dir",
                os.path.join(results_dir, "recognition"),
                "--max_rows",
                str(max_pages),
                "--model_path",
                recognition_model,
            ]
        )
    except Exception as e:
        click.echo(f"❌ Recognition benchmark failed: {e}", err=True)
        return

    # Generate summary report
    click.echo("\n" + "=" * 70)
    click.echo("BENCHMARK SUMMARY")
    click.echo("=" * 70)

    detection_results_file = os.path.join(
        results_dir, "detection", Path(pdf_path).stem, "results.json"
    )
    recognition_results_file = os.path.join(
        results_dir, "recognition", f"{Path(pdf_path).stem}_results.json"
    )

    try:
        with open(detection_results_file) as f:
            det_results = json.load(f)
        with open(recognition_results_file) as f:
            rec_results = json.load(f)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "pdf": pdf_path,
            "detection": {
                "model": det_results.get("model"),
                "metrics": det_results.get("metrics", {}).get("surya", {}),
                "inference_time": det_results.get("times", {}).get("surya"),
            },
            "recognition": {
                "model": rec_results.get("model"),
                "metrics": {
                    "cer": rec_results.get("metrics", {}).get("cer"),
                    "wer": rec_results.get("metrics", {}).get("wer"),
                    "accuracy": rec_results.get("metrics", {}).get("accuracy"),
                },
                "inference_time": rec_results.get("inference_time_total"),
            },
        }

        # Save summary
        summary_path = os.path.join(results_dir, "benchmark_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary table
        summary_data = [
            [
                "Detection - Precision",
                f"{summary['detection']['metrics'].get('precision', 0):.4f}",
            ],
            [
                "Detection - Recall",
                f"{summary['detection']['metrics'].get('recall', 0):.4f}",
            ],
            ["Detection - IOU", f"{summary['detection']['metrics'].get('iou', 0):.4f}"],
            [
                "Detection - Inference Time",
                f"{summary['detection']['inference_time']:.2f}s",
            ],
            ["", ""],
            [
                "Recognition - WER",
                f"{summary['recognition']['metrics'].get('wer', 0):.4f}",
            ],
            [
                "Recognition - CER",
                f"{summary['recognition']['metrics'].get('cer', 0):.4f}",
            ],
            [
                "Recognition - Accuracy",
                f"{summary['recognition']['metrics'].get('accuracy', 0):.4f}",
            ],
            [
                "Recognition - Inference Time",
                f"{summary['recognition']['inference_time']:.2f}s",
            ],
        ]

        print(tabulate(summary_data, headers=["Metric", "Value"], tablefmt="github"))
        click.echo(f"\n✅ Benchmark complete! Summary saved to: {summary_path}")

    except Exception as e:
        click.echo(f"⚠️  Could not generate summary: {e}", err=True)


@cli.command()
@click.option(
    "--dataset_name", type=str, required=True, help="HuggingFace dataset name"
)
@click.option(
    "--detection_model",
    type=str,
    required=True,
    help="Detection model path or HuggingFace ID",
)
@click.option(
    "--recognition_model",
    type=str,
    required=True,
    help="Recognition model path or HuggingFace ID",
)
@click.option(
    "--results_dir", type=str, default="./benchmark_results", help="Results directory"
)
@click.option(
    "--max_samples", type=int, default=100, help="Maximum samples to evaluate"
)
@click.option("--hf_token", type=str, default=None, help="HuggingFace token")
def benchmark_dataset(
    dataset_name: str,
    detection_model: str,
    recognition_model: str,
    results_dir: str,
    max_samples: int,
    hf_token: Optional[str],
):
    """Run full benchmark on a HuggingFace dataset."""

    click.echo(f"\n{'='*70}")
    click.echo("OCR BENCHMARK - Dataset Evaluation")
    click.echo(f"{'='*70}")
    click.echo(f"Dataset: {dataset_name}")
    click.echo(f"Detection Model: {detection_model}")
    click.echo(f"Recognition Model: {recognition_model}")
    click.echo(f"Max Samples: {max_samples}")
    click.echo(f"{'='*70}\n")

    os.makedirs(results_dir, exist_ok=True)

    # Run detection benchmark
    click.echo("📊 Running detection benchmark...")
    from text_detection import main as detection_benchmark

    try:
        detection_benchmark(
            [
                "--dataset_name",
                dataset_name,
                "--results_dir",
                os.path.join(results_dir, "detection"),
                "--max_rows",
                str(max_samples),
                "--model_path",
                detection_model,
            ]
        )
    except Exception as e:
        click.echo(f"❌ Detection benchmark failed: {e}", err=True)
        return

    click.echo("\n✅ Benchmark complete!")


@cli.command()
@click.option("--detection_model", type=str, required=True, help="Detection model path")
@click.option(
    "--recognition_model", type=str, required=True, help="Recognition model path"
)
@click.option("--hf_token", type=str, default=None, help="HuggingFace token")
def verify_models(
    detection_model: str, recognition_model: str, hf_token: Optional[str]
):
    """Verify that models can be loaded and used."""

    click.echo("\n📦 Verifying models...")

    try:
        from load_model import load_detection_model, load_recognition_model

        click.echo(f"Loading detection model: {detection_model}")
        load_detection_model(detection_model, hf_token)
        click.echo("✓ Detection model loaded successfully")

        click.echo(f"Loading recognition model: {recognition_model}")
        load_recognition_model(recognition_model, hf_token)
        click.echo("✓ Recognition model loaded successfully")

        click.echo("\n✅ All models verified!")

    except Exception as e:
        click.echo(f"\n❌ Error loading models: {e}", err=True)
        raise


@cli.command()
@click.option(
    "--results_dir", type=str, default="./benchmark_results", help="Results directory"
)
def compare_results(results_dir: str):
    """Compare results from multiple benchmark runs."""

    if not os.path.exists(results_dir):
        click.echo(f"❌ Results directory not found: {results_dir}", err=True)
        return

    click.echo(f"\n📊 Benchmark Results in {results_dir}")
    click.echo("=" * 70)

    # Find all summary files
    summaries = []
    for root, dirs, files in os.walk(results_dir):
        if "benchmark_summary.json" in files:
            path = os.path.join(root, "benchmark_summary.json")
            with open(path) as f:
                summary = json.load(f)
                summaries.append(summary)

    if not summaries:
        click.echo("No benchmark results found.")
        return

    # Create comparison table
    table_data = []
    for summary in summaries:
        table_data.append(
            [
                summary.get("pdf", summary.get("dataset", "N/A")),
                f"{summary['detection']['metrics'].get('precision', 0):.4f}",
                f"{summary['detection']['metrics'].get('recall', 0):.4f}",
                f"{summary['detection']['metrics'].get('iou', 0):.4f}",
                f"{summary['recognition']['metrics'].get('wer', 0):.4f}",
                f"{summary['recognition']['metrics'].get('cer', 0):.4f}",
                f"{summary['recognition']['metrics'].get('accuracy', 0):.4f}",
            ]
        )

    headers = [
        "Dataset/PDF",
        "Det-Prec",
        "Det-Rec",
        "Det-IOU",
        "Rec-WER",
        "Rec-CER",
        "Rec-Acc",
    ]
    print(tabulate(table_data, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    cli()
