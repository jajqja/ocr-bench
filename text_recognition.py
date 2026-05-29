import click
import json
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import fitz as pymupdf

from surya.input.processing import open_pdf, get_page_images, convert_if_not_rgb
from surya.settings import settings
from surya.recognition import RecognitionPredictor
from tabulate import tabulate
import datasets

from utils.metrics import calculate_recognition_metrics


def extract_text_from_pdf(pdf_path: str, max_pages: int = 100) -> Tuple[List, List]:
    """Extract text from PDF using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum number of pages to extract
    
    Returns:
        Tuple of (images, ground_truth_texts)
    """
    doc = open_pdf(pdf_path)
    page_count = min(len(doc), max_pages)
    page_indices = list(range(page_count))
    
    images = get_page_images(doc, page_indices)
    images = convert_if_not_rgb(images)
    
    # Extract ground truth text using PyMuPDF
    ground_truth_texts = []
    for idx in page_indices:
        page = doc[idx]
        text = page.get_text()
        ground_truth_texts.append(text)
    
    doc.close()
    return images, ground_truth_texts


def load_recognition_dataset(dataset_name: str, max_rows: int = 100) -> Tuple[List, List]:
    """Load recognition dataset from Hugging Face.
    
    Args:
        dataset_name: Name of the dataset
        max_rows: Maximum number of samples
    
    Returns:
        Tuple of (images, texts)
    """
    dataset = datasets.load_dataset(dataset_name, split=f"train[:{max_rows}]")
    images = list(dataset["image"])
    images = convert_if_not_rgb(images)
    texts = list(dataset["text"])
    return images, texts


def batch_recognize(predictor, images: List, batch_size: int = 8) -> List[str]:
    """Recognize text from images in batches.
    
    Args:
        predictor: Recognition predictor model
        images: List of PIL images
        batch_size: Batch size for processing
    
    Returns:
        List of recognized texts
    """
    predictions = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_results = predictor(batch)
        for result in batch_results:
            predictions.append(result.text)
    return predictions


@click.command(help="Benchmark recognition model on PDF dataset.")
@click.option("--pdf_path", type=str, help="Path to PDF file for evaluation.", default=None)
@click.option("--dataset_name", type=str, help="Hugging Face dataset name.", 
              default=None)
@click.option("--results_dir", type=str, 
              help="Path to directory for results.", 
              default=os.path.join(settings.RESULT_DIR, "recognition_benchmark"))
@click.option("--max_rows", type=int, help="Maximum number of samples to evaluate.", default=100)
@click.option("--batch_size", type=int, help="Batch size for inference.", default=8)
@click.option("--model_path", type=str, required=True, help="Path to recognition model")
def main(
    pdf_path: Optional[str],
    dataset_name: Optional[str],
    results_dir: str,
    max_rows: int,
    batch_size: int,
    model_path: str,
):
    """Benchmark text recognition model."""
    
    # Load model
    print(f"Loading recognition model from {model_path}...")
    rec_predictor = RecognitionPredictor(checkpoint=model_path)
    
    # Load data
    if pdf_path is not None:
        print(f"Loading data from PDF: {pdf_path}")
        pathname = Path(pdf_path).stem
        images, ground_truth_texts = extract_text_from_pdf(pdf_path, max_rows)
    elif dataset_name is not None:
        print(f"Loading dataset: {dataset_name}")
        pathname = dataset_name
        images, ground_truth_texts = load_recognition_dataset(dataset_name, max_rows)
    else:
        raise ValueError("Either pdf_path or dataset_name must be provided")
    
    print(f"Loaded {len(images)} images")
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    predictions = batch_recognize(rec_predictor, images, batch_size)
    inference_time = time.time() - start_time
    
    print(f"Inference completed in {inference_time:.2f}s ({inference_time/len(images):.4f}s per image)")
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_recognition_metrics(ground_truth_texts, predictions)
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{pathname}_results.json")
    
    output_data = {
        "model": model_path,
        "dataset": pathname,
        "num_samples": len(images),
        "inference_time_total": inference_time,
        "inference_time_per_sample": inference_time / len(images),
        "metrics": {
            "cer": metrics["cer"],
            "wer": metrics["wer"],
            "accuracy": metrics["accuracy"],
        },
        "predictions": [
            {
                "ground_truth": gt,
                "prediction": pred,
                "cer": metrics["cer_scores"][i],
                "wer": metrics["wer_scores"][i],
            }
            for i, (gt, pred) in enumerate(zip(ground_truth_texts, predictions))
        ]
    }
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print results table
    table_data = [
        ["Character Error Rate (CER)", f"{metrics['cer']:.4f}"],
        ["Word Error Rate (WER)", f"{metrics['wer']:.4f}"],
        ["Accuracy (exact match)", f"{metrics['accuracy']:.4f}"],
        ["Inference Time (total)", f"{inference_time:.2f}s"],
        ["Inference Time (per sample)", f"{inference_time/len(images):.4f}s"],
    ]
    
    print("\n" + "="*50)
    print("RECOGNITION METRICS")
    print("="*50)
    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="github"))
    print(f"\nResults saved to: {result_path}")
    
    # Show sample predictions
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS (first 5)")
    print("="*50)
    for i in range(min(5, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"  Ground Truth: {ground_truth_texts[i][:100]}...")
        print(f"  Prediction:   {predictions[i][:100]}...")
        print(f"  CER: {metrics['cer_scores'][i]:.4f}, WER: {metrics['wer_scores'][i]:.4f}")


if __name__ == "__main__":
    main()
