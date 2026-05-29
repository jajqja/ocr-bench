#!/usr/bin/env python3
"""
Quick start script for OCR Benchmark
Shows examples of how to use the benchmark suite
"""

from pathlib import Path


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main():
    """Show usage examples."""

    print_section("OCR Benchmark Suite - Quick Start Guide")

    # Detect if models exist
    detection_model_exists = Path("./model_path/detection").exists()
    recognition_model_exists = Path("./model_path/recognition").exists()
    sample_pdf = Path("./sample.pdf").exists()

    print("📋 System Check:")
    print(
        "  ✓ Detection model found"
        if detection_model_exists
        else "  ✗ Detection model NOT found"
    )
    print(
        "  ✓ Recognition model found"
        if recognition_model_exists
        else "  ✗ Recognition model NOT found"
    )
    print("  ✓ Sample PDF found" if sample_pdf else "  ✗ Sample PDF NOT found")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)

    if not detection_model_exists or not recognition_model_exists:
        print_section("Step 1: Download Models")
        print("Option A - From HuggingFace Hub:")
        print("""
python -m load_model download-detection \\
  --repo_id jajqja/surya-detection

python -m load_model download-recognition \\
  --repo_id jajqja/surya-recognition
        """)

        print("\nOption B - From Local Storage:")
        print("""
# Just point to your local model directory
# The models will be used from: ./model_path/detection and ./model_path/recognition
        """)

        print("\nVerify models are loaded:")
        print("""
python -m load_model verify-models \\
  --detection_model ./model_path/detection \\
  --recognition_model ./model_path/recognition
        """)

    if sample_pdf:
        print_section("Step 2: Run Benchmark on PDF")
        print("""
python benchmark.py benchmark-pdf \\
  --pdf_path ./sample.pdf \\
  --detection_model ./model_path/detection \\
  --recognition_model ./model_path/recognition \\
  --max_pages 100 \\
  --results_dir ./results
        """)
    else:
        print_section("Step 2: Prepare Your PDF")
        print("""
Place your PDF file in the project directory or specify the full path:

python benchmark.py benchmark-pdf \\
  --pdf_path /path/to/your/document.pdf \\
  --detection_model ./model_path/detection \\
  --recognition_model ./model_path/recognition \\
  --max_pages 100
        """)

    print_section("Step 3: View Results")
    print("""
Results are saved in ./results/ directory:

📁 results/
├── detection/document/results.json      # Detection metrics
├── recognition/document_results.json    # Recognition metrics  
└── benchmark_summary.json               # Summary of both

View summary:
python benchmark.py compare-results --results_dir ./results
    """)

    print_section("Step 4: (Optional) Run Individual Benchmarks")
    print("""
# Detection only
python -m text_detection \\
  --pdf_path document.pdf \\
  --model_path ./model_path/detection \\
  --debug  # Optional: save visualization images

# Recognition only
python -m text_recognition \\
  --pdf_path document.pdf \\
  --model_path ./model_path/recognition
    """)

    print_section("Common Commands Reference")
    print("""
# List all available commands
python benchmark.py --help
python -m load_model --help
python -m text_detection --help
python -m text_recognition --help

# Download models from HuggingFace
python -m load_model download-detection --repo_id username/model
python -m load_model download-recognition --repo_id username/model

# Use HuggingFace datasets
python -m text_detection \\
  --dataset_name mnist-ocr-digits \\
  --model_path ./model_path/detection \\
  --max_rows 1000

# Compare multiple benchmark runs
python benchmark.py compare-results --results_dir ./results

# Use private HuggingFace models
python benchmark.py benchmark-pdf \\
  --pdf_path document.pdf \\
  --detection_model username/private-model \\
  --recognition_model username/private-model \\
  --hf_token your_token
    """)

    print_section("Metrics Explained")
    print("""
DETECTION METRICS:
  • Precision: % of predicted boxes that correctly cover ground truth
  • Recall: % of ground truth boxes covered by predictions
  • IOU: Intersection over Union with overlap penalty

RECOGNITION METRICS:
  • CER (Character Error Rate): Levenshtein distance at character level
  • WER (Word Error Rate): Levenshtein distance at word level
  • Accuracy: Exact match rate (% perfectly correct)
    """)

    print_section("For More Information")
    print("""
📖 See README.md for:
   - Detailed feature descriptions
   - Complete API documentation
   - Advanced usage examples
   - Troubleshooting guide

📊 Example workflow with your own PDF:
   1. python benchmark.py verify-models \\
      --detection_model ./model_path/detection \\
      --recognition_model ./model_path/recognition
   2. python benchmark.py benchmark-pdf \\
      --pdf_path ./my_document.pdf \\
      --detection_model ./model_path/detection \\
      --recognition_model ./model_path/recognition
   3. python benchmark.py compare-results --results_dir ./results
    """)

    print("\n✨ Ready to benchmark! Start with Step 1 above.\n")


if __name__ == "__main__":
    main()
