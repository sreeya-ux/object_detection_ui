"""
batch_process.py
────────────────
Automates the InfrastructurePipeline across multiple images.
Generates annotated visuals and a CSV summary report.

Usage:
  python files/batch_process.py --input ./my_photos --output ./results
"""

import os
import csv
import argparse
import time
from pathlib import Path
from typing import List

# Fix relative imports
import sys
curr_dir = Path(__file__).parent.absolute()
if str(curr_dir) not in sys.path:
    sys.path.append(str(curr_dir))

from pipeline import InfrastructurePipeline

def main():
    parser = argparse.ArgumentParser(description="Batch process infrastructure images.")
    parser.add_argument("--input",  type=str, required=True, help="Path to input image directory.")
    parser.add_argument("--output", type=str, default="batch_results", help="Path to output directory.")
    parser.add_argument("--model",  type=str, default="best_whole.pt", help="Path to component model.")
    parser.add_argument("--shed",   type=str, default="best_disc.pt",  help="Path to shed counter model.")
    args = parser.parse_args()

    # 1. Setup Directories
    input_path = Path(args.input)
    output_path = Path(args.output)
    annotated_path = output_path / "annotated"
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_path}' not found.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    annotated_path.mkdir(parents=True, exist_ok=True)

    # 2. Instantiate Pipeline (Loads models once)
    print("Initializing Pipeline...")
    try:
        pipeline = InfrastructurePipeline(
            component_model_path = args.model,
            shed_model_path      = args.shed
        )
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 3. Discover Images
    valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = [f for f in input_path.iterdir() if f.suffix in valid_exts]
    
    if not image_files:
        print(f"No images found in {input_path}")
        return

    print(f"Found {len(image_files)} images. Starting processing...\n")

    # 4. Process Loop
    results_data = []
    start_time = time.time()

    for idx, img_file in enumerate(image_files):
        print(f"[{idx+1}/{len(image_files)}] Processing {img_file.name}...", end=" ", flush=True)
        
        try:
            save_name = annotated_path / f"res_{img_file.name}"
            result = pipeline.predict(str(img_file), visualize=True, save_path=str(save_name))
            
            # Record data for CSV
            results_data.append({
                "image_name": img_file.name,
                "final_class": result.final_class,
                "voltage": result.voltage,
                "confidence": result.confidence,
                "reason": result.reason,
                "signals": ", ".join(result.signals_used),
                "faults": len(result.adjustment_faults)
            })
            print("Done.")
            
        except Exception as e:
            print(f"FAILED: {e}")
            results_data.append({
                "image_name": img_file.name,
                "final_class": "ERROR",
                "voltage": "N/A",
                "confidence": "N/A",
                "reason": str(e),
                "signals": "",
                "faults": 0
            })

    # 5. Save Summary Report
    csv_file = output_path / "summary.csv"
    fieldnames = ["image_name", "final_class", "voltage", "confidence", "reason", "signals", "faults"]
    
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)

    duration = time.time() - start_time
    print(f"\n=======================================================")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Processed: {len(image_files)} images")
    print(f"Time Taken: {duration:.2f} seconds ({duration/len(image_files):.2f}s per image)")
    print(f"Results saved to: {output_path.absolute()}")
    print(f"CSV Report: {csv_file.name}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
