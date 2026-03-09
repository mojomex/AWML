#!/usr/bin/env python3
# /// script
# dependencies = [
#   "rerun-sdk",
#   "numpy",
#   "natsort",
# ]
# ///
"""
Visualize point cloud segmentation results from AWML save directory using Rerun.

This script loads an AWML save directory and visualizes the point cloud segmentation results in 3D.

Usage:
    uv run tools/render_lidarseg_result.py <save_path>
    uv run tools/render_lidarseg_result.py <save_path> --recording-name <recording_name> --recording-path <recording_path>

Args:
    save_path: AWML save path (containing `result/`, `config.py`, etc.) to visualize
    recording-name: Name for the Rerun recording (default: "pointcloud_visualization")
    recording-path: Path to save the Rerun recording (default: don't save)
"""

from types import ModuleType


import argparse
import re
from typing import Optional
import numpy as np
import rerun as rr
from pathlib import Path
import importlib.util
import sys
import natsort


def _import_module(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _get_class_mapping(config: ModuleType) -> dict[str, int]:
    if not hasattr(config, "class_mapping"):
        raise ValueError(f"Config file does not have 'class_mapping' attribute")
    return config.class_mapping


def _get_class_colors(config: ModuleType) -> dict[int, tuple[int, int, int]]:
    if not hasattr(config, "class_colors"):
        raise ValueError(f"Config file does not have 'class_colors' attribute")
    print("Class colors:")
    for class_id, color in config.class_colors.items():
        print(f"  Class ID {class_id}: Color {color}")
    return config.class_colors


def _extract_sequence_number(filename):
    """
    Extract sequence number from filename in format <number>_....npz

    Args:
        filename: Name of the npz file

    Returns:
        Sequence number as integer, or None if not found
    """
    match = re.match(r"^(\d+)_.*\.npz$", filename)
    if match:
        return int(match.group(1))
    return None


def _make_rerun_annotation_context(
    class_mapping: dict[str, int], class_colors: dict[int, tuple[int, int, int]]
) -> rr.AnnotationContext:
    id_labels_colors = [
        (class_id, class_name, class_colors.get(class_id)) for class_name, class_id in class_mapping.items()
    ]

    annotation_infos = [
        rr.AnnotationInfo(
            # ID field is u16, negative values are not allowed
            id=class_id if class_id >= 0 else 0xFFFF,
            label=class_name,
            color=class_color,
        )
        for class_id, class_name, class_color in id_labels_colors
    ]

    return rr.AnnotationContext(annotation_infos)


def visualize_lidarseg_result(
    result_dir: Path, recording_name: str, annotation_context: rr.AnnotationContext, recording_path: Optional[str] = None
):
    """
    Visualize all npz files in the input directory using Rerun.

    Args:
        result_dir: Directory containing npz files
        recording_name: Name for the Rerun recording
        class_mapping: Class mapping from config file
        class_colors: Class colors from config file
    """
    if not result_dir.exists():
        raise ValueError(f"`result/` directory does not exist: {result_dir}")

    # Initialize Rerun
    rr.init(recording_name, spawn=True)
    rr.log("/", annotation_context, static=True)
    # server_uri = rr.serve_grpc()
    # print(f"Rerun server running at: {server_uri}\n")

    # Find all npz files
    npz_files = natsort.natsorted(result_dir.glob("*.npz"))

    if len(npz_files) == 0:
        print(f"No npz files found in {result_dir}")
        return

    print(f"Found {len(npz_files)} npz files")

    # Process each file
    processed_count = 0
    for npz_file in npz_files:
        try:
            # Extract sequence number from filename: `<number>_<rest>.npz` -> `<number>`
            seq_num = _extract_sequence_number(npz_file.name)
            if seq_num is None:
                print(f"Warning: Could not extract sequence number from {npz_file.name}, skipping")
                continue

            # Load npz file
            data = np.load(npz_file)

            # Check for required keys
            if "pred" not in data or "feat" not in data:
                print(f"Warning: Skipping {npz_file.name} - missing 'pred' or 'feat' keys")
                continue

            # Extract data
            labels = data["pred"]
            features = data["feat"]

            # Take only first 3 columns (x, y, z)
            if features.shape[1] < 3:
                print(f"Warning: Skipping {npz_file.name} - 'feat' has less than 3 columns")
                continue

            points = features[:, :3]

            # Check dimensions match
            if len(labels) != len(points):
                print(f"Warning: Skipping {npz_file.name} - dimension mismatch")
                continue

            # Log to Rerun with sequence number as timeline
            # rr.set_time("frame", sequence=seq_num)

            # Log the point cloud with class labels
            rr.log("/pointcloud", rr.Points3D(positions=points, class_ids=labels.astype(np.uint16)))

            print(f"Processed {npz_file.name} (sequence {seq_num}). Logged {len(points)} points.")
            processed_count += 1

        except Exception as e:
            print(f"Error processing {npz_file.name}: {e}")
            continue

    print(f"\nVisualization complete! Processed {processed_count} point clouds")

    if recording_path is not None:
        print(f"Saving Rerun recording to {recording_path}...")
        rr.save(recording_path)
        print(f"Rerun recording saved.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D point clouds from npz files using Rerun",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "save_path", type=str, help="AWML save path (containing `result/`, `config.py`, etc.) to visualize"
    )
    parser.add_argument(
        "--recording-name", type=str, default="pointcloud_visualization", help="Name for the Rerun recording"
    )
    parser.add_argument(
        "--recording-path", type=str, default=None, help="Path to save the Rerun recording (default: don't save)"
    )

    args = parser.parse_args()

    save_path = Path(args.save_path)
    if not save_path.exists():
        raise ValueError(f"Save path does not exist: {save_path}")

    result_dir = save_path / "result"
    if not result_dir.exists():
        raise ValueError(f"Result directory does not exist: {result_dir}")

    config_path = save_path / "config.py"
    if not config_path.exists():
        raise ValueError(f"Config file does not exist: {config_path}")

    config = _import_module("config", config_path)
    class_mapping = _get_class_mapping(config)
    class_colors = _get_class_colors(config)

    annotation_context = _make_rerun_annotation_context(class_mapping, class_colors)
    visualize_lidarseg_result(result_dir, args.recording_name, annotation_context, args.recording_path)


if __name__ == "__main__":
    main()
