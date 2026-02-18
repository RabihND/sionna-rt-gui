#!/usr/bin/env python3
"""Load and print CIR data from cirs.bin file."""

import numpy as np
import json
import argparse


def load_cirs(cir_export_json: str):
    """
    Load CIR data from a folder containing config.json and cirs.bin.

    Binary format per CIR entry:
      - float32: norm (channel norm)
      - float32[num_taps * 2]: Interleaved real/imag CIR tap values
      - uint16_t[num_taps]: Tap delay indices
      - padding to 4-byte alignment
    """
    # Load config
    with open(cir_export_json, "r") as f:
        config = json.load(f)

    num_taps = config["channel_emulation"]["num_taps"]
    num_cirs = config["channel_emulation"]["num_cirs"]

    # Calculate entry size with 4-byte alignment
    raw_entry_size = (
        4 + num_taps * 4 * 2 + num_taps * 2
    )  # float32 + taps + uint16_t indices
    padded_entry_size = (raw_entry_size + 3) & ~3  # Align to 4 bytes

    # Read binary data
    with open(cir_export_json.replace(".json", ".bin"), "rb") as f:
        data = f.read()

    norms = []
    taps = []
    tap_indices = []

    for i in range(num_cirs):
        offset = i * padded_entry_size

        # Read norm (float32)
        norm = np.frombuffer(data[offset : offset + 4], dtype=np.float32)[0]
        offset += 4

        # Read taps (num_taps * 2 float32 values, interleaved real/imag)
        tap_values = np.frombuffer(
            data[offset : offset + num_taps * 8], dtype=np.float32
        )
        tap_values = tap_values.reshape(num_taps, 2)  # [real, imag] pairs
        offset += num_taps * 8

        # Read tap indices (num_taps uint16_t values)
        indices = np.frombuffer(data[offset : offset + num_taps * 2], dtype=np.uint16)

        norms.append(norm)
        taps.append(tap_values)
        tap_indices.append(indices)

    return {
        "num_taps": num_taps,
        "num_cirs": num_cirs,
        "norms": np.array(norms),
        "taps": np.array(taps),
        "tap_indices": np.array(tap_indices),
    }


def print_cirs(cir_data: dict, verbose: bool = False):
    """Print all CIRs in a readable format."""
    num_cirs = cir_data["num_cirs"]
    num_taps = cir_data["num_taps"]

    print(f"CIR Data: {num_cirs} CIRs, {num_taps} taps each\n")
    print("=" * 80)

    for i in range(num_cirs):
        norm = cir_data["norms"][i]
        taps = cir_data["taps"][i]  # shape: [num_taps, 2]
        indices = cir_data["tap_indices"][i]

        print(f"\nCIR {i:3d}  |  norm = {norm:.6f}")
        print("-" * 60)

        if verbose:
            print(
                f"  {'Tap':>4}  {'Delay':>6}  {'Real':>12}  {'Imag':>12}  {'|H|':>10}"
            )
            print(f"  {'-'*4}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*10}")
            for t in range(num_taps):
                real, imag = taps[t]
                magnitude = np.sqrt(real**2 + imag**2)
                print(
                    f"  {t:4d}  {indices[t]:6d}  {real:12.6f}  {imag:12.6f}  {magnitude:10.6f}"
                )
        else:
            # Compact view: show first few taps
            for t in range(min(3, num_taps)):
                real, imag = taps[t]
                print(f"  tap[{t}]: delay={indices[t]:3d}, h={real:+.4f}{imag:+.4f}j")
            if num_taps > 3:
                print(f"  ... ({num_taps - 3} more taps)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and print CIR data from cirs.bin"
    )
    parser.add_argument(
        "cir_export_json",
        type=str,
        help="Path to the cir_export.json file",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show all taps for each CIR"
    )
    args = parser.parse_args()

    cir_data = load_cirs(args.cir_export_json)
    print_cirs(cir_data, verbose=args.verbose)
