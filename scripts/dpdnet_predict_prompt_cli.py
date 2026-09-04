#!/usr/bin/env python3
"""CLI wrapper: DpDNet STUNet_prompt inference (prompt-aware predictor)."""
from nnunetv2.inference.predict_from_raw_data_prompt import predict_entry_point

if __name__ == "__main__":
    predict_entry_point()
