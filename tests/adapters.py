from __future__ import annotations

import os
from typing import Any


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    from cs336_data.tool import run_extract_text_from_html_bytes
    text = run_extract_text_from_html_bytes(html_bytes)
    return text


def run_identify_language(text: str) -> tuple[Any, float]:
    from cs336_data.tool import indentify_language
    return indentify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    from cs336_data.tool import mask_email
    return mask_email(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    from cs336_data.tool import mask_phone_numbers
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    from cs336_data.tool import mask_ip_address
    return mask_ip_address(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    from cs336_data.tool import identify_nsfw
    return identify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    from cs336_data.tool import identify_toxic
    return identify_toxic(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    from cs336_data.tool import gopher_test
    return gopher_test(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError
