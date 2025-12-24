#!/usr/bin/env python3
"""
Quick price fetch + trend check for a few cards.
Note: Requires network access to pokemontcg.io.
"""

import argparse
import logging
from typing import List

from src.utils.logging_config import setup_logging
from src.api.service import PriceService


def run(card_ids: List[str]) -> None:
    service = PriceService(enable_history=True)

    for card_id in card_ids:
        price, trend = service.get_price_with_trend(card_id)
        if not price:
            logging.info("❌ %s: no price returned", card_id)
            continue

        display = price.display_price
        change = f"{trend:+.2f}%" if trend is not None else "n/a"
        logging.info("✅ %s -> %s (trend: %s, source: %s)", card_id, display, change, price.source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test price fetching with trend tracking.")
    parser.add_argument(
        "--cards",
        nargs="+",
        default=["base1-1", "base1-4", "base1-15"],
        help="Card IDs to fetch (default: a few Base Set IDs)",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    run(args.cards)
