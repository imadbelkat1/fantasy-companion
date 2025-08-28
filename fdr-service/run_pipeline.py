#!/usr/bin/env python3
"""
Main pipeline runner for FDR service
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.pipeline import FDRPipeline
from src.core.config import Config


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('fdr_pipeline.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(description='Run FDR pipeline')
    parser.add_argument('--input', type=str, help='Input JSON path (overrides config)')
    parser.add_argument('--output-dir', type=str, help='Output directory (overrides config)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--leagues', nargs='+', help='Leagues to process (overrides config)', default=None)

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting FDR pipeline...")

    try:
        # Override config if arguments provided
        config = Config()

        if args.input:
            config.DATA_PATH = Path(args.input)

        if args.output_dir:
            config.OUTPUT_DIR = Path(args.output_dir)

        if args.leagues:
            config.LEAGUES = args.leagues

        # Check data file exists
        if not config.DATA_PATH.exists():
            logger.error(f"Data file not found: {config.DATA_PATH}")
            sys.exit(1)

        # Run pipeline
        pipeline = FDRPipeline(config)
        results = pipeline.run_full_pipeline()

        # Print summary
        print(f"\n{'=' * 50}")
        print(f"FDR Pipeline Complete!")
        print(f"{'=' * 50}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Leagues processed: {', '.join(results['leagues_processed'])}")

        for league in results['leagues_processed']:
            team_count = len(results['team_ratings'].get(league, []))
            fixture_count = len(results['fixture_difficulties'].get(league, []))
            print(f"{league.upper()}: {team_count} teams, {fixture_count} fixtures")

        if results['errors']:
            print(f"\nErrors encountered:")
            for error in results['errors']:
                print(f"  - {error}")

        print(f"\nOutput files saved to: {config.OUTPUT_DIR}")

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()