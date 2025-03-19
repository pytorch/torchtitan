import argparse


def extend_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--custom_args.how-is-your-day",
        type=str,
        default="good",
        help="Just an example.",
    )
