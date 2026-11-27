"""Ask PinkFlag about a message.

    python -m pinkflag.predict "where were you last night? send me ur location"
    python -m pinkflag.predict            # interactive mode
"""
import argparse
import sys

from pinkflag.agent.tools import load_detector
from pinkflag.models.baseline import predict


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Score a message with PinkFlag")
    parser.add_argument("text", nargs="*")
    parser.add_argument("--model", help="path to a specific trained model")
    args = parser.parse_args(argv)

    model, _ = load_detector(args.model)
    if model is None:
        parser.error("Train a detector first: python -m pinkflag.train --config configs/baseline.yaml")

    if args.text:
        print(predict(model, [" ".join(args.text)])[0].pretty())
        return

    print("spill the tea 🍵 (paste a message, empty line to quit)")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            break
        print(predict(model, [text])[0].pretty(), "\n")


if __name__ == "__main__":
    main()
