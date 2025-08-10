import argparse
import importlib.util
import inspect
from pathlib import Path
from transforms.base import BaseTransform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--transform-script", required=True)
    parser.add_argument("--output-data", required=True)
    args = parser.parse_args()

    # Run the transform script in a subprocess for sandboxing
    Path(args.output_data).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "python",
            args.transform_script,
            "--input-data", args.input_data,
            "--output-data", args.output_data
        ],
        check=True
    )

if __name__ == "__main__":
    main()