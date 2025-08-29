import argparse
import shutil
from pathlib import Path
from tests import setup_script_output, get_output_path, initialize_script_output, add_output_dir_arguments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()

    source_path = Path(args.source)
    dest_path = Path(args.destination)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(dest_path))

if __name__ == "__main__":
    main()
