import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--transform-script", required=True)
    parser.add_argument("--output-data", required=True)
    args = parser.parse_args()

    # Run the transform script in a subprocess for sandboxing
    Path(args.output_data).parent.mkdir(parents=True, exist_ok=True)
    # Ensure transform script can import local project modules like `transforms.base`
    env = os.environ.copy()
    root_dir = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = root_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    result = subprocess.run(
        [
            sys.executable,
            args.transform_script,
            "--input-data", args.input_data,
            "--output-data", args.output_data
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=root_dir,
    )
    if result.returncode != 0:
        print("Error running transform script:")
        print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)
    print(result.stdout)


if __name__ == "__main__":
    main()