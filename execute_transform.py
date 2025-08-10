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

    # Load the transform script as a module
    spec = importlib.util.spec_from_file_location("transform_module", args.transform_script)
    transform_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transform_module)

    # Find the transform class in the module
    transform_class = None
    for name, obj in inspect.getmembers(transform_module):
        if inspect.isclass(obj) and issubclass(obj, BaseTransform) and obj is not BaseTransform:
            transform_class = obj
            break

    if not transform_class:
        raise TypeError(f"Could not find a class that inherits from BaseTransform in {args.transform_script}")

    # Instantiate and run the transform
    Path(args.output_data).parent.mkdir(parents=True, exist_ok=True)
    transform_instance = transform_class(args.input_data, args.output_data)
    transform_instance.run()

if __name__ == "__main__":
    main()