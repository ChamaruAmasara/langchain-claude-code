import importlib
import sys

if __name__ == "__main__":
    for file in sys.argv[1:]:
        # Convert file path to module path
        module = file.replace("/", ".").replace(".py", "")
        try:
            importlib.import_module(module)
        except Exception as e:
            print(f"Error importing {module}: {e}")
            sys.exit(1)
