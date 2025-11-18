import json
import argparse
import re


## WARNING: This script will only run in python 3.4+


def load_jsonc(file_path):
    """Load a JSON file that may contain comments (JSONC format)."""
    with open(file_path, "r") as f:
        content = f.read()

    # Remove single-line comments (// ...)
    content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)

    # Remove multi-line comments (/* ... */)
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

    # Parse the cleaned JSON
    return json.loads(content)


def is_extension_installed(extension_id):
    """Check if a VS Code extension is already in the recommended list."""
    data = load_jsonc("project/.vscode/extensions.json")
    for extension in data.get("recommendations", []):
        if extension == extension_id:
            return True
    return False


def get_recommended_extensions(reference_file):
    """Get the list of recommended extensions from a reference file."""
    data = load_jsonc(reference_file)
    return data.get("recommendations", [])


def add_recommendation(extension):
    """Add a VS Code extension to the recommended list."""
    data = load_jsonc("project/.vscode/extensions.json")
    data["recommendations"].append(extension)
    with open("project/.vscode/extensions.json", "w") as f:
        json.dump(data, f, indent=4)


def main(reference_file):
    recommended_extensions = get_recommended_extensions(reference_file)
    for extension in recommended_extensions:
        if not is_extension_installed(extension):
            add_recommendation(extension)
            print(f"Added recommendation: {extension}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add recommended VS Code extensions from a reference file."
    )
    parser.add_argument(
        "reference_file",
        type=str,
        help="Path to the reference JSON file with recommended extensions.",
    )
    args = parser.parse_args()
    main(args.reference_file)
