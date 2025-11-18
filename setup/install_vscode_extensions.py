import json
import subprocess
import sys
import os

# Python 2/3 compatibility
try:
    from pathlib import Path

    HAS_PATHLIB = True
except ImportError:
    HAS_PATHLIB = False


def check_command_exists(command):
    """Check if a command exists in the system PATH."""
    try:
        # Python 2/3 compatible subprocess call
        if sys.version_info[0] >= 3:
            subprocess.run([command, "--version"], capture_output=True, check=True)
        else:
            subprocess.check_call(
                [command, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False


def is_extension_installed(extension_id):
    """Check if a VS Code extension is already installed."""
    try:
        # Python 2/3 compatible subprocess call
        if sys.version_info[0] >= 3:
            result = subprocess.run(
                ["code", "--list-extensions"],
                capture_output=True,
                text=True,
                check=True,
            )
            installed_extensions = result.stdout.strip().split("\n")
        else:
            result = subprocess.check_output(["code", "--list-extensions"])
            installed_extensions = result.decode("utf-8").strip().split("\n")
        return extension_id in installed_extensions
    except (subprocess.CalledProcessError, OSError):
        return False


def install_extension(extension_id):
    """Install a VS Code extension."""
    try:
        # Python 2/3 compatible subprocess call with suppressed output
        if sys.version_info[0] >= 3:
            subprocess.run(
                ["code", "--install-extension", extension_id, "--force"],
                check=True,
                capture_output=True,
            )
        else:
            subprocess.check_call(
                ["code", "--install-extension", extension_id, "--force"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def main():
    print("🔧 Installing VS Code Extensions...")

    # Get the workspace directory (parent of scripts directory)
    if HAS_PATHLIB:
        script_dir = Path(__file__).parent
        workspace_dir = script_dir.parent
        extensions_json_path = workspace_dir / ".vscode" / "extensions.json"
    else:
        # Python 2 compatible path handling
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(script_dir)
        extensions_json_path = os.path.join(workspace_dir, ".vscode", "extensions.json")

    # Check if extensions.json exists
    if not os.path.exists(extensions_json_path):
        print("❌ Error: extensions.json not found at {0}".format(extensions_json_path))
        sys.exit(1)

    print("📄 Reading extensions from: {0}".format(extensions_json_path))

    # Read and parse extensions.json
    try:
        with open(extensions_json_path, "r") as f:
            # Remove comments from JSON (simple approach for JSONC)
            content = f.read()
            lines = content.split("\n")
            cleaned_lines = []
            for line in lines:
                # Remove lines that start with // (simple comment removal)
                stripped = line.strip()
                if not stripped.startswith("//"):
                    cleaned_lines.append(line)
            cleaned_content = "\n".join(cleaned_lines)

            extensions_data = json.loads(cleaned_content)
    except (ValueError, IOError) as e:  # ValueError for JSON decode error in Python 2
        print("❌ Error reading extensions.json: {0}".format(e))
        sys.exit(1)

    # Get recommended extensions
    recommended_extensions = extensions_data.get("recommendations", [])

    if not recommended_extensions:
        print("⚠️  No recommended extensions found in extensions.json")
        return

    print("📋 Found {0} recommended extension(s)".format(len(recommended_extensions)))

    # Install each recommended extension
    failed_extensions = []

    for extension in recommended_extensions:
        if is_extension_installed(extension):
            print("✅ {0} (already installed)".format(extension))
        else:
            print("🔄 Installing {0}...".format(extension))
            if install_extension(extension):
                print("✅ {0} (installed successfully)".format(extension))
            else:
                print("❌ {0} (installation failed)".format(extension))
                failed_extensions.append(extension)

    print()

    if failed_extensions:
        print("\n❌ Failed extensions:")
        for ext in failed_extensions:
            print("  - {0}".format(ext))
        print("Please check your internet connection and VS Code installation.")
        sys.exit(1)
    else:
        print("\n🎉 All extensions processed successfully!")
        print(
            "� Reload VS Code to activate new extensions (Ctrl+Shift+P > 'Developer: Reload Window')"
        )


if __name__ == "__main__":
    main()
