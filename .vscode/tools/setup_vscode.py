import json
import sys
from pathlib import Path

def update_paths(base_path):
    settings_template_path = Path(__file__).resolve().parent / 'settings.template.json'
    settings_path = Path(__file__).resolve().parent.parent / 'settings.json'
    
    with open(settings_template_path, 'r') as file:
        settings = json.load(file)

    base_path = Path(base_path).resolve()
    settings["python.analysis.extraPaths"] = [
        str(base_path / "source/extensions/omni.isaac.lab_tasks"),
        str(base_path / "source/extensions/omni.isaac.lab_assets"),
        str(base_path / "source/extensions/omni.isaac.lab")
    ]

    with open(settings_path, 'w') as file:
        json.dump(settings, file, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_paths.py <path_to_isaaclab_directory>")
        sys.exit(1)
    
    base_path_input = sys.argv[1]
    update_paths(base_path_input)
