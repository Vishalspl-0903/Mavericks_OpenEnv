"""Cleanup script to remove temporary files"""
import os
from pathlib import Path

# Files to remove
temp_files = [
    r"c:\Users\surya\OneDrive\Desktop\Mavericks_OpenEnv\_create_config_dir.py",
    r"c:\Users\surya\OneDrive\Desktop\Mavericks_OpenEnv\config_tasks_temp.yaml",
    r"c:\Users\surya\OneDrive\Desktop\Mavericks_OpenEnv\medical_triage_env\tasks_new.py",
]

removed = []
not_found = []

for filepath in temp_files:
    path = Path(filepath)
    if path.exists():
        path.unlink()
        removed.append(str(path.name))
        print(f"✓ Removed: {path.name}")
    else:
        not_found.append(str(path.name))
        print(f"✗ Not found: {path.name}")

print(f"\n✓ Removed {len(removed)} files")
if not_found:
    print(f"✗ {len(not_found)} files not found (may have been already removed)")

# Remove this cleanup script itself
Path(__file__).unlink()
print(f"✓ Removed cleanup script itself")
