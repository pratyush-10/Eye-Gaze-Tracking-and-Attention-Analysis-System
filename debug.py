import os
from pathlib import Path

# Exact paths the script checks
PROJECT_ROOT = Path(".")
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
ORIGINAL_PATH = DATA_ROOT / "Data" / "Original"

print("=== DEBUG: MPIIGaze Dataset Structure ===\n")

print(f"1. Project root: {PROJECT_ROOT.absolute()}")
print(f"2. Data root: {DATA_ROOT.absolute()}")
print(f"3. Expected Original path: {ORIGINAL_PATH.absolute()}")
print(f"4. Original path exists: {ORIGINAL_PATH.exists()}\n")

if ORIGINAL_PATH.exists():
    print("✅ Original folder found!")
    print(f"Contents of Original:")
    for item in ORIGINAL_PATH.iterdir():
        print(f"  {item.name} {'📁' if item.is_dir() else '📄'}")
    
    # Check p00 specifically
    p00_path = ORIGINAL_PATH / "p00"
    print(f"\n5. p00 path: {p00_path.absolute()}")
    print(f"6. p00 exists: {p00_path.exists()}")
    
    if p00_path.exists():
        print("✅ p00 found!")
        print("Contents of p00:")
        for item in p00_path.iterdir():
            print(f"  {item.name} {'📁' if item.is_dir() else '📄'}")
        
        # Check first day
        day01_path = p00_path / "day01"
        print(f"\n7. day01 path: {day01_path.absolute()}")
        print(f"8. day01 exists: {day01_path.exists()}")
        
        if day01_path.exists():
            print("✅ day01 found!")
            print("First 10 items in day01:")
            for item in list(day01_path.iterdir())[:10]:
                print(f"  {item.name} {'📁' if item.is_dir() else '📄'}")
            
            # Check annotation.txt
            annotation_file = day01_path / "annotation.txt"
            print(f"\n9. annotation.txt: {annotation_file.absolute()}")
            print(f"10. annotation.txt exists: {annotation_file.exists()}")
            
            if annotation_file.exists():
                print(f"✅ annotation.txt found! Size: {annotation_file.stat().st_size} bytes")
            else:
                print("❌ annotation.txt MISSING - this is the problem!")
        else:
            print("❌ day01 folder missing")
    else:
        print("❌ p00 folder missing")
else:
    print("❌ Original folder missing")

print("\n" + "="*60)
print("Copy-paste this entire output to get exact fix!")
