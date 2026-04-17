import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_fred_simulations(freds_dir="freds", output_base_dir="output"):
    """Run FRED simulation for each .inp file in freds_dir."""
    freds_path = Path(freds_dir)
    fred_files = sorted(freds_path.glob("fred_*.inp"))

    if not fred_files:
        print(f"No fred .inp files found in {freds_dir}/")
        sys.exit(1)

    print(f"Found {len(fred_files)} FRED input file(s)")

    successful = []

   

    for fred_inp in fred_files:
        # e.g. fred_Field1.inp -> Field1
        stem = fred_inp.stem.replace("fred_", "", 1)
        output_dir =  f"{output_base_dir}\out_{stem}"

        print(f"\n--- Running FRED for {fred_inp.name} ---")

        #cmd = ["fred", "-f", str(fred_inp), "-o", str(output_dir)]
        cmd = f"fred -f {fred_inp} -o {output_dir}"
        print(f"  Command: {' '.join(cmd)}")
        start_time = datetime.now()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=True)
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"  ✓ Completed in {elapsed:.1f}s")
            successful.append(fred_inp.name)

        except subprocess.CalledProcessError as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"  ✗ FRED failed after {elapsed:.1f}s (return code: {e.returncode})")
            if e.stderr:
                print(f"    Error: {e.stderr[:500]}")

        except FileNotFoundError:
            print("  ✗ 'fred' command not found. Is FRED installed and in PATH?")
            sys.exit(1)

    print(f"\nComplete: {len(successful)}/{len(fred_files)} simulations succeeded")
    return successful


if __name__ == "__main__":
    run_fred_simulations()