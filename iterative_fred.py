#!/usr/bin/env python3
"""
Generates a fred treatment plan with single spot using get_plan_monospot code and runs fred for all energies of interest

Author: Generated for proton therapy workflow
"""

from pathlib import Path
import sys
import subprocess
import shutil

def clear_directories(*dirs):
    """Clear all files in directories without removing the directories themselves."""
    for d in dirs:
        path = Path(d)
        if path.exists():
            for file in path.iterdir():
                if file.is_file():
                    file.unlink()
        else:
            path.mkdir(parents=True)

# =============================================================================
# Configuration of key parameters
# =============================================================================

BM_FILE = "CustomBeamModel.bm"
CT_FILE = "CT_altered.mhd"
FRED_EXE = "fred"
ENERGIES = [80, 100, 120, 140, 160, 180, 200]  # MeV

# Shared beam / geometry settings
GANTRY_ANGLE = 0.0
COUCH_ANGLE = 0.0
SNOUT_POS_MM = 200.0
ISOCENTER_MM = [0.0, 0.0, 0.0]
NPRIM = 5E4

# =============================================================================

if __name__ == "__main__":
    # clear before the energy sweep
    clear_directories("rtplans", "freds", "regions")
    for energy in ENERGIES:
        print(f"\n--- {energy:.1f} MeV ---")
        # Step 1: generate input files with get_plan_monospot
        prep = subprocess.run([
            sys.executable, Path(__file__).parent / "get_plan_monospot.py",
            BM_FILE, CT_FILE,
            "--energy", str(energy),
            "--gantry-angle", str(GANTRY_ANGLE),
            "--couch-angle", str(COUCH_ANGLE),
            "--snout-pos", str(SNOUT_POS_MM),
            "--isocenter", *[str(x) for x in ISOCENTER_MM],
            "--nprim", str(NPRIM),
        ])

        if prep.returncode != 0:
            print(f"  ERROR: input generation failed for {energy:.1f} MeV")
            continue

        # Step 2: run FRED
        fred_inp = next(Path("freds").glob(f"fred_*{energy:.1f}MeV.inp"))
        sim = subprocess.run([FRED_EXE, "-f", str(fred_inp)], shell=True)

        # Move FRED output to an energy-specific name
        shutil.move("out", f"out_{energy:.1f}MeV")

        if sim.returncode != 0:
            print(f"  ERROR: FRED failed for {energy:.1f} MeV")

    print("\n=== Sweep done ===")
