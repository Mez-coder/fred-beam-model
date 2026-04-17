import sys
import argparse
import SimpleITK as sitk


def main():
    parser = argparse.ArgumentParser(
        description="Read a CT .mhd, set every voxel to a constant value, and write it back out."
    )
    parser.add_argument("ct_path", help="Path to input CT .mhd file")
    parser.add_argument(
        "-v", "--value", type=float, default=0.0,
        help="Value to set every voxel to (default: 0)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path (default: <input>_altered.mhd)"
    )
    args = parser.parse_args()

    out_path = args.output or args.ct_path.replace(".mhd", "_altered.mhd")

    ct = sitk.ReadImage(args.ct_path)
    new_ct = sitk.Image(ct.GetSize(), ct.GetPixelID())
    new_ct.CopyInformation(ct)
    new_ct += int(args.value) if new_ct.GetPixelID() in (
        sitk.sitkInt8, sitk.sitkUInt8, sitk.sitkInt16, sitk.sitkUInt16,
        sitk.sitkInt32, sitk.sitkUInt32, sitk.sitkInt64, sitk.sitkUInt64,
    ) else args.value

    sitk.WriteImage(new_ct, out_path)
    print(f"Wrote {out_path}  (size={new_ct.GetSize()}, value={args.value})")


if __name__ == "__main__":
    main()