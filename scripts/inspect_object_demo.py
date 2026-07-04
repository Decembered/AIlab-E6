from pathlib import Path
import h5py

root = Path("data/human_demo")

print("root exists:", root.exists(), root)

for demo in sorted(root.iterdir()):
    if not demo.is_dir():
        continue

    print("\n====", demo.name, "====")

    video_dir = demo / "video"
    calib_dir = demo / "camera_calib"
    h5_path = demo / "pose_3d.hdf5"

    print("video dir:", video_dir.exists(), video_dir)
    if video_dir.exists():
        for p in sorted(video_dir.rglob("*"))[:30]:
            if p.is_file():
                print("  video/file:", p)

    print("calib dir:", calib_dir.exists(), calib_dir)
    if calib_dir.exists():
        for p in sorted(calib_dir.rglob("*"))[:30]:
            if p.is_file():
                print("  calib/file:", p)

    print("pose_3d:", h5_path.exists(), h5_path)
    if h5_path.exists():
        with h5py.File(h5_path, "r") as f:
            def visit(name, obj):
                if hasattr(obj, "shape"):
                    print("  H5 dataset:", name, "shape=", obj.shape, "dtype=", obj.dtype)
                else:
                    print("  H5 group:", name)
            f.visititems(visit)
