from pathlib import Path
import json
import xml.etree.ElementTree as ET
import trimesh

roots = [
    Path("runs/object_asset_v1/drink_yykx"),
    Path("runs/object_asset_v1/drink_ad"),
    Path("runs/object_asset_v1/bread"),
    Path("runs/object_asset_v1/pipette"),
]

summary = []

for root in roots:
    print("\n====", root.name, "====")

    urdf = root / "asset/object.urdf"
    meta = root / "asset/object_meta.json"
    visual_path = root / "mesh/visual_mesh.obj"
    collision_path = root / "mesh/collision_mesh.obj"

    ET.parse(urdf)
    print("URDF: ok")

    with open(meta, "r") as f:
        data = json.load(f)
    print("JSON: ok")
    print("object:", data["object_name"])

    visual = trimesh.load(visual_path, force="mesh")
    collision = trimesh.load(collision_path, force="mesh")

    print("visual faces:", len(visual.faces))
    print("visual watertight:", visual.is_watertight)
    print("visual extents:", visual.extents)

    print("collision faces:", len(collision.faces))
    print("collision watertight:", collision.is_watertight)
    print("collision extents:", collision.extents)

    item = {
        "name": root.name,
        "object_name": data["object_name"],
        "urdf": str(urdf),
        "visual_mesh": str(visual_path),
        "collision_mesh": str(collision_path),
        "visual_faces": int(len(visual.faces)),
        "collision_faces": int(len(collision.faces)),
        "visual_watertight": bool(visual.is_watertight),
        "collision_watertight": bool(collision.is_watertight),
        "visual_extents_m": [float(x) for x in visual.extents],
        "collision_extents_m": [float(x) for x in collision.extents],
    }
    summary.append(item)

out = Path("runs/object_asset_v1/asset_summary.json")
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\nWrote:", out)
