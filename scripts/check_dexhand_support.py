#!/usr/bin/env python3
"""Validate a dexhand registration, mapping, and optional IsaacGym asset load."""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dexhand", required=True, help="Dexhand name, e.g. inspire or sharpa")
    parser.add_argument("--side", default="left", choices=["left", "right"])
    parser.add_argument("--load-asset", action="store_true", help="Load the URDF in IsaacGym")
    return parser.parse_args()


def main():
    args = parse_args()

    # Import IsaacGym before anything that may import torch. The package
    # initializer for maniptrans_envs pulls in task modules, which import
    # gymtorch and will fail if torch was imported first.
    import isaacgym  # noqa: F401

    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

    try:
        dexhand = DexHandFactory.create_hand(args.dexhand, args.side)
    except Exception as exc:
        print(f"FAILED: DexHandFactory.create_hand({args.dexhand!r}, {args.side!r})")
        print(f"{type(exc).__name__}: {exc}")
        return 1

    print(f"dexhand: {dexhand}")
    print(f"side: {dexhand.side}")
    print(f"urdf_path: {dexhand.urdf_path}")
    print(f"urdf_exists: {os.path.exists(dexhand.urdf_path)}")
    print(f"n_bodies: {dexhand.n_bodies}")
    print(f"n_dofs: {dexhand.n_dofs}")
    print(f"body_names: {dexhand.body_names}")
    print(f"dof_names: {dexhand.dof_names}")
    print(f"contact_body_names: {dexhand.contact_body_names}")
    print(f"weight_idx: {dexhand.weight_idx}")

    missing_mapped_bodies = [name for name in dexhand.body_names if name not in dexhand.dex2hand_mapping]
    missing_contacts = [name for name in dexhand.contact_body_names if name not in dexhand.body_names]
    missing_weight_indices = []
    for key, indices in dexhand.weight_idx.items():
        for idx in indices:
            if idx < 0 or idx >= dexhand.n_bodies:
                missing_weight_indices.append((key, idx))

    if missing_mapped_bodies:
        print(f"ERROR: body_names missing from dex2hand_mapping: {missing_mapped_bodies}")
        return 2
    if missing_contacts:
        print(f"ERROR: contact_body_names not in body_names: {missing_contacts}")
        return 3
    if missing_weight_indices:
        print(f"ERROR: weight_idx contains out-of-range indices: {missing_weight_indices}")
        return 4

    if not os.path.exists(dexhand.urdf_path):
        print("ERROR: URDF file does not exist.")
        return 5

    if args.load_asset:
        from isaacgym import gymapi

        gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = True
        sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)
        if sim is None:
            print("ERROR: gym.create_sim returned None")
            return 6

        asset_root, asset_file = os.path.split(dexhand.urdf_path)
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        options.disable_gravity = True
        asset = gym.load_asset(sim, asset_root, asset_file, options)
        if asset is None:
            print("ERROR: gym.load_asset returned None")
            gym.destroy_sim(sim)
            return 7

        asset_body_names = [gym.get_asset_rigid_body_name(asset, i) for i in range(gym.get_asset_rigid_body_count(asset))]
        asset_dof_names = [gym.get_asset_dof_name(asset, i) for i in range(gym.get_asset_dof_count(asset))]
        print(f"asset_body_count: {len(asset_body_names)}")
        print(f"asset_dof_count: {len(asset_dof_names)}")
        print(f"asset_body_names: {asset_body_names}")
        print(f"asset_dof_names: {asset_dof_names}")

        missing_asset_bodies = [name for name in dexhand.body_names if name not in asset_body_names]
        missing_asset_dofs = [name for name in dexhand.dof_names if name not in asset_dof_names]
        gym.destroy_sim(sim)

        if missing_asset_bodies:
            print(f"ERROR: dexhand body_names missing from asset: {missing_asset_bodies}")
            return 8
        if missing_asset_dofs:
            print(f"ERROR: dexhand dof_names missing from asset: {missing_asset_dofs}")
            return 9

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
