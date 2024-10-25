import h5py
import numpy as np
import os
import cv2
import argparse
from robosuite.utils.transform_utils import mat2quat
from tqdm import tqdm
from robomimic.scripts.split_train_val import split_train_val_from_hdf5
import json

def zed_image_transform_fn(image):
    assert len(image.shape) == 3, "expected image to have 3 dimensions, got {}".format(len(image.shape))
    height, width = image.shape[:2]
    new_width = 720
    x_start = (width - new_width) // 2
    image = image[:, x_start:x_start + new_width]
    image = cv2.resize(image, (128,128)).astype(np.uint8)
    return image

def rs_image_transform_fn(image):
    assert len(image.shape) == 3, "expected image to have 3 dimensions, got {}".format(len(image.shape))
    image = cv2.resize(image, (128,128)).astype(np.uint8)
    return image

def get_eef_pos(eef_state):
    O_T_EE = np.array(eef_state).reshape(4, 4).transpose()
    return O_T_EE[:3, 3]

def get_eef_quat(eef_state):
    O_T_EE = np.array(eef_state).reshape(4, 4).transpose()
    return mat2quat(O_T_EE[:3, :3])

def get_gripper_pos(gripper_state):
    return gripper_state[..., np.newaxis]

CONVERSION_MAP = {
    "obs/robot0_agentview_right_image": {
        "filename": "testing_demo_camera_zed_0.npz",
        "transform": zed_image_transform_fn
    },
    "obs/robot0_agentview_left_image": {
        "filename": "testing_demo_camera_zed_1.npz",
        "transform": zed_image_transform_fn
    },
    "obs/robot0_eye_in_hand_image": {
        "filename": "testing_demo_camera_rs_0.npz",
        "transform": rs_image_transform_fn
    },
    "obs/robot0_eef_pos": {
        "filename": "testing_demo_ee_states.npz",
        "transform": get_eef_pos
    },
    "obs/robot0_eef_quat": {
        "filename": "testing_demo_ee_states.npz",
        "transform": get_eef_quat
    },
    "obs/robot0_gripper_qpos": {
        "filename": "testing_demo_gripper_states.npz",
        "transform": get_gripper_pos
    },
    "obs/robot0_joint_pos": {
        "filename": "testing_demo_joint_states.npz",
        "transform": lambda x: x
    },
    "actions": {
        "filename": "testing_demo_action.npz",
        "transform": lambda x: x
    }
}

LANG_INSTR = "pick the can from the counter and place it in the sink"

def convert_deoxys_to_hdf5(deoxys_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    demos = os.listdir(deoxys_dir)
    with h5py.File(os.path.join(output_dir, "demo.hdf5"), "w") as f:
        f.create_group("data")
        for i,demo in enumerate(tqdm(demos)):
            demo_dir = os.path.join(deoxys_dir, demo)
            f.create_group(f"data/demo_{i}")
            for key, value in CONVERSION_MAP.items():
                filename = os.path.join(demo_dir, value["filename"])
                data = np.load(filename)["data"]
                data = np.array([value["transform"](d) for d in data])
                if "image" in key:
                    f[f"data/demo_{i}"].create_dataset(key, data=data, compression="gzip")
                else:
                    f[f"data/demo_{i}"].create_dataset(key, data=data)
            
            f[f"data/demo_{i}"].attrs["ep_meta"] = json.dumps({"lang": LANG_INSTR})

def convert_hdf5_for_training(path):
    with h5py.File(path, "a") as f:
        total_samples = 0
        for ep in f["data"]:
            # add "num_samples" into per-episode metadata
            if "num_samples" in f["data/{}".format(ep)].attrs:
                del f["data/{}".format(ep)].attrs["num_samples"]
            n_sample = f["data/{}/actions".format(ep)].shape[0]
            f["data/{}".format(ep)].attrs["num_samples"] = n_sample
            total_samples += n_sample

        # add total samples to global metadata
        if "total" in f["data"].attrs:
            del f["data"].attrs["total"]
        f["data"].attrs["total"] = total_samples

        f.close()

        # create 90-10 train-validation split in the dataset
        split_train_val_from_hdf5(hdf5_path=path, val_ratio=0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deoxys_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    convert_deoxys_to_hdf5(args.deoxys_dir, args.output_dir) 
    convert_hdf5_for_training(os.path.join(args.output_dir, "demo.hdf5"))       