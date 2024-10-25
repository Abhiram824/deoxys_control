"""Teleoperating robot arm with a SpaceMouse to collect demonstration data"""

import argparse
import json
import os
import pickle
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from deoxys_vision.networking.camera_redis_interface import \
    CameraRedisSubInterface
from deoxys_vision.utils.camera_utils import get_camera_info
from deoxys.experimental.motion_utils import reset_joints_to


from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import robot_config_parse_args
from deoxys.utils.input_utils import input2action
# from deoxys.k4a_interface import K4aInterface
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor_id",
        type=int,
        default=9583,
    )
    parser.add_argument(
        "--product_id",
        type=int,
        default=50741,
    )
    robot_config_parse_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    args.folder.mkdir(parents=True, exist_ok=True)

    experiment_id = 0

    logger.info(f"Saving to {args.folder}")

    # Create a folder that saves the demonstration raw states.
    for path in args.folder.glob("run*"):
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split("run")[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    experiment_id += 1
    folder = str(args.folder / f"run{experiment_id}")

    device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
    device.start_control()

    # Franka Interface
    robot_interface = FrankaInterface(os.path.join(config_root, args.interface_cfg))
    

    camera_ids = ["rs_0", "zed_0", "zed_1"]
    cr_interfaces = {}
    for camera_id in camera_ids:
        cr_interface = CameraRedisSubInterface(camera_info=get_camera_info(camera_id), redis_host="127.0.0.1")
        cr_interface.start()
        cr_interfaces[camera_id] = cr_interface

    controller_cfg = YamlConfig(
        os.path.join(config_root, args.controller_cfg)
    ).as_easydict()

    # demo_file = h5py.File(demo_file_name, "w")
    controller_type = args.controller_type

    data = {"action": [], "ee_states": [], "joint_states": [], "gripper_states": []}
    for camera_id in camera_ids:
        data[f"camera_{camera_id}"] = []
    i = 0
    start = False

    previous_state_dict = None
    reset_joint_positions = [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ] 
    reset_joints_to(robot_interface, np.array(reset_joint_positions), gripper_open=True)
    time.sleep(2)
    print("Start teleoperation")
    while True:
        i += 1
        start_time = time.time_ns()
        action, grasp = input2action(
            device=device,
            controller_type=controller_type,
        )
        if action is None:
            break

        # set unused orientation dims to 0
        if controller_type == "OSC_YAW":
            action[3:5] = 0.0
        elif controller_type == "OSC_POSITION":
            action[3:6] = 0.0
        assert len(robot_interface._state_buffer) > 0, "state buffer should be initialized since we reset joints!"
        # assert actions are in between 1 and -1
        assert np.all(action <= 1) and np.all(action >= -1), "actions should be in between 1 and -1, got action: {}".format(action)
        if np.linalg.norm(action[:-1]) < 1e-3 and not start:
            continue
        last_state = robot_interface._state_buffer[-1]
        last_gripper_state = robot_interface._gripper_state_buffer[-1]
        for camera_id in camera_ids:
            img = cr_interfaces[camera_id].get_img()
            data[f"camera_{camera_id}"].append(img["color"])

        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )


        start = True
        # print(action.shape)
        # Record ee pose,  joints

        data["action"].append(action)

        state_dict = {
            "ee_states": np.array(last_state.O_T_EE),
            "joint_states": np.array(last_state.q),
            "gripper_states": np.array(last_gripper_state.width),
        }

        if previous_state_dict is not None:
            for proprio_key in state_dict.keys():
                proprio_state = state_dict[proprio_key]
                if np.sum(np.abs(proprio_state)) <= 1e-6:
                    proprio_state = previous_state_dict[proprio_key]
                state_dict[proprio_key] = np.copy(proprio_state)
        for proprio_key in state_dict.keys():
            data[proprio_key].append(state_dict[proprio_key])

        previous_state_dict = state_dict
        # data["ee_states"].append(np.array(last_state.O_T_EE))
        # joints = np.array(last_state.q)
        # if np.sum(np.abs(joints)) < 1e-6:
        #     print("Joints missing!!!!")
        # data["joint_states"].append(np.array(last_state.q))
        # data["gripper_states"].append(np.array(last_gripper_state.width))
        # Get img info

        

        # TODO: Test if we can directly save img (probably not)
        # img = cr_interface.get_img()

        end_time = time.time_ns()
        print(f"Time profile: {(end_time - start_time) / 10 ** 9}")
    reset_joints_to(robot_interface, np.array(reset_joint_positions), gripper_open=True)
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/config.json", "w") as f:
        config_dict = {
            "controller_cfg": dict(controller_cfg),
            "controller_type": controller_type,
        }
        json.dump(config_dict, f)
        np.savez(f"{folder}/testing_demo_action", data=np.array(data["action"]))
        np.savez(f"{folder}/testing_demo_ee_states", data=np.array(data["ee_states"]))
        np.savez(
            f"{folder}/testing_demo_joint_states", data=np.array(data["joint_states"])
        )
        np.savez(
            f"{folder}/testing_demo_gripper_states",
            data=np.array(data["gripper_states"]),
        )

    for camera_id in camera_ids:
        if len(data[f"camera_{camera_id}"]) != len(data["action"]):
            print("camera has one extra image")
            del data[f"camera_{camera_id}"][-1]
        assert len(data[f"camera_{camera_id}"]) == len(data["action"]), "images not matched up with actions"
        np.savez(
            f"{folder}/testing_demo_camera_{camera_id}",
            data=np.array(data[f"camera_{camera_id}"]),
        )
        cr_interfaces[camera_id].stop()
    robot_interface.close()
    print("Total length of the trajectory: ", len(data["action"]))
    valid_input = False
    while not valid_input:
        try:
            save = input("Save or not? (enter 0 or 1)")
            save = bool(int(save))
            valid_input = True
        except:
            pass
    if not save:
        import shutil

        shutil.rmtree(f"{folder}")
    else:
        print(f"saved to {folder}")


if __name__ == "__main__":
    main()
