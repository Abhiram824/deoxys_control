"""Replay demonstration trajectories."""

import argparse
import json
import os
import pickle
import threading
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict
from deoxys.utils.config_utils import (get_default_controller_config,
                                       verify_controller_config)

from deoxys import config_root
from deoxys.experimental.motion_utils import follow_joint_traj, reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import robot_config_parse_args
from deoxys.utils.input_utils import input2action
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="recorded_trajecotry.hdf5",
    )
    parser.add_argument(
        "--cfg_num",
        type=int,
        default=10
    )
    robot_config_parse_args(parser)
    return parser.parse_args()

def get_controller_config(cfg="osc_cfg10"):
    pth = os.path.join(config_root, "osc_configs", f"{cfg}.yml")
    # pth = os.path.join(config_root, "osc-pose-controller.yml")

    controller_cfg = YamlConfig(
            pth
        ).as_easydict()
    return controller_cfg


def main():

    DEMO = "demo_3"


    args = parse_args()
    CFG_NUM= args.cfg_num
    with open("info.json", "r") as f:
        info = json.load(f)
    info[f"osc_cfg{CFG_NUM}"] = {}

    # Load recorded demonstration trajectories
    with open(args.dataset, "r") as f:
        demo_file = h5py.File(args.dataset)

        # config = json.loads(demo_file[f"data/{DEMO}"].attrs["config"])

        joint_sequence = demo_file[f"data/{DEMO}/obs/robot0_joint_pos"]
        action_sequence = demo_file[f"data/{DEMO}/actions"]
        eef_state = np.concatenate((demo_file[f"data/{DEMO}/obs/robot0_eef_pos"], demo_file[f"data/{DEMO}/obs/robot0_eef_quat"]), axis=1)
        assert eef_state.shape[1] == 7

    # Initialize franka interface
    device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
    device.start_control()

    # Franka Interface
    robot_interface = FrankaInterface(os.path.join(config_root, args.interface_cfg))

    # Reset to the same initial joint configuration
    logger.info("Resetting to the initial configuration")
    reset_joints_to(robot_interface, joint_sequence[0])

    # Command the same sequence of actions

    # if "OSC" in config["controller_type"]:
    logger.info("Start replay recorded actions using a OSC-family controller")
    controller_cfg = get_controller_config(f"osc_cfg{CFG_NUM}")

    total_joint_err = 0
    total_eef_error = 0
    for i,action in enumerate(action_sequence):
        robot_interface.control(
            controller_type="OSC_POSE",
            action=action[:7],
            controller_cfg=EasyDict(controller_cfg),
        )

        total_joint_err += ((robot_interface.last_q - joint_sequence[i]) ** 2).mean()
        curr_quat, curr_pos = robot_interface.last_eef_quat_and_pos
        curr_pos = curr_pos.squeeze(axis=1)
        curr_eef_state = np.concatenate((curr_pos, curr_quat))
        total_eef_error += ((curr_eef_state - eef_state[i]) ** 2).mean()
        # print(curr_eef_state, eef_state[i])

        
    # elif config["controller_type"] == "JOINT_IMPEDANCE":
    #     follow_joint_traj(robot_interface, joint_sequence)
    logger.info("Finish replaying.")
    print(f"Joint error {total_joint_err}, eef error {total_eef_error}")
    info[f"osc_cfg{CFG_NUM}"]["Joint Error"] = total_joint_err
    info[f"osc_cfg{CFG_NUM}"]["Eef Error"] = total_eef_error
    with open("info.json", "w") as f:
        json.dump(info, f, indent=4)
    robot_interface.close()


if __name__ == "__main__":
    main()
