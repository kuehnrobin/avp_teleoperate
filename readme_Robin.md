# How to use quest_teleop

This Guid explains how to start the Hand Teleoperation with the Meta Quest 3.

## Short Guide

    unitree@PC2:~/image_server$ python image_server.py


    python ~/avp_teleoperate/teleop/teleop_hand_and_arm.py --arm=G1_29 --hand=dex3 --cyclonedds_uri=enxa0cec8616f27

    python teleop_hand_and_arm.py --arm=G1_29 --hand=dex3 --cyclonedds_uri=enxa0cec8616f27 --record

    Reccomended: limit speed
    python teleop_hand_and_arm.py --arm=G1_29 --hand=dex3 --arm-speed=10.0 --no-gradual-speed --cyclonedds_uri=enxa0cec8616f27 --record --force --active-camera
    python teleop_hand_and_arm.py --arm=G1_29 --hand=dex3 --arm-speed=10.0 --no-gradual-speed --cyclonedds_uri=eno1 --record --force --active-camera

    Note: press s in the OpenCV window (with the camera image) to start /stop recording an episode and q to stop


    ngrok http 8012

    adb reverse tcp:8012 tcp:8012 
    ngrok http 8012

### Static domain

    

## Now located in Unitree Robot PC2 terminal

    unitree@PC2:~/image_server$ python image_server.py
    # You can see the terminal output as follows:
    # {'fps': 30, 'head_camera_type': 'opencv', 'head_camera_image_shape': [480, 1280], 'head_camera_id_numbers': [0]}
    # [Image Server] Head camera 0 resolution: 480.0 x 1280.0
    # [Image Server] Image server has started, waiting for client connections...

## Simulation Teleoperation Example

After setup up streaming with either local or network streaming following the above instructions, you can try teleoperating two robot hands in Issac Gym:

    cd teleop && python teleop_hand.py

Go to your vuer site on Quest3, click Enter VR and Allow to enter immersive environment.

## Start

Everyone must keep a safe distance from the robot to prevent any potential danger!

Please make sure to read the Official Documentation at least once before running this program.

Always make sure that the robot has entered debug mode (L2+R2) to stop the motion control program, this will avoid potential command conflict problems.

It's best to have two operators to run this program, referred to as Operator A and Operator B.

First, Operator B needs to perform the following steps:

Modify the img_config image client configuration under the if __name__ == '__main__': section in ~/avp_teleoperate/teleop/teleop_hand_and_arm.py. It should match the image server parameters you configured on PC2 in Section 3.1.

Choose different launch parameters based on your robot configuration

* 1. G1 (29DoF) Robot + Dex3-1 Dexterous Hand (Note: G1_29 is the default value for --arm, so it can be omitted)
(tv) unitree@Host:~/avp_teleoperate/teleop$ 
    python teleop_hand_and_arm.py --arm=G1_29 --hand=dex3

* 2. G1 (29DoF) Robot only
(tv) unitree@Host:~/avp_teleoperate/teleop$ python teleop_hand_and_arm.py

* 3. H1_2 Robot (Note: The first-generation Inspire Dexterous Hand is currently only supported in the H1_2 branch. Support for the Main branch will be added later.)
(tv) unitree@Host:~/avp_teleoperate/teleop$ python teleop_hand_and_arm.py --arm=H1_2

* 4. If you want to enable data visualization + recording, you can add the --record option
(tv) unitree@Host:~/avp_teleoperate/teleop$ python teleop_hand_and_arm.py --record
If the program starts successfully, the terminal will pause at the final line displaying the message: "Please enter the start signal (enter 'r' to start the subsequent program):"

And then, Operator A needs to perform the following steps:

Wear your Quest3 device.

Open the Browser on Quest3 and visit : https://192.168.123.2:8012?ws=wss://192.168.123.2:8012

p.s. This IP address should match the IP address of your Host machine.

Click Enter VR and Allow to start the VR session.

You will see the robot's first-person perspective in the Apple Vision Pro.

Next, Operator B can start teleoperation program by pressing the r key in the terminal.

At this time, Operator A can remotely control the robot's arms (and dexterous hands).

If the --record parameter is used, Operator B can press s key in the opened "record image" window to start recording data, and press s again to stop. This operation can be repeated as needed for multiple recordings.

p.s.1 Recorded data is stored in avp_teleoperate/teleop/utils/data by default, with usage instructions at this repo: unitree_IL_lerobot.

p.s.2 Please pay attention to your disk space size during data recording.

## Enhanced Recording Controls

The recording system has been improved with more flexible controls and quality labeling. When using the `--record` parameter, you can now:

### Key Controls (in the OpenCV "record image" window):

**When NOT recording:**
- **`s`** - Start recording a new episode
- **`ESC`** - Quit the program (no recording mode)


**When RECORDING:**
- **`r`** - Abort and delete the current recording (no save)
- **`q`** - Save episode with quality label "optimal"
- **`w`** - Save episode with quality label "suboptimal"  
- **`e`** - Save episode with quality label "recovery"

### Visual Feedback:
- The recording window displays the current status (READY TO RECORD / RECORDING)
- Available controls are shown on screen for easy reference
- Recording status is color-coded (green for ready, red for recording)

### Quality Labels:
Episodes are now saved with quality metadata in the `data.json` file:
- **"optimal"** - High-quality demonstrations for training
- **"suboptimal"** - Lower quality but still usable data
- **"recovery"** - Recovery behaviors or edge cases

This allows for better data curation and filtering during training.

### Workflow Example:
1. Press `s` to start recording
2. Perform the demonstration
3. Choose outcome:
   - Press `q` for good demonstrations (optimal quality)
   - Press `w` for acceptable demonstrations (suboptimal quality)
   - Press `e` for recovery/emergency behaviors (recovery quality)
   - Press `r` to discard bad demonstrations entirely

# Hot to setup the Package

