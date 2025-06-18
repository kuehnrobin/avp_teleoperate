from .dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
import yaml
from enum import Enum
import os

class HandType(Enum):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INSPIRE_HAND = os.path.join(base_dir, "assets/inspire_hand/inspire_hand.yml")
    INSPIRE_HAND_Unit_Test = os.path.join(base_dir, "assets/inspire_hand/inspire_hand.yml")
    UNITREE_DEX3 = os.path.join(base_dir, "assets/unitree_hand/unitree_dex3.yml")
    UNITREE_DEX3_Unit_Test = os.path.join(base_dir, "assets/unitree_hand/unitree_dex3.yml")
    UNITREE_DEX3_DEXPILOT = os.path.join(base_dir, "assets/unitree_hand/unitree_dex3_dexpilot.yml")
    UNITREE_DEX3_DEXPILOT_Unit_Test = os.path.join(base_dir, "assets/unitree_hand/unitree_dex3_dexpilot.yml")

class HandRetargeting:
    def __init__(self, hand_type: HandType, retargeting_method: str = 'vector'):
        # Dynamically select the configuration based on retargeting method
        if hand_type == HandType.UNITREE_DEX3:
            if retargeting_method == 'dexpilot':
                config_hand_type = HandType.UNITREE_DEX3_DEXPILOT
            else:
                config_hand_type = HandType.UNITREE_DEX3
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            RetargetingConfig.set_default_urdf_dir(os.path.join(base_dir, 'assets'))
        elif hand_type == HandType.UNITREE_DEX3_Unit_Test:
            if retargeting_method == 'dexpilot':
                config_hand_type = HandType.UNITREE_DEX3_DEXPILOT_Unit_Test
            else:
                config_hand_type = HandType.UNITREE_DEX3_Unit_Test
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            RetargetingConfig.set_default_urdf_dir(os.path.join(base_dir, 'assets'))
        elif hand_type == HandType.INSPIRE_HAND:
            config_hand_type = HandType.INSPIRE_HAND
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            RetargetingConfig.set_default_urdf_dir(os.path.join(base_dir, 'assets'))
        elif hand_type == HandType.INSPIRE_HAND_Unit_Test:
            config_hand_type = HandType.INSPIRE_HAND_Unit_Test
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            RetargetingConfig.set_default_urdf_dir(os.path.join(base_dir, 'assets'))
        else:
            config_hand_type = hand_type
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            RetargetingConfig.set_default_urdf_dir(os.path.join(base_dir, 'assets'))

        config_file_path = Path(config_hand_type.value)

        try:
            with config_file_path.open('r') as f:
                self.cfg = yaml.safe_load(f)
                
            if 'left' not in self.cfg or 'right' not in self.cfg:
                raise ValueError("Configuration file must contain 'left' and 'right' keys.")


            left_retargeting_config = RetargetingConfig.from_dict(self.cfg['left'])
            right_retargeting_config = RetargetingConfig.from_dict(self.cfg['right'])
            self.left_retargeting = left_retargeting_config.build()
            self.right_retargeting = right_retargeting_config.build()

            self.left_retargeting_joint_names = self.left_retargeting.joint_names
            self.right_retargeting_joint_names = self.right_retargeting.joint_names

            if hand_type == HandType.UNITREE_DEX3 or hand_type == HandType.UNITREE_DEX3_Unit_Test:
                # In section "Sort by message structure" of https://support.unitree.com/home/en/G1_developer/dexterous_hand
                self.left_dex3_api_joint_names  = [ 'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
                                                    'left_hand_middle_0_joint', 'left_hand_middle_1_joint', 
                                                    'left_hand_index_0_joint', 'left_hand_index_1_joint' ]
                self.right_dex3_api_joint_names = [ 'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
                                                    'right_hand_middle_0_joint', 'right_hand_middle_1_joint',
                                                    'right_hand_index_0_joint', 'right_hand_index_1_joint' ]
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_dex3_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_dex3_api_joint_names]

                # Archive: This is the joint order of the dex-retargeting library version 0.1.1.
                # print([joint.get_name() for joint in self.left_retargeting.optimizer.robot.get_active_joints()])
                # ['left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint', 
                #  'left_hand_middle_0_joint', 'left_hand_middle_1_joint', 
                #  'left_hand_index_0_joint', 'left_hand_index_1_joint']
                # print([joint.get_name() for joint in self.right_retargeting.optimizer.robot.get_active_joints()])
                # ['right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
                #  'right_hand_middle_0_joint', 'right_hand_middle_1_joint', 
                #  'right_hand_index_0_joint', 'right_hand_index_1_joint']
            elif hand_type == HandType.INSPIRE_HAND or hand_type == HandType.INSPIRE_HAND_Unit_Test:
                self.left_inspire_api_joint_names  = [ 'L_pinky_proximal_joint', 'L_ring_proximal_joint', 'L_middle_proximal_joint',
                                                       'L_index_proximal_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_proximal_yaw_joint' ]
                self.right_inspire_api_joint_names = [ 'R_pinky_proximal_joint', 'R_ring_proximal_joint', 'R_middle_proximal_joint',
                                                       'R_index_proximal_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_proximal_yaw_joint' ]
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_inspire_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_inspire_api_joint_names]
        
        except FileNotFoundError:
            print(f"Configuration file not found: {config_file_path}")
            raise
        except yaml.YAMLError as e:
            print(f"YAML error while reading {config_file_path}: {e}")
            raise
        except Exception as e:
            print(f"An error occurred: {e}")
            raise