from .dex_retargeting.retargeting_config import RetargetingConfig
import tempfile
import os
import shutil
from pathlib import Path
import yaml
from enum import Enum

class HandType(Enum):
    INSPIRE_HAND = "../assets/inspire_hand/inspire_hand.yml"
    INSPIRE_HAND_Unit_Test = "../../assets/inspire_hand/inspire_hand.yml"
    UNITREE_DEX3 = "../assets/unitree_hand/unitree_dex3.yml"
    UNITREE_DEX3_Unit_Test = "../../assets/unitree_hand/unitree_dex3.yml"

class RetargetingMethod(Enum):
    VECTOR = "vector"
    DEXPILOT = "dexpilot"

class HandRetargetingConfigModifier:
    """Utility class to modify hand retargeting configuration files."""
    
    @staticmethod
    def modify_config_file(hand_type: HandType, method: RetargetingMethod) -> Path:
        """
        Modifies the hand configuration file to use the specified retargeting method.
        
        Args:
            hand_type: The type of hand to configure
            method: The retargeting method to use
            
        Returns:
            Path to the modified configuration file
        """
        config_file_path = Path(hand_type.value)
        
        # Create a temporary directory to store the modified config
        temp_dir = tempfile.mkdtemp(prefix="hand_retargeting_")
        temp_path = Path(temp_dir) / config_file_path.name
        
        if not config_file_path.exists():
            base_dir = "../assets" if not "_Unit_Test" in hand_type.name else "../../assets"
            config_file_path = Path(base_dir) / config_file_path.name
            if not config_file_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {hand_type.value}")
        
        # Read the original config
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Modify the configuration to use the specified method
        if 'left' in config and 'right' in config:
            config['left']['type'] = method.value
            config['right']['type'] = method.value
            
            # For dexpilot, we need to ensure certain parameters are set
            if method == RetargetingMethod.DEXPILOT:
                # Ensure dexpilot-specific parameters are set
                for side in ['left', 'right']:
                    config[side]['project_dist'] = config[side].get('project_dist', 0.03)
                    config[side]['escape_dist'] = config[side].get('escape_dist', 0.05)
                    
                    # DexPilot might need finger_tip_link_names and wrist_link_name
                    # We don't set these if they're already present
                    if 'finger_tip_link_names' not in config[side]:
                        if 'unitree' in hand_type.name.lower():
                            # Default finger tips for unitree dex3
                            finger_tips = []
                            side_prefix = 'left' if side == 'left' else 'right'
                            finger_tips = [
                                f"{side_prefix}_hand_thumb_2_link",
                                f"{side_prefix}_hand_index_1_link",
                                f"{side_prefix}_hand_middle_1_link"
                            ]
                            config[side]['finger_tip_link_names'] = finger_tips
                            config[side]['wrist_link_name'] = f"{side_prefix}_hand_mount"
                
        # Write the modified config to the temporary file
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, 'w') as f:
            yaml.dump(config, f)
        
        return temp_path

class HandRetargeting:
    def __init__(self, hand_type: HandType, retargeting_method: RetargetingMethod = RetargetingMethod.VECTOR):
        # Modify the configuration file to use the specified retargeting method
        try:
            modified_config_path = HandRetargetingConfigModifier.modify_config_file(hand_type, retargeting_method)
            
            if hand_type == HandType.UNITREE_DEX3:
                RetargetingConfig.set_default_urdf_dir('../assets')
            elif hand_type == HandType.UNITREE_DEX3_Unit_Test:
                RetargetingConfig.set_default_urdf_dir('../../assets')
            elif hand_type == HandType.INSPIRE_HAND:
                RetargetingConfig.set_default_urdf_dir('../assets')
            elif hand_type == HandType.INSPIRE_HAND_Unit_Test:
                RetargetingConfig.set_default_urdf_dir('../../assets')

            # Now use the modified config
            with open(modified_config_path, 'r') as f:
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
            print(f"Configuration file not found: {modified_config_path}")
            raise
        except yaml.YAMLError as e:
            print(f"YAML error while reading {modified_config_path}: {e}")
            raise
        except Exception as e:
            print(f"An error occurred: {e}")
            raise