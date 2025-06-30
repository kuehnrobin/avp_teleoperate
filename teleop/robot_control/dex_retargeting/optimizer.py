from abc import abstractmethod
from typing import List, Optional

import nlopt
import numpy as np
import torch

from .kinematics_adaptor import KinematicAdaptor, MimicJointKinematicAdaptor
from .robot_wrapper import RobotWrapper


class Optimizer:
    retargeting_type = "BASE"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_human_indices: np.ndarray,
    ):
        self.robot = robot
        self.num_joints = robot.dof

        joint_names = robot.dof_joint_names
        idx_pin2target = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(f"Joint {target_joint_name} given does not appear to be in robot XML.")
            idx_pin2target.append(joint_names.index(target_joint_name))
        self.target_joint_names = target_joint_names
        self.idx_pin2target = np.array(idx_pin2target)

        self.idx_pin2fixed = np.array([i for i in range(robot.dof) if i not in idx_pin2target], dtype=int)
        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(idx_pin2target))
        self.opt_dof = len(idx_pin2target)  # This dof includes the mimic joints

        # Target
        self.target_link_human_indices = target_link_human_indices

        # Free joint
        link_names = robot.link_names
        self.has_free_joint = len([name for name in link_names if "dummy" in name]) >= 6

        # Kinematics adaptor
        self.adaptor: Optional[KinematicAdaptor] = None

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        if joint_limits.shape != (self.opt_dof, 2):
            raise ValueError(f"Expect joint limits have shape: {(self.opt_dof, 2)}, but get {joint_limits.shape}")
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

    def get_link_indices(self, target_link_names):
        return [self.robot.get_link_index(link_name) for link_name in target_link_names]

    def set_kinematic_adaptor(self, adaptor: KinematicAdaptor):
        self.adaptor = adaptor

        # Remove mimic joints from fixed joint list
        if isinstance(adaptor, MimicJointKinematicAdaptor):
            fixed_idx = self.idx_pin2fixed
            mimic_idx = adaptor.idx_pin2mimic
            new_fixed_id = np.array([x for x in fixed_idx if x not in mimic_idx], dtype=int)
            self.idx_pin2fixed = new_fixed_id

    def retarget(self, ref_value, fixed_qpos, last_qpos):
        """
        Compute the retargeting results using non-linear optimization
        Args:
            ref_value: the reference value in cartesian space as input, different optimizer has different reference
            fixed_qpos: the fixed value (not optimized) in retargeting, consistent with self.fixed_joint_names
            last_qpos: the last retargeting results or initial value, consistent with function return

        Returns: joint position of robot, the joint order and dim is consistent with self.target_joint_names

        """
        if len(fixed_qpos) != len(self.idx_pin2fixed):
            raise ValueError(
                f"Optimizer has {len(self.idx_pin2fixed)} joints but non_target_qpos {fixed_qpos} is given"
            )
        
        print(f"[DEBUG] Optimizer.retarget called")
        print(f"[DEBUG] Optimizer class: {type(self).__name__}")
        print(f"[DEBUG] target_joint_names: {self.target_joint_names}")
        print(f"[DEBUG] idx_pin2target: {self.idx_pin2target}")
        print(f"[DEBUG] ref_value shape: {ref_value.shape}, type: {type(ref_value)}")
        print(f"[DEBUG] ref_value contains NaN: {np.isnan(ref_value).any()}")
        print(f"[DEBUG] ref_value contains Inf: {np.isinf(ref_value).any()}")
        print(f"[DEBUG] fixed_qpos shape: {fixed_qpos.shape}, values: {fixed_qpos}")
        print(f"[DEBUG] last_qpos shape: {np.array(last_qpos).shape}, values: {last_qpos}")
        
        # Debug joint name to value mapping
        if len(last_qpos) == len(self.target_joint_names):
            print(f"[DEBUG] Joint name to value mapping:")
            for i, (name, value) in enumerate(zip(self.target_joint_names, last_qpos)):
                print(f"[DEBUG]   {i}: {name} = {value:.6f} rad ({np.rad2deg(value):.2f}°)")
        else:
            print(f"[DEBUG] WARNING: last_qpos length {len(last_qpos)} != target_joint_names length {len(self.target_joint_names)}")
        
        print(f"[DEBUG] last_qpos shape: {np.array(last_qpos).shape}, values: {last_qpos}")
        print(f"[DEBUG] last_qpos contains NaN: {np.isnan(last_qpos).any()}")
        print(f"[DEBUG] last_qpos contains Inf: {np.isinf(last_qpos).any()}")
        
        # Validate inputs
        if np.isnan(ref_value).any() or np.isinf(ref_value).any():
            print(f"[ERROR] Invalid ref_value detected in retarget!")
            return np.array(last_qpos, dtype=np.float32)
        
        if np.isnan(last_qpos).any() or np.isinf(last_qpos).any():
            print(f"[ERROR] Invalid last_qpos detected in retarget!")
            return np.zeros(len(last_qpos), dtype=np.float32)
        
        objective_fn = self.get_objective_function(ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32))

        self.opt.set_min_objective(objective_fn)
        print(f"[DEBUG] About to call opt.optimize with last_qpos: {last_qpos}")
        try:
            qpos = self.opt.optimize(last_qpos)
            print(f"[DEBUG] Optimization successful, result: {qpos}")
            
            # Debug the final result mapping
            if len(qpos) == len(self.target_joint_names):
                print(f"[DEBUG] Final optimized joint values:")
                for i, (name, value) in enumerate(zip(self.target_joint_names, qpos)):
                    print(f"[DEBUG]   {i}: {name} = {value:.6f} rad ({np.rad2deg(value):.2f}°)")
                    
                # Specifically highlight thumb joints
                for i, (name, value) in enumerate(zip(self.target_joint_names, qpos)):
                    if "thumb" in name.lower():
                        print(f"[THUMB] {name} = {value:.6f} rad ({np.rad2deg(value):.2f}°)")
            
            return np.array(qpos, dtype=np.float32)
        except RuntimeError as e:
            print(f"[ERROR] RuntimeError in optimization: {e}")
            return np.array(last_qpos, dtype=np.float32)
        except ValueError as e:
            print(f"[ERROR] ValueError in optimization: {e}")
            return np.array(last_qpos, dtype=np.float32)
        except Exception as e:
            print(f"[ERROR] Unexpected error in optimization: {e}")
            import traceback
            traceback.print_exc()
            return np.array(last_qpos, dtype=np.float32)

    @abstractmethod
    def get_objective_function(self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        pass

    @property
    def fixed_joint_names(self):
        joint_names = self.robot.dof_joint_names
        return [joint_names[i] for i in self.idx_pin2fixed]


class PositionOptimizer(Optimizer):
    retargeting_type = "POSITION"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.body_names = target_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta

        # Sanity check and cache link indices
        self.target_link_indices = self.get_link_indices(target_link_names)

        self.opt.set_ftol_abs(1e-5)

    def get_objective_function(self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [self.robot.get_link_pose(index) for index in self.target_link_indices]
            body_pos = np.stack([pose[:3, 3] for pose in target_link_poses], axis=0)  # (n ,3)

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Loss term for kinematics retargeting based on 3D position error
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.target_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(qpos, index)[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                # Compute the gradient to the qpos
                grad_qpos = np.matmul(grad_pos, jacobians)
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class VectorOptimizer(Optimizer):
    retargeting_type = "VECTOR"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_origin_link_names: List[str],
        target_task_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
        scaling=1.0,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="mean")
        self.norm_delta = norm_delta
        self.scaling = scaling

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(set(target_origin_link_names).union(set(target_task_link_names)))
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor([self.computed_link_names.index(name) for name in target_task_link_names])

        # Cache link indices that will involve in kinematics computation
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

    def get_objective_function(self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_vec = torch.as_tensor(target_vector) * self.scaling
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [self.robot.get_link_pose(index) for index in self.computed_link_indices]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(qpos, index)[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class DexPilotOptimizerAnyTeleop(Optimizer):
    """Retargeting optimizer using the method proposed in DexPilot

    This is a broader adaptation of the original optimizer delineated in the DexPilot paper.
    While the initial DexPilot study focused solely on the four-fingered Allegro Hand, this version of the optimizer
    embraces the same principles for both four-fingered and five-fingered hands. It projects the distance between the
    thumb and the other fingers to facilitate more stable grasping.
    Reference: https://arxiv.org/abs/1910.03135

    Args:
        robot:
        target_joint_names:
        finger_tip_link_names:
        wrist_link_name:
        gamma:
        project_dist:
        escape_dist:
        eta1:
        eta2:
        scaling:
    """

    retargeting_type = "DEXPILOT"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        finger_tip_link_names: List[str],
        wrist_link_name: str,
        target_link_human_indices: Optional[np.ndarray] = None,
        huber_delta=0.03,
        norm_delta=4e-3,
        # DexPilot parameters
        #gamma=2.5e-3,
        project_dist=0.03,
        escape_dist=0.05,
        eta1=1e-4,
        eta2=3e-2,
        scaling=1.0,
    ):
        if len(finger_tip_link_names) < 2 or len(finger_tip_link_names) > 5:
            raise ValueError(
                f"DexPilot optimizer can only be applied to hands with 2 to 5 fingers, but got "
                f"{len(finger_tip_link_names)} fingers."
            )
        self.num_fingers = len(finger_tip_link_names)

        origin_link_index, task_link_index = self.generate_link_indices(self.num_fingers)

        if target_link_human_indices is None:
            target_link_human_indices = (np.stack([origin_link_index, task_link_index], axis=0) * 4).astype(int)
        link_names = [wrist_link_name] + finger_tip_link_names
        target_origin_link_names = [link_names[index] for index in origin_link_index]
        target_task_link_names = [link_names[index] for index in task_link_index]

        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.scaling = scaling
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.norm_delta = norm_delta

        # DexPilot parameters
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(set(target_origin_link_names).union(set(target_task_link_names)))
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor([self.computed_link_names.index(name) for name in target_task_link_names])

        # Sanity check and cache link indices
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

        # DexPilot cache
        self.projected, self.s2_project_index_origin, self.s2_project_index_task, self.projected_dist = (
            self.set_dexpilot_cache(self.num_fingers, eta1, eta2)
        )

    @staticmethod
    def generate_link_indices(num_fingers):
        """
        Example:
        >>> generate_link_indices(4)
        ([2, 3, 4, 3, 4, 4, 0, 0, 0, 0], [1, 1, 1, 2, 2, 3, 1, 2, 3, 4])
        """
        origin_link_index = []
        task_link_index = []

        # Add indices for connections between fingers
        for i in range(1, num_fingers):
            for j in range(i + 1, num_fingers + 1):
                origin_link_index.append(j)
                task_link_index.append(i)

        # Add indices for connections to the base (0)
        for i in range(1, num_fingers + 1):
            origin_link_index.append(0)
            task_link_index.append(i)

        return origin_link_index, task_link_index

    @staticmethod
    def set_dexpilot_cache(num_fingers, eta1, eta2):
        """
        Example:
        >>> set_dexpilot_cache(4, 0.1, 0.2)
        (array([False, False, False, False, False, False]),
        [1, 2, 2],
        [0, 0, 1],
        array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        """
        projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)

        s2_project_index_origin = []
        s2_project_index_task = []
        for i in range(0, num_fingers - 2):
            for j in range(i + 1, num_fingers - 1):
                s2_project_index_origin.append(j)
                s2_project_index_task.append(i)

        projected_dist = np.array([eta1] * (num_fingers - 1) + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2))

        return projected, s2_project_index_origin, s2_project_index_task, projected_dist

    def get_objective_function(self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        # Update projection indicator
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin], self.projected[:len_s1][self.s2_project_index_task]
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        )

        # Update weight vector
        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)

        # We change the weight to 10 instead of 1 here, for vector originate from wrist to fingertips
        # This ensures better intuitive mapping due wrong pose detection
        weight = torch.from_numpy(
            np.concatenate([weight, np.ones(self.num_fingers, dtype=np.float32) * len_proj + self.num_fingers])
        )

        # Compute reference distance vector
        normal_vec = target_vector * self.scaling  # (10, 3)
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (6, 3)
        projected_vec = dir_vec * self.projected_dist[:, None]  # (6, 3)

        # Compute final reference vector
        reference_vec = np.where(self.projected[:, None], projected_vec, normal_vec[:len_proj])  # (6, 3)
        reference_vec = np.concatenate([reference_vec, normal_vec[len_proj:]], axis=0)  # (10, 3)
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [self.robot.get_link_pose(index) for index in self.computed_link_indices]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            # Different from the original DexPilot, we use huber loss here instead of the squared dist
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = (
                self.huber_loss(vec_dist, torch.zeros_like(vec_dist)) * weight / (robot_vec.shape[0])
            ).sum()
            huber_distance = huber_distance.sum()
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(qpos, index)[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                # In the original DexPilot, γ = 2.5 × 10−3 is a weight on regularizing the Allegro angles to zero
                # which is equivalent to fully opened the hand
                # In our implementation, we regularize the joint angles to the previous joint angles
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class DexPilotOptimizer(Optimizer):
    """Retargeting optimizer using the method proposed in DexPilot
    This is a broader adaptation of the original optimizer delineated in the DexPilot paper.
    While the initial DexPilot study focused solely on the four-fingered Allegro Hand, this version of the optimizer
    embraces the same principles for both four-fingered and five-fingered hands. It projects the distance between the
    thumb and the other fingers to facilitate more stable grasping.
    Reference: https://arxiv.org/abs/1910.03135
    Args:
        robot:
        target_joint_names:
        finger_tip_link_names:
        wrist_link_name:
        gamma:
        project_dist:
        escape_dist:
        eta1:
        eta2:
        scaling:
    """
    retargeting_type = "DEXPILOT"
    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        finger_tip_link_names: List[str],
        wrist_link_name: str,
        target_link_human_indices: Optional[np.ndarray] = None,
        huber_delta=0.03,
        norm_delta=4e-3,
        # DexPilot parameters
        gamma=6e-3,  # Increased from 2.5e-3 to encourage more open hand position
        project_dist=0.03,
        escape_dist=0.05,
        eta1=1e-4,
        eta2=0,
        scaling=1.0,
    ):
        if len(finger_tip_link_names) < 2 or len(finger_tip_link_names) > 5:
            raise ValueError(
                f"DexPilot optimizer can only be applied to hands with 2 to 5 fingers, but got "
                f"{len(finger_tip_link_names)} fingers."
            )
        self.num_fingers = len(finger_tip_link_names)
        origin_link_index, task_link_index = self.generate_link_indices(self.num_fingers)
        if target_link_human_indices is None:
            target_link_human_indices = (np.stack([origin_link_index, task_link_index], axis=0) * 4).astype(int)
        link_names = [wrist_link_name] + finger_tip_link_names
        target_origin_link_names = [link_names[index] for index in origin_link_index]
        target_task_link_names = [link_names[index] for index in task_link_index]
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.scaling = scaling
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.norm_delta = norm_delta
        self.gamma = gamma  # Store gamma as an instance variable
        # DexPilot parameters
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2
        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(set(target_origin_link_names).union(set(target_task_link_names)))
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor([self.computed_link_names.index(name) for name in target_task_link_names])
        # Sanity check and cache link indices
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)
        
        # Set more relaxed tolerances to avoid roundoff-limited errors
        #self.opt.set_ftol_abs(1e-6)
        self.opt.set_ftol_abs(1e-4)
        self.opt.set_ftol_rel(1e-4)
        self.opt.set_xtol_abs(1e-6)
        self.opt.set_xtol_rel(1e-6)
        # DexPilot cache
        self.projected, self.s2_project_index_origin, self.s2_project_index_task, self.projected_dist = (
            self.set_dexpilot_cache(self.num_fingers, eta1, eta2)
        )
    @staticmethod
    def generate_link_indices(num_fingers):
        """
        Example:
        >>> generate_link_indices(4)
        ([2, 3, 4, 3, 4, 4, 0, 0, 0, 0], [1, 1, 1, 2, 2, 3, 1, 2, 3, 4])
        """
        origin_link_index = []
        task_link_index = []
        # Add indices for connections between fingers
        for i in range(1, num_fingers):
            for j in range(i + 1, num_fingers + 1):
                origin_link_index.append(j)
                task_link_index.append(i)
        # Add indices for connections to the base (0)
        for i in range(1, num_fingers + 1):
            origin_link_index.append(0)
            task_link_index.append(i)
        return origin_link_index, task_link_index
    @staticmethod
    def set_dexpilot_cache(num_fingers, eta1, eta2):
        """
        Example:
        >>> set_dexpilot_cache(4, 0.1, 0.2)
        (array([False, False, False, False, False, False]),
        [1, 2, 2],
        [0, 0, 1],
        array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        """
        projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)
        s2_project_index_origin = []
        s2_project_index_task = []
        for i in range(0, num_fingers - 2):
            for j in range(i + 1, num_fingers - 1):
                s2_project_index_origin.append(j)
                s2_project_index_task.append(i)
        projected_dist = np.array([eta1] * (num_fingers - 1) + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2))
        return projected, s2_project_index_origin, s2_project_index_task, projected_dist
    def get_objective_function(self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        debug_mode = True  # Enable debug for DYNAMIC EARTH CORE MISSION! 🌍🔄
        if debug_mode:
            print(f"[DEBUG] DexPilot get_objective_function called")
            print(f"[DEBUG] target_vector shape: {target_vector.shape}")
            print(f"[DEBUG] target_vector min/max: {target_vector.min():.6f} / {target_vector.max():.6f}")
            print(f"[DEBUG] fixed_qpos shape: {fixed_qpos.shape}")
            print(f"[DEBUG] last_qpos shape: {last_qpos.shape}")
            print(f"[DEBUG] computed_link_names: {self.computed_link_names}")
            print(f"[DEBUG] target_joint_names: {self.target_joint_names}")
        
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2
        
        if debug_mode:
            print(f"[DEBUG] len_proj: {len_proj}, len_s1: {len_s1}, len_s2: {len_s2}")
        
        # Update projection indicator
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        if debug_mode:
            print(f"[DEBUG] target_vec_dist: {target_vec_dist}")
        
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin], self.projected[:len_s1][self.s2_project_index_task]
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        )
        # Update weight vector
        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)
        # We change the weight to 10 instead of 1 here, for vector originate from wrist to fingertips
        # This ensures better intuitive mapping due wrong pose detection
        weight = torch.from_numpy(
            np.concatenate([weight, np.ones(self.num_fingers, dtype=np.float32) * len_proj + self.num_fingers])
        )
        if debug_mode:
            print(f"[DEBUG] weight shape: {weight.shape}, weight: {weight}")
        
        # Compute reference distance vector
        normal_vec = target_vector * self.scaling  # (10, 3)
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (6, 3)
        projected_vec = dir_vec * self.projected_dist[:, None]  # (6, 3)
        # Compute final reference vector
        reference_vec = np.where(self.projected[:, None], projected_vec, normal_vec[:len_proj])  # (6, 3)
        reference_vec = np.concatenate([reference_vec, normal_vec[len_proj:]], axis=0)  # (10, 3)
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)
        
        if debug_mode:
            print(f"[DEBUG] torch_target_vec shape: {torch_target_vec.shape}")
            print(f"[DEBUG] torch_target_vec contains NaN: {torch.isnan(torch_target_vec).any()}")
            print(f"[DEBUG] torch_target_vec contains Inf: {torch.isinf(torch_target_vec).any()}")
        
        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            try:
                if debug_mode:
                    print(f"[DEBUG] Objective function called with x shape: {x.shape}")
                    print(f"[DEBUG] x values: {x}")
                    print(f"[DEBUG] x contains NaN: {np.isnan(x).any()}")
                    print(f"[DEBUG] x contains Inf: {np.isinf(x).any()}")
                
                if np.isnan(x).any() or np.isinf(x).any():
                    if debug_mode:
                        print(f"[ERROR] Invalid x values detected!")
                    return float('inf')
                
                qpos[self.idx_pin2target] = x
                # Kinematics forwarding for qpos
                if self.adaptor is not None:
                    qpos[:] = self.adaptor.forward_qpos(qpos)[:]
                self.robot.compute_forward_kinematics(qpos)
                target_link_poses = [self.robot.get_link_pose(index) for index in self.computed_link_indices]
                body_pos = np.array([pose[:3, 3] for pose in target_link_poses])
                if debug_mode:
                    print(f"[DEBUG] body_pos shape: {body_pos.shape}")
                
                # Torch computation for accurate loss and grad
                torch_body_pos = torch.as_tensor(body_pos)
                torch_body_pos.requires_grad_()
                # Index link for computation
                origin_link_pos = torch_body_pos[self.origin_link_indices, :]
                task_link_pos = torch_body_pos[self.task_link_indices, :]
                robot_vec = task_link_pos - origin_link_pos
                if debug_mode:
                    print(f"[DEBUG] robot_vec shape: {robot_vec.shape}")
                
                # Loss term for kinematics retargeting based on 3D position error
                # Different from the original DexPilot, we use huber loss here instead of the squared dist
                vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
                if debug_mode:
                    print(f"[DEBUG] vec_dist: {vec_dist}")
                
                huber_distance = (
                    self.huber_loss(vec_dist, torch.zeros_like(vec_dist)) * weight / (robot_vec.shape[0])
                ).sum()
                huber_distance = huber_distance.sum()
                if debug_mode:
                    print(f"[DEBUG] huber_distance: {huber_distance}")

                # Add penalty for thumb being too low
                # Use actual link names from config
                thumb_tip_idx = None
                index_tip_idx = None  
                middle_tip_idx = None
                
                # Find link indices by searching for patterns in computed_link_names
                for i, name in enumerate(self.computed_link_names):
                    if "thumb" in name and ("tip" in name or "2_link" in name):
                        thumb_tip_idx = i
                    elif "index" in name and ("tip" in name or "1_link" in name):
                        index_tip_idx = i
                    elif "middle" in name and ("tip" in name or "1_link" in name):
                        middle_tip_idx = i
                
                if debug_mode:
                    print(f"[DEBUG] Found link indices - thumb: {thumb_tip_idx}, index: {index_tip_idx}, middle: {middle_tip_idx}")
                
                # Skip penalties if we can't find the required links
                if thumb_tip_idx is None or index_tip_idx is None or middle_tip_idx is None:
                    if debug_mode:
                        print(f"[DEBUG] Skipping penalties - missing link indices")
                    result = huber_distance.cpu().detach().item()
                    if debug_mode:
                        print(f"[DEBUG] Final result (no penalties): {result}")
                    
                    if grad.size > 0:
                        jacobians = []
                        for i, index in enumerate(self.computed_link_indices):
                            link_body_jacobian = self.robot.compute_single_link_local_jacobian(qpos, index)[:3, ...]
                            link_pose = target_link_poses[i]
                            link_rot = link_pose[:3, :3]
                            link_kinematics_jacobian = link_rot @ link_body_jacobian
                            jacobians.append(link_kinematics_jacobian)
                        jacobians = np.stack(jacobians, axis=0)
                        huber_distance.backward()
                        grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
                        if self.adaptor is not None:
                            jacobians = self.adaptor.backward_jacobian(jacobians)
                        else:
                            jacobians = jacobians[..., self.idx_pin2target]
                        grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                        grad_qpos = grad_qpos.mean(1).sum(0)
                        grad_qpos += 2 * self.gamma * x
                        grad[:] = grad_qpos[:]
                        if debug_mode:
                            print(f"[DEBUG] Gradient computed, norm: {np.linalg.norm(grad_qpos)}")
                    return result

                thumb_pos = torch_body_pos[thumb_tip_idx, :]
                index_pos = torch_body_pos[index_tip_idx, :]
                middle_pos = torch_body_pos[middle_tip_idx, :]

                thumb_index_diff = thumb_pos[2] - index_pos[2]
                thumb_middle_diff = thumb_pos[2] - middle_pos[2]

                thumb_penalty = 0.0
                if thumb_index_diff.item() < -0.02:  # If thumb is more than 2 cm below index finger
                    penalty_val = 300 * (thumb_index_diff + 0.02)**2  # Increased from 100 to 300
                    thumb_penalty += penalty_val.detach().item() if hasattr(penalty_val, 'detach') else penalty_val
                if thumb_middle_diff.item() < -0.02:  # If thumb is more than 2 cm below middle finger
                    penalty_val = 300 * (thumb_middle_diff + 0.02)**2  # Increased from 100 to 300
                    thumb_penalty += penalty_val.detach().item() if hasattr(penalty_val, 'detach') else penalty_val

                # Add penalty for thumb_0 joint angle deviating from optimal range
                thumb_0_idx = None
                for i, joint_name in enumerate(self.target_joint_names):
                    if "thumb_0_joint" in joint_name:
                        thumb_0_idx = i
                        break
                
                thumb_angle_penalty = 0.0
                if thumb_0_idx is not None:
                    thumb_0_angle = x[thumb_0_idx]
                    
                    # DYNAMIC EARTH CORE MISSION: Adaptive thumb behavior! 🌍🔄
                    # Check if pinching is happening by looking at target distances
                    thumb_index_target_dist = torch.norm(torch_target_vec[0, :]) if len(torch_target_vec) > 0 else float('inf')
                    thumb_middle_target_dist = torch.norm(torch_target_vec[1, :]) if len(torch_target_vec) > 1 else float('inf')
                    
                    # Detect if user is trying to pinch (target distance < 5cm)
                    is_pinching = thumb_index_target_dist < 0.05 or thumb_middle_target_dist < 0.05
                    
                    if is_pinching:
                        # PINCHING MODE: Target -60° for optimal thumb-to-index opposition!
                        # Target -60° specifically for better pinching geometry
                        optimal_target = np.deg2rad(-60.0)  # TARGET -60° for pinching! 🤏
                        hard_min = np.deg2rad(-60.0)        # Joint limit: -60°
                        hard_max = np.deg2rad(60.0)         # Joint limit: +60°
                        
                        # GENTLE penalties when pinching - target -60° specifically!
                        if thumb_0_angle < hard_min:
                            penalty_val = 50 * (thumb_0_angle - hard_min)**2  # Gentle boundary
                            thumb_angle_penalty += penalty_val
                            if debug_mode:
                                print(f"[EARTH-PINCH] 🌍🤏 Gentle penalty at -60° limit: {penalty_val:.6f}")
                        elif thumb_0_angle > hard_max:
                            penalty_val = 50 * (thumb_0_angle - hard_max)**2  # Gentle boundary
                            thumb_angle_penalty += penalty_val
                            if debug_mode:
                                print(f"[EARTH-PINCH] 🌍🤏 Gentle penalty at +60° limit: {penalty_val:.6f}")
                        else:
                            # STRONG bias toward -60° when pinching for optimal geometry!
                            bias_penalty = 200 * (thumb_0_angle - optimal_target)**2  # Strong bias toward -60°!
                            thumb_angle_penalty += bias_penalty
                            if debug_mode:
                                print(f"[EARTH-PINCH] 🌍🤏 TARGETING -60° for pinching! Bias: {bias_penalty:.6f}")
                    else:
                        # NON-PINCHING MODE: Strong bias toward -45° (Earth Core default)
                        optimal_target = np.deg2rad(-45.0)  # TARGET: -45° (EARTH CORE!)
                        hard_min = np.deg2rad(-60.0)        # Hard limit: -60°
                        hard_max = np.deg2rad(-30.0)        # Hard limit: -30°
                        
                        # NUCLEAR penalties when NOT pinching - keep thumb at -45°
                        if thumb_0_angle > hard_max:
                            penalty_val = 5000 * (thumb_0_angle - hard_max)**2  # NUCLEAR penalty!
                            thumb_angle_penalty += penalty_val
                            if debug_mode:
                                print(f"[EARTH] 🌍⛏️ Thumb above -30°! NUCLEAR penalty: {penalty_val:.6f}")
                        elif thumb_0_angle < hard_min:
                            penalty_val = 1500 * (thumb_0_angle - hard_min)**2  # Strong boundary
                            thumb_angle_penalty += penalty_val
                            if debug_mode:
                                print(f"[EARTH] Thumb below -60°! Strong penalty: {penalty_val:.6f}")
                        else:
                            # NUCLEAR bias toward -45° when not pinching
                            bias_penalty = 1000 * (thumb_0_angle - optimal_target)**2  # NUCLEAR!
                            thumb_angle_penalty += bias_penalty
                            if debug_mode:
                                print(f"[EARTH] 🌍⛏️ DRILLING toward -45°! Nuclear bias: {bias_penalty:.6f}")
                    
                    if debug_mode:
                        print(f"[DEBUG] thumb_0_angle: {thumb_0_angle:.5f} rad ({np.rad2deg(thumb_0_angle):.2f}°)")
                        print(f"[DEBUG] is_pinching: {is_pinching}, thumb_index_dist: {thumb_index_target_dist:.4f}, thumb_middle_dist: {thumb_middle_target_dist:.4f}")

                # Add penalty for gaps between thumb and primary fingers during pinching
                thumb_index_target_dist = torch.norm(torch_target_vec[0, :]) if len(torch_target_vec) > 0 else float('inf')
                thumb_middle_target_dist = torch.norm(torch_target_vec[1, :]) if len(torch_target_vec) > 1 else float('inf')

                gap_penalty = 0.0
                if thumb_index_target_dist < 0.03:  # If target distance is less than 3 cm (pinching)
                    thumb_index_actual_dist = torch.norm(thumb_pos - index_pos)
                    if thumb_index_actual_dist > 0.03:
                        gap_penalty_val = 200 * (thumb_index_actual_dist - 0.03)**2
                        gap_penalty += gap_penalty_val.detach().item() if hasattr(gap_penalty_val, 'detach') else gap_penalty_val.item()

                if thumb_middle_target_dist < 0.03:  # If target distance is less than 3 cm (pinching)
                    thumb_middle_actual_dist = torch.norm(thumb_pos - middle_pos)
                    if thumb_middle_actual_dist > 0.03:
                        gap_penalty_val = 200 * (thumb_middle_actual_dist - 0.03)**2
                        gap_penalty += gap_penalty_val.detach().item() if hasattr(gap_penalty_val, 'detach') else gap_penalty_val.item()

                result = huber_distance.cpu().detach().item() + thumb_penalty + thumb_angle_penalty + gap_penalty
                if debug_mode:
                    print(f"[DEBUG] Final result: {result} (huber: {huber_distance.cpu().detach().item()}, thumb: {thumb_penalty}, angle: {thumb_angle_penalty}, gap: {gap_penalty})")
                
                if np.isnan(result) or np.isinf(result):
                    if debug_mode:
                        print(f"[ERROR] Invalid result detected: {result}")
                    return float('inf')

                if grad.size > 0:
                    jacobians = []
                    for i, index in enumerate(self.computed_link_indices):
                        link_body_jacobian = self.robot.compute_single_link_local_jacobian(qpos, index)[:3, ...]
                        link_pose = target_link_poses[i]
                        link_rot = link_pose[:3, :3]
                        link_kinematics_jacobian = link_rot @ link_body_jacobian
                        jacobians.append(link_kinematics_jacobian)
                    # Note: the joint order in this jacobian is consistent pinocchio
                    jacobians = np.stack(jacobians, axis=0)
                    huber_distance.backward()
                    grad_pos = torch_body_pos.grad.detach().cpu().numpy()[:, None, :]

                    # Compute gradient of thumb position penalty term
                    if thumb_index_diff.item() < -0.02:
                        grad_pos[thumb_tip_idx, 0, 2] += 600 * (thumb_index_diff.detach() + 0.02).item()  # Updated gradient weight
                        grad_pos[index_tip_idx, 0, 2] -= 600 * (thumb_index_diff.detach() + 0.02).item()
                    if thumb_middle_diff.item() < -0.02:
                        grad_pos[thumb_tip_idx, 0, 2] += 600 * (thumb_middle_diff.detach() + 0.02).item()  # Updated gradient weight
                        grad_pos[middle_tip_idx, 0, 2] -= 600 * (thumb_middle_diff.detach() + 0.02).item()

                    # Compute gradient of gap penalty term
                    thumb_index_actual_dist = torch.norm(thumb_pos - index_pos)
                    thumb_middle_actual_dist = torch.norm(thumb_pos - middle_pos)

                    # Pinching gap penalties
                    if thumb_index_target_dist < 0.03 and thumb_index_actual_dist > 0.03:
                        direction = (thumb_pos - index_pos).detach().cpu().numpy()
                        if np.linalg.norm(direction) > 1e-6:
                            direction = direction / np.linalg.norm(direction)
                        grad_pos[thumb_tip_idx, 0, :] += 400 * (thumb_index_actual_dist.detach() - 0.03).item() * direction
                        grad_pos[index_tip_idx, 0, :] -= 400 * (thumb_index_actual_dist.detach() - 0.03).item() * direction

                    if thumb_middle_target_dist < 0.03 and thumb_middle_actual_dist > 0.03:
                        direction = (thumb_pos - middle_pos).detach().cpu().numpy()
                        if np.linalg.norm(direction) > 1e-6:
                            direction = direction / np.linalg.norm(direction)
                        grad_pos[thumb_tip_idx, 0, :] += 400 * (thumb_middle_actual_dist.detach() - 0.03).item() * direction
                        grad_pos[middle_tip_idx, 0, :] -= 400 * (thumb_middle_actual_dist.detach() - 0.03).item() * direction

                    # Convert the jacobian from pinocchio order to target order
                    if self.adaptor is not None:
                        jacobians = self.adaptor.backward_jacobian(jacobians)
                    else:
                        jacobians = jacobians[..., self.idx_pin2target]

                    # Final gradient computation - combine all penalties
                    grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                    grad_qpos = grad_qpos.mean(1).sum(0)
                    
                    # Add the joint angle penalty gradient and regularization
                    if thumb_0_idx is not None:
                        # DYNAMIC EARTH CORE GRADIENT: Adaptive based on pinching state 🌍🔄
                        thumb_index_target_dist = torch.norm(torch_target_vec[0, :]) if len(torch_target_vec) > 0 else float('inf')
                        thumb_middle_target_dist = torch.norm(torch_target_vec[1, :]) if len(torch_target_vec) > 1 else float('inf')
                        is_pinching = thumb_index_target_dist < 0.05 or thumb_middle_target_dist < 0.05
                        
                        if is_pinching:
                            # PINCHING GRADIENTS: Strong bias toward -60° for optimal pinching!
                            optimal_target = np.deg2rad(-60.0)  # TARGET -60° for pinching! 🤏
                            hard_min = np.deg2rad(-60.0)        # Joint limit: -60°
                            hard_max = np.deg2rad(60.0)         # Joint limit: +60°
                            
                            if thumb_0_angle < hard_min:
                                # Gentle gradient at joint boundary
                                grad_qpos[thumb_0_idx] += 100 * (thumb_0_angle - hard_min)  # Gentle boundary push
                            elif thumb_0_angle > hard_max:
                                # Gentle gradient at joint boundary
                                grad_qpos[thumb_0_idx] += 100 * (thumb_0_angle - hard_max)  # Gentle boundary push
                            else:
                                # STRONG bias toward -60° when pinching for optimal geometry!
                                grad_qpos[thumb_0_idx] += 400 * (thumb_0_angle - optimal_target)  # STRONG bias toward -60°!
                        else:
                            # NON-PINCHING GRADIENTS: Nuclear bias toward -45°
                            optimal_target = np.deg2rad(-45.0)  # Target: -45° (EARTH CORE!)
                            hard_min = np.deg2rad(-60.0)        # Hard limit: -60°
                            hard_max = np.deg2rad(-30.0)        # Hard limit: -30°
                            
                            if thumb_0_angle > hard_max:
                                # NUCLEAR gradient pushes thumb away from being above -30°
                                grad_qpos[thumb_0_idx] += 10000 * (thumb_0_angle - hard_max)  # NUCLEAR push toward -30°!
                            elif thumb_0_angle < hard_min:
                                # Strong gradient pushes thumb away from hard minimum
                                grad_qpos[thumb_0_idx] += 3000 * (thumb_0_angle - hard_min)  # Strong push away from -60°
                            else:
                                # NUCLEAR gradient biases thumb toward optimal target (-45°)
                                grad_qpos[thumb_0_idx] += 2000 * (thumb_0_angle - optimal_target)  # NUCLEAR bias toward -45°!
                    
                    grad_qpos += 2 * self.gamma * x  # Add regularization
                    
                    if np.isnan(grad_qpos).any() or np.isinf(grad_qpos).any():
                        if debug_mode:
                            print(f"[ERROR] Invalid gradient detected!")
                        grad[:] = np.zeros_like(grad_qpos)
                    else:
                        grad[:] = grad_qpos[:]
                        if debug_mode:
                            print(f"[DEBUG] Gradient computed, norm: {np.linalg.norm(grad_qpos)}")
                        
                return result
            except Exception as e:
                if debug_mode:
                    print(f"[ERROR] Exception in objective function: {e}")
                    import traceback
                    traceback.print_exc()
                return float('inf')
        return objective