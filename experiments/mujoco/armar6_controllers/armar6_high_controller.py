import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../..')
os.sys.path.insert(0, '../../..')

import numpy as np
from math_tools.Quaternion import Quaternion
from armar6_low_controller import RIGHT_JOINT

class JointSpaceVMPController:
    def __init__(self, jvmp, low_ctrl=None, motion_duration=1, joint_name_list=None):
        self.low_ctrl = low_ctrl
        self.jvmp = jvmp
        self.motion_duration = motion_duration
        self.last_period_end_time = 0.0
        if joint_name_list is None:
            self.joint_name_list = RIGHT_JOINT
        else:
            self.joint_name_list = joint_name_list

        self.targets={}
        for i in range(len(self.joint_name_list)):
            self.targets[self.joint_name_list[i]] = 0.0


    def control(self, sim_duration):
        if self.low_ctrl is None:
            print("Error: low level controller cannot be None ")
            return True

        done = False
        normalizedTime = (sim_duration - self.last_period_end_time) / self.motion_duration
        if normalizedTime > 1:
            done = True
            for i in range(len(self.joint_name_list)):
                self.targets[self.joint_name_list[i]] = 0.0

            self.low_ctrl.control(self.targets)
            return done

        jvs = self.jvmp.get_vel(normalizedTime)

        for i in range(len(jvs)):
            self.targets[self.joint_name_list[i]] = jvs[i] / self.motion_duration

        self.low_ctrl.control(self.targets)
        return done


class TaskSpaceVMPController:
    def __init__(self, ts_vmp, low_ctrl=None, periodic=False, motion_duration=1, js_vmp=None, velocity_mode=False):
        self.low_ctrl = low_ctrl
        self.ts_vmp = ts_vmp
        self.js_vmp = js_vmp
        self.motion_duration = motion_duration
        self.periodic = periodic
        self.velocity_mode = velocity_mode
        self.last_period_end_time = 0.0
        if velocity_mode:
            self.get_target_func = self.ts_vmp.get_vel
        else:
            self.get_target_func = self.ts_vmp.get_target

    def pack(self, position, orientation, desired_joints, canVal):
        targets = {'position': position,
                   'orientation': orientation,
                   'desired_joints': desired_joints,
                   'canonical_value': canVal
                   }
        return targets

    def control(self, sim_duration):
        if self.low_ctrl is None:
            print("Error: low level controller cannot be None ")
            return True

        done = False
        normalizedTime = (sim_duration - self.last_period_end_time) / self.motion_duration
        if normalizedTime > 1:
            if self.periodic:
                self.last_period_end_time = self.low_ctrl.sim.data.time
                normalizedTime = 1
                # for wiping
                start = self.ts_vmp.goal.copy()
                goal = self.ts_vmp.goal.copy()
                print(start, goal)
                goal[0] = goal[0] + 0.020
                self.ts_vmp.set_start_goal(start, goal)
            else:
                done = True
                return done

        if self.js_vmp is None:
            desired_joints = None
        else:
            desired_joints = np.squeeze(np.asarray(self.js_vmp.get_target(normalizedTime)[:, 0]))

        ts_target = self.get_target_func(normalizedTime)
        target_xyz = np.asarray(ts_target[0]).flatten()
        ts_quat = ts_target[1].flatten()
        target_quat = Quaternion.normalize(ts_quat)

        # print("vmp: ", target_xyz, target_quat, normalizedTime)

        if self.velocity_mode:
            targets = self.pack(target_xyz/self.motion_duration, target_quat, desired_joints, normalizedTime)
        else:
            targets = self.pack(target_xyz, target_quat, desired_joints, normalizedTime)

        self.low_ctrl.control(targets)
        return done


class TaskSpacePositionVMPController:
    def __init__(self, vmp, low_ctrl=None, periodic=False, motion_duration=1, velocity_mode=False):
        self.low_ctrl = low_ctrl
        self.vmp = vmp
        self.motion_duration = motion_duration
        self.periodic = periodic
        self.velocity_mode = velocity_mode
        self.last_period_end_time = 0.0
        self.get_target_func = self.vmp.get_target
        self.desired_joints = None
        self.firstRun = True

    def pack(self, position, orientation, canVal, desired_joints=None):
        targets = {'position': position,
                   'orientation': orientation,
                   'nullspace': desired_joints,
                   'canonical_value': canVal
                   }
        return targets

    def control(self, sim_duration):
        if self.low_ctrl is None:
            print("Error: low level controller cannot be None ")
            return True

        self.firstRun = False
        done = False
        normalizedTime = (sim_duration - self.last_period_end_time) / self.motion_duration
        if normalizedTime > 1 and not self.firstRun:
            done = True
            self.low_ctrl.control(self.last_targets)

            return done
        else:
            target = self.get_target_func(normalizedTime)
            target_xyz = np.zeros(3)
            target_xyz[:2] = np.asarray(target).flatten()
            target_xyz[2] = self.target_z
            target_quat = Quaternion.normalize(self.target_quat)
            targets = self.pack(target_xyz, target_quat, normalizedTime, self.desired_joints)

            self.low_ctrl.control(targets)
            self.last_targets = targets
            return done


class TaskSpaceVMPWipingController:
    def __init__(self, ts_vmp, low_ctrl=None, periodic=False, motion_duration=1, js_vmp=None, velocity_mode=False):
        self.low_ctrl = low_ctrl
        self.ts_vmp = ts_vmp
        self.js_vmp = js_vmp
        self.motion_duration = motion_duration
        self.periodic = periodic
        self.velocity_mode = velocity_mode
        self.last_period_end_time = 0.0
        self.change_wipe_location_counter = 2
        if velocity_mode:
            self.get_target_func = self.ts_vmp.get_vel
        else:
            self.get_target_func = self.ts_vmp.get_target

    def pack(self, position, orientation, desired_joints, canVal):
        targets = {'position': position,
                   'orientation': orientation,
                   'desired_joints': desired_joints,
                   'canonical_value': canVal
                   }
        return targets

    def control(self, sim_duration):
        if self.low_ctrl is None:
            print("Error: low level controller cannot be None ")
            return True

        done = False
        if self.low_ctrl.restart_high_ctrl:
            self.last_period_end_time = self.low_ctrl.sim.data.time
            start = self.low_ctrl.tcp_pose.copy()
            goal = self.low_ctrl.tcp_pose.copy()
            print(start, goal)
            goal[0] = goal[0] + 0.040
            self.ts_vmp.set_start_goal(start, goal)

        normalized_time = (sim_duration - self.last_period_end_time) / self.motion_duration
        if normalized_time > 1:
            if self.periodic:
                self.last_period_end_time = self.low_ctrl.sim.data.time
                normalized_time = 1
                # for wiping
                if self.change_wipe_location_counter == 0:
                    print("change start and goal to: ")
                    start = self.ts_vmp.goal.copy()
                    goal = self.ts_vmp.goal.copy()
                    print(start, goal)
                    goal[0] = goal[0] + 0.040
                    self.ts_vmp.set_start_goal(start, goal)
                else:
                    self.change_wipe_location_counter -= 1
            else:
                done = True
                return done

        if self.js_vmp is None:
            desired_joints = None
        else:
            desired_joints = np.squeeze(np.asarray(self.js_vmp.get_target(normalized_time)[:, 0]))

        ts_target = self.get_target_func(normalized_time)
        target_xyz = np.asarray(ts_target[0]).flatten()
        ts_quat = ts_target[1].flatten()
        target_quat = Quaternion.normalize(ts_quat)

        # print("vmp: ", target_xyz, target_quat, normalizedTime)

        if self.velocity_mode:
            targets = self.pack(target_xyz/self.motion_duration, target_quat, desired_joints, normalized_time)
        else:
            targets = self.pack(target_xyz, target_quat, desired_joints, normalized_time)

        self.low_ctrl.control(targets)
        return done

