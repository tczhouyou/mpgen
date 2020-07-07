import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../..')
os.sys.path.insert(0, '../../..')

import mujoco_py
import numpy as np
from armar6_controllers.armar6_high_controller import TaskSpaceVMPController, TaskSpacePositionVMPController
from armar6_controllers.armar6_low_controller import RIGHT_HAND_JOINT_CONFIG, RIGHT_HAND_JOINT, get_actuator_data, JointController

ENV_DIR = '/experiments/mujoco/robot-models/armar6-mujoco/environment/'
EXP_DIR = '/experiments/mujoco/hitball/'
INIT_BALL_POS = np.array([0.5, 0.8, 0.9])

class Armar6HitBallExpV0:
    def __init__(self, high_ctrl, low_ctrl, env_path,
                 low_ctrl_config=None, arm_name="RightArm", desired_joints=None, isdraw=False):
        self.world = mujoco_py.load_model_from_path(env_path)
        self.sim = mujoco_py.MjSim(self.world)

        high_ctrl.motion_duration = 1.8
        self.high_ctrl = high_ctrl
        self.high_ctrl.low_ctrl = low_ctrl(model=self.world, sim=self.sim, config=low_ctrl_config, \
                                           arm_name=arm_name, desired_joints=desired_joints)

        self.isdraw = isdraw
        if self.isdraw:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer._paused = True

        self.joint_ctrl = JointController(model=self.world, sim=self.sim, arm_name=arm_name, ctrl_mode="Torque")
        self.init_joints = np.array([0, -0.2, 0, 0, 1.8, 3.14, 0, 0])
        self.is_ball_pos_change = False

    def run(self, stop_sim_after=True):
        start_time = self.sim.data.time
        done = False
        ts_traj = []
        js_traj = []
        is_error = False
        final_ball_pos = self.sim.data.get_body_xpos("Ball")
        has_moved = False
        while not stop_sim_after or not done:
            sim_duration = self.sim.data.time - start_time
            motion_done = self.high_ctrl.control(sim_duration)
            ball_pos = self.sim.data.get_body_xpos("Ball")
            ball_vel = self.sim.data.get_body_xvelp("Ball")

            if np.linalg.norm(ball_vel[:2]) > 0.01:
                has_moved = True

            if has_moved and np.linalg.norm(ball_vel[:2]) < 0.01:
                final_ball_pos = ball_pos
                done = True

            if not motion_done:
                tcp_pos = self.sim.data.get_site_xpos(self.high_ctrl.low_ctrl.tcp_name)
                tcp_xmat = self.sim.data.get_site_xmat(self.high_ctrl.low_ctrl.tcp_name)
                tcp_quat = np.zeros(4)
                mujoco_py.functions.mju_mat2Quat(tcp_quat, np.reshape(tcp_xmat, (-1,), order='C'))
                t_pose = np.concatenate([[sim_duration], tcp_pos.copy(), tcp_quat.copy()], axis=0)
                ts_traj.append(t_pose[:3])

                qpos = get_actuator_data(self.sim.data.qpos,  self.high_ctrl.low_ctrl.actuator_id_in_order)
                t_qpos = np.concatenate([[sim_duration], qpos.copy()], axis=0)
                js_traj.append(t_qpos)

            try:
                self.sim.step()
            except:
                print('mujoco step error')

            if ball_pos[2] < 0.8 or (sim_duration > 5 and not has_moved):
                is_error = True
                break

            if self.isdraw:
                self.viewer.render()

        ts_traj = np.stack(ts_traj)
        js_traj = np.stack(js_traj)
        return final_ball_pos.copy(), ts_traj, js_traj, is_error

    def reset(self, init_ball_pos=None, target_pos=None):
        states = self.sim.get_state()
        joint_name_list = self.high_ctrl.low_ctrl.joint_name_list
        for joint_id, joint in enumerate(joint_name_list):
            addr = self.sim.model.get_joint_qpos_addr(joint)
            states.qpos[addr] = self.init_joints[joint_id]

        for joint_id, joint in enumerate(RIGHT_HAND_JOINT):
            addr = self.sim.model.get_joint_qpos_addr(joint)
            states.qpos[addr] = RIGHT_HAND_JOINT_CONFIG[joint][1]

        height = 0.98
        if target_pos is not None:
            addr = self.sim.model.get_joint_qpos_addr("ref_box_x")
            states.qpos[addr] = target_pos[0]
            addr = self.sim.model.get_joint_qpos_addr("ref_box_y")
            states.qpos[addr] = target_pos[1]

        if init_ball_pos is None:
            addr = self.sim.model.get_joint_qpos_addr("box_x")
            rx = 0.2 + np.random.rand(1) * 0.5
            ry = 0.8 + 0.1 * np.random.rand(1)
            states.qpos[addr] = rx
            addr = self.sim.model.get_joint_qpos_addr("box_y")
            states.qpos[addr] = ry
            addr = self.sim.model.get_joint_qpos_addr("box_z")
            states.qpos[addr] = height

            addr = self.sim.model.get_joint_qvel_addr("box_x")
            states.qvel[addr] = 0
            addr = self.sim.model.get_joint_qvel_addr("box_y")
            states.qvel[addr] = 0
            addr = self.sim.model.get_joint_qvel_addr("box_z")
            states.qvel[addr] = 0
        else:
            addr = self.sim.model.get_joint_qpos_addr("box_x")
            states.qpos[addr] = init_ball_pos[0]
            addr = self.sim.model.get_joint_qpos_addr("box_y")
            states.qpos[addr] = init_ball_pos[1]
            addr = self.sim.model.get_joint_qpos_addr("box_z")
            states.qpos[addr] = height

            addr = self.sim.model.get_joint_qvel_addr("box_x")
            states.qvel[addr] = 0
            addr = self.sim.model.get_joint_qvel_addr("box_y")
            states.qvel[addr] = 0
            addr = self.sim.model.get_joint_qvel_addr("box_z")
            states.qvel[addr] = 0

        self.sim.set_state(states)
        self.sim.forward()
        if self.isdraw:
            self.viewer.render()

        tcp_xpos = self.sim.data.get_site_xpos(self.high_ctrl.low_ctrl.tcp_name)
        tcp_xmat = self.sim.data.get_site_xmat(self.high_ctrl.low_ctrl.tcp_name)
        target_quat = np.zeros(4)
        mujoco_py.functions.mju_mat2Quat(target_quat, np.reshape(tcp_xmat, (-1,), order='C'))
        start = np.concatenate([tcp_xpos, target_quat], axis=0)
        ball_pos = self.sim.data.get_body_xpos("Ball")
        return start, ball_pos.copy()


class Armar6HitBallExpV1:
    def __init__(self, high_ctrl, low_ctrl, env_path,
                 low_ctrl_config=None, arm_name="RightArm", desired_joints=None, isdraw=False):
        self.world = mujoco_py.load_model_from_path(env_path)
        self.sim = mujoco_py.MjSim(self.world)

        high_ctrl.motion_duration = 1.8
        self.high_ctrl = high_ctrl
        self.high_ctrl.low_ctrl = low_ctrl(model=self.world, sim=self.sim, config=low_ctrl_config, \
                                           arm_name=arm_name, desired_joints=desired_joints)

        self.isdraw = isdraw
        if self.isdraw:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer._paused = True

        self.init_joints = np.array([0, -0.2, 0, 0, 1.8, 3.14, 0, 0])
        self.is_ball_pos_change = False

    def run(self, stop_sim_after=True):
        start_time = self.sim.data.time
        done = False
        ts_traj = []
        js_traj = []
        is_error = False
        final_ball_pos = self.sim.data.get_body_xpos("Ball")
        has_moved = False
        while not stop_sim_after or not done:
            sim_duration = self.sim.data.time - start_time
            motion_done = self.high_ctrl.control(sim_duration)
            ball_pos = self.sim.data.get_body_xpos("Ball")
            ball_vel = self.sim.data.get_body_xvelp("Ball")

            if np.linalg.norm(ball_vel[:2]) > 0.01:
                has_moved = True

            if has_moved and np.linalg.norm(ball_vel[:2]) < 0.01:
                final_ball_pos = ball_pos
                done = True

            if not motion_done:
                tcp_pos = self.sim.data.get_site_xpos(self.high_ctrl.low_ctrl.tcp_name)
                tcp_xmat = self.sim.data.get_site_xmat(self.high_ctrl.low_ctrl.tcp_name)
                tcp_quat = np.zeros(4)
                mujoco_py.functions.mju_mat2Quat(tcp_quat, np.reshape(tcp_xmat, (-1,), order='C'))
                t_pose = np.concatenate([[sim_duration], tcp_pos.copy(), tcp_quat.copy()], axis=0)
                ts_traj.append(t_pose[:3])

                qpos = get_actuator_data(self.sim.data.qpos,  self.high_ctrl.low_ctrl.actuator_id_in_order)
                t_qpos = np.concatenate([[sim_duration], qpos.copy()], axis=0)
                js_traj.append(t_qpos)

            try:
                self.sim.step()
            except:
                print('mujoco step error')

            if ball_pos[2] < 0.8 or (sim_duration > 5 and not has_moved):
                is_error = True
                break

            if self.isdraw:
                self.viewer.render()

        ts_traj = np.stack(ts_traj)
        js_traj = np.stack(js_traj)
        return final_ball_pos.copy(), ts_traj, js_traj, is_error

    def reset(self, init_ball_pos=None, target_pos=None):
        states = self.sim.get_state()
        joint_name_list = self.high_ctrl.low_ctrl.joint_name_list
        for joint_id, joint in enumerate(joint_name_list):
            addr = self.sim.model.get_joint_qpos_addr(joint)
            states.qpos[addr] = self.init_joints[joint_id]

        height = 0.98
        if target_pos is not None:
            addr = self.sim.model.get_joint_qpos_addr("ref_box_x")
            states.qpos[addr] = target_pos[0]
            addr = self.sim.model.get_joint_qpos_addr("ref_box_y")
            states.qpos[addr] = target_pos[1]

        if init_ball_pos is None:
            addr = self.sim.model.get_joint_qpos_addr("box_x")
            rx = 0.2 + np.random.rand(1) * 0.5
            ry = 0.8 + 0.1 * np.random.rand(1)
            states.qpos[addr] = rx
            addr = self.sim.model.get_joint_qpos_addr("box_y")
            states.qpos[addr] = ry
            addr = self.sim.model.get_joint_qpos_addr("box_z")
            states.qpos[addr] = height

            addr = self.sim.model.get_joint_qvel_addr("box_x")
            states.qvel[addr] = 0
            addr = self.sim.model.get_joint_qvel_addr("box_y")
            states.qvel[addr] = 0
            addr = self.sim.model.get_joint_qvel_addr("box_z")
            states.qvel[addr] = 0
        else:
            addr = self.sim.model.get_joint_qpos_addr("box_x")
            states.qpos[addr] = init_ball_pos[0]
            addr = self.sim.model.get_joint_qpos_addr("box_y")
            states.qpos[addr] = init_ball_pos[1]
            addr = self.sim.model.get_joint_qpos_addr("box_z")
            states.qpos[addr] = height

            addr = self.sim.model.get_joint_qvel_addr("box_x")
            states.qvel[addr] = 0
            addr = self.sim.model.get_joint_qvel_addr("box_y")
            states.qvel[addr] = 0
            addr = self.sim.model.get_joint_qvel_addr("box_z")
            states.qvel[addr] = 0

        self.sim.set_state(states)
        self.sim.forward()
        if self.isdraw:
            self.viewer.render()

        tcp_xpos = self.sim.data.get_site_xpos(self.high_ctrl.low_ctrl.tcp_name)
        tcp_xmat = self.sim.data.get_site_xmat(self.high_ctrl.low_ctrl.tcp_name)
        target_quat = np.zeros(4)
        mujoco_py.functions.mju_mat2Quat(target_quat, np.reshape(tcp_xmat, (-1,), order='C'))
        start = np.concatenate([tcp_xpos, target_quat], axis=0)
        ball_pos = self.sim.data.get_body_xpos("Ball")
        return start, ball_pos.copy()


def run_env(data, q):
    final_ball_pos, _, _, is_error = data['env'].run()

    if not is_error and np.linalg.norm(data['query']- final_ball_pos[:2]) < 0.18:
        q.put(1)
    else:
        q.put(0)

def evaluate_hitball_multiprocess(wout, queries, starts, goals, low_ctrl, high_ctrl, env_path, EXP=Armar6HitBallExpV1):
    import multiprocessing
    processes = []
    q = multiprocessing.Queue()
    for i in range(np.shape(wout)[0]):
        for sampleId in range(np.shape(wout)[1]):
            env = EXP(high_ctrl=high_ctrl, low_ctrl=low_ctrl, isdraw=False, env_path=os.environ['MPGEN_DIR'] + env_path)
            st, _ = env.reset(init_ball_pos=goals[i, :], target_pos=queries[i, :])
            env.high_ctrl.target_quat = st[3:]
            env.high_ctrl.target_posi = st[:3]
            env.high_ctrl.desired_joints = np.array([0, -0.2, 0, 0, 1.8, 3.14, 0, 0])
            env.high_ctrl.vmp.set_weights(wout[i, sampleId, :])
            env.high_ctrl.vmp.set_start_goal(starts[i, :], goals[i, :])
            data = {'env': env, 'query': queries[i,:]}
            p = multiprocessing.Process(target=run_env, args=(data,q,))
            processes.append(p)
            p.start()

    for process in processes:
        process.join()

    srate = 0.0
    for i in range(np.shape(wout)[0]):
        for sampleId in range(np.shape(wout)[1]):
            srate = srate + q.get()

    print('srate: {}'.format(srate/np.shape(wout)[0]))
    return srate/np.shape(wout)[0]


def evaluate_hitball(wout, queries, starts, goals, low_ctrl, high_ctrl, env_path, EXP=Armar6HitBallExpV1, isdraw=False):
    # wout: N x S x dim, N: number of experiments, S: number of samples, dim: dimension of MP
    env = EXP(high_ctrl=high_ctrl, low_ctrl=low_ctrl, isdraw=isdraw, env_path=os.environ['MPGEN_DIR']+env_path)

    srate = 0.0
    for i in range(np.shape(wout)[0]):
        success_counter = 0
        for sampleId in range(np.shape(wout)[1]):
            st, _ = env.reset(init_ball_pos=goals[i,:], target_pos=queries[i,:])
            env.high_ctrl.target_quat = st[3:]
            env.high_ctrl.target_posi = st[:3]
            env.high_ctrl.desired_joints = np.array([0, -0.2, 0, 0, 1.8, 3.14, 0, 0])
            env.high_ctrl.vmp.set_weights(wout[i,sampleId,:])
            env.high_ctrl.vmp.set_start_goal(starts[i,:], goals[i,:])
            final_ball_pos, _, _, is_error = env.run()

            if not is_error and np.linalg.norm(queries[i,:] - final_ball_pos[:2]) < 0.18:
                success_counter = success_counter + 1
                break

        if success_counter != 0:
            srate = srate + 1

        print('testId: %1d,  success_num: %1d, success_rate: %.2f' % (i, srate, srate / (i + 1)), end='\r', flush=True)

    print('sample_num: %1d, success_num: %1d, success_rate: %.2f' % (np.shape(wout)[1], srate, srate / (i + 1)), end='\n')
    return srate/np.shape(wout)[0]
