import dm_control.mjcf as mjcf
import dm_control.mujoco as mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import gym
from gym import spaces
from pettingzoo.utils import aec_to_parallel
from pettingzoo.utils import wrappers
from pettingzoo import ParallelEnv
from copy import deepcopy
import functools
import supersuit as ss


class Pong:
    reduction_factor = 20
    stage_width = 512 / reduction_factor
    stage_height = 256 / reduction_factor
    wall_thickness = 5 / reduction_factor
    paddle_width = 2 / reduction_factor
    paddle_height = 28 / reduction_factor
    ball_diameter = 5 / reduction_factor
    ball_speed = 7 * 30 * 1.414 / reduction_factor
    paddle_speed = 4 * 30 * 1.414 / reduction_factor
    paddle_max_acc = 30
    paddle_mass = 1
    ball_mass = 0.01
    target_vel_bias = 0.8

    def __init__(self, name):
        self.model = mjcf.RootElement(model=name)
        self.model.option.set_attributes(gravity="0 0 0")
        self.model.option.flag.set_attributes(frictionloss="disable")

        # region ball
        self.ball = self.model.worldbody.add("body", name="ball")
        self.ball.add(
            "geom",
            name="ball_col",
            type="sphere",
            size=f"{Pong.ball_diameter / 2}",
            mass=f"{Pong.ball_mass}",
            rgba="1 0 0 1",
            solref="-10000 0",
            condim="1",
        )
        self.ball.add(
            "joint",
            name="ball_jx",
            type="slide",
            axis="1 0 0",
        )
        ball_vertical_leeway = Pong.stage_height / 2 - Pong.ball_diameter / 2
        self.ball.add(
            "joint",
            name="ball_jz",
            type="slide",
            axis="0 0 1",
            limited="true",
            range=f"{-ball_vertical_leeway} {ball_vertical_leeway}",
        )
        self.model.actuator.add(
            "velocity", name="ball_ax", joint="ball_jx", kv=f"{Pong.ball_mass}"
        )
        self.model.actuator.add(
            "velocity", name="ball_az", joint="ball_jz", kv=f"{Pong.ball_mass}"
        )
        # endregion

        self.top_wall = self.model.worldbody.add(
            "geom",
            name="top_wall",
            type="box",
            pos=f"0 0 {Pong.stage_height / 2 + Pong.wall_thickness / 2}",
            size=f"{Pong.stage_width / 2+ 1} 1 {Pong.wall_thickness / 2}",
            rgba="0 0 1 1",
            condim="1",
        )
        self.bottom_wall = self.model.worldbody.add(
            "geom",
            name="bottom_wall",
            type="box",
            pos=f"0 0 {-Pong.stage_height / 2 - Pong.wall_thickness / 2}",
            size=f"{Pong.stage_width / 2 + 1} 1 {Pong.wall_thickness / 2}",
            rgba="0 0 1 1",
            condim="1",
        )

        paddle_deltax = (
            Pong.stage_width / 2 - Pong.paddle_width / 2 - Pong.wall_thickness / 2
        )
        paddle_range = Pong.stage_width / 2 - Pong.paddle_width
        paddle_deltay = Pong.stage_height - Pong.paddle_height
        # region lpad
        self.left_paddle = self.model.worldbody.add("body", name="lpad")
        self.left_paddle.add(
            "geom",
            name="lpad_col",
            type="box",
            pos=f"{-paddle_deltax} 0 0",
            size=f"{Pong.paddle_width} 1 {Pong.paddle_height}",
            mass=f"{Pong.paddle_mass}",
            condim="1",
        )
        self.left_paddle.add(
            "joint",
            name="lpad_jx",
            type="slide",
            axis="1 0 0",
            limited="true",
            range=f"0 {paddle_range}",
        )

        self.left_paddle.add(
            "joint",
            name="lpad_jz",
            type="slide",
            axis="0 0 1",
            limited="true",
            range=f"{-paddle_deltay / 2} {paddle_deltay / 2}",
        )
        self.model.actuator.add(
            "general", name="lpad_ax", joint="lpad_jx", gainprm=f"{Pong.paddle_max_acc}"
        )
        self.model.actuator.add(
            "general", name="lpad_az", joint="lpad_jz", gainprm=f"{Pong.paddle_max_acc}"
        )
        # endregion

        # region rpad
        self.right_paddle = self.model.worldbody.add("body", name="rpad")
        self.right_paddle.add(
            "geom",
            name="rpad_col",
            type="box",
            pos=f"{paddle_deltax} 0 0",
            size=f"{Pong.paddle_width} 1 {Pong.paddle_height}",
            mass=f"{Pong.paddle_mass}",
            condim="1",
        )
        self.right_paddle.add(
            "joint",
            name="rpad_jx",
            type="slide",
            axis="-1 0 0",
            limited="true",
            range=f"0 {paddle_range}",
        )

        self.right_paddle.add(
            "joint",
            name="rpad_jz",
            type="slide",
            axis="0 0 1",
            limited="true",
            range=f"{-paddle_deltay / 2} {paddle_deltay / 2}",
        )
        self.model.actuator.add(
            "general",
            name="rpad_ax",
            joint="rpad_jx",
            gainprm=f"-{Pong.paddle_max_acc}",
        )
        self.model.actuator.add(
            "general",
            name="rpad_az",
            joint="rpad_jz",
            gainprm=f"-{Pong.paddle_max_acc}",
        )
        # endregion

        self.light = self.model.worldbody.add(
            "light",
            pos="0 -2 0",
            directional="true",
            dir="1 1 -1",
            ambient="0.3 0.3 0.3",
        )
        self.cam_target = self.model.worldbody.add("body", name="cam_target")
        self.cam_target.add("site", name="cam_target_site", rgba="0.5 0.5 0.5 0.5")
        self.camera = self.model.worldbody.add(
            "camera",
            name="maincam",
            mode="targetbody",
            pos="0 -30 0",
            target="cam_target",
        )


def env(envs):
    env = PongEnv()
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, envs, num_cpus=8, base_class='stable_baselines3')
    return env


class PongEnv(ParallelEnv):
    metadata = {"render.mode": ["human"], "name": "pong", "is_parallelizable": True}

    def __init__(self, framerate=30):
        self.physics = None
        self.framerate = framerate
        self.possible_agents = ["l", "r"]
        self.action_spaces = {
            "l": self.action_space("l"),
            "r": self.action_space("r"),
        }
        self.observation_spaces = {
            "l": self.observation_space("l"),
            "r": self.observation_space("r"),
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(
            np.array(
                [  # self pos
                    -Pong.stage_width / 2,
                    -Pong.stage_height / 2 + Pong.paddle_height / 2,
                    # self vel
                    -18,
                    -13,
                    # ball pos
                    -Pong.stage_width / 2,
                    -Pong.stage_height / 2,
                    # ball vel
                    -Pong.ball_speed,
                    -Pong.ball_speed,
                    # other pos
                    0,
                    -Pong.stage_height / 2 + Pong.paddle_height / 2,
                    # other vel
                    -18,
                    -13,
                ]
            ),
            np.array(
                [
                    # self pos
                    0,
                    Pong.stage_height / 2 - Pong.paddle_height / 2,
                    # self vel
                    18,
                    13,
                    # ball pos
                    Pong.stage_width / 2,
                    Pong.stage_height / 2,
                    # ball vel
                    Pong.ball_speed,
                    Pong.ball_speed,
                    # other pos
                    Pong.stage_width / 2,
                    Pong.stage_height / 2 - Pong.paddle_height / 2,
                    # other vel
                    18,
                    13,
                ]
            ),
            dtype=np.float64,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(np.ones(2) * -1, np.ones(2), dtype=np.float64)

    def render(self, mode="human"):
        pix = self.physics.render(camera_id="maincam")
        return pix.copy()

    def observe(self, agent):
        if agent == "l":
            return np.concatenate((
                self.physics.named.data.geom_xpos["lpad_col"][[0, 2]],
                self.physics.named.data.qvel[["lpad_jx", "lpad_jz"]],
                self.physics.named.data.geom_xpos["ball_col"][[0, 2]],
                self.physics.named.data.qvel[["ball_jx", "ball_jz"]],
                self.physics.named.data.geom_xpos["rpad_col"][[0, 2]],
                self.physics.named.data.qvel[["rpad_jx", "rpad_jz"]],
            ))
        else:
            return np.concatenate((
                self.physics.named.data.geom_xpos["rpad_col"][[0, 2]],
                self.physics.named.data.qvel[["rpad_jx", "rpad_jz"]],
                self.physics.named.data.geom_xpos["ball_col"][[0, 2]],
                self.physics.named.data.qvel[["ball_jx", "ball_jz"]],
                self.physics.named.data.geom_xpos["lpad_col"][[0, 2]],
                self.physics.named.data.qvel[["lpad_jx", "lpad_jz"]],
            )) * np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed=seed)
        self.agents = self.possible_agents[:]
        self.dones = {"l": False, "r": False}
        self.infos = {"l": {}, "r": {}}
        self.rewards = {"l": 0, "r": 0}
        pongmodel = Pong("pong")
        self.physics = get_physics(pongmodel)

        theta = (
            np.pi / 4
            + self.rng.uniform(-np.pi / 5, np.pi / 5)
            + self.rng.choice(4) * np.pi / 2
        )
        ball_vel = np.array([np.cos(theta), np.sin(theta)]) * Pong.ball_speed
        set_ball_vel(self.physics, ball_vel)

        return {a: self.observe(a) for a in self.agents}

    def step(self, actions):
        ctime = self.physics.data.time
        while self.physics.data.time < ctime + 1 / self.framerate:
            step_simulation(self.physics, actions["l"], actions["r"])
        obs = {"l": self.observe("l"), "r": self.observe("r")}

        self.rewards = {"l": 1, "r": 1}
        ball_x = self.physics.named.data.geom_xpos["ball_col"][0]
        if ball_x <= -Pong.stage_width / 2:
            self.rewards["l"] = 0
            self.rewards["r"] = 0

        elif ball_x >= Pong.stage_width / 2:
            self.rewards["l"] = 0
            self.rewards["r"] = 0

        self.dones = {a: r != 1 for a, r in self.rewards.items()}

        return obs, self.rewards, self.dones, self.infos


def get_physics(pongmodel):
    phys = mjcf.Physics.from_mjcf_model(pongmodel.model)
    with phys.reset_context():
        phys.data.qvel[0:2] = Pong.ball_speed
    return phys


def set_ball_vel(phys, vel):
    with phys.reset_context():
        phys.data.qvel[0:2] = vel


def get_sceneopt():
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.frame = enums.mjtFrame.mjFRAME_GEOM
    scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True


def step_simulation(physics, ctrl_l, ctrl_r):
    physics.data.ctrl[2:4] = ctrl_l
    physics.data.ctrl[4:6] = ctrl_r
    ball_vel = physics.data.actuator_velocity[0:2]
    ball_speed = np.linalg.norm(ball_vel)
    target_vel = ball_vel * (
        Pong.ball_speed / ball_speed
        # * Pong.target_vel_bias + (1 - Pong.target_vel_bias)
    )
    physics.data.ctrl[0:2] = target_vel
    physics.step()


def get_video(physics: mjcf.physics.Physics, act, duration=10, framerate=30, **kwargs):
    video = []
    while physics.data.time < duration:
        step_simulation(physics, 0, 0)
        if len(video) < physics.data.time * framerate:
            pix = physics.render(**kwargs)
            video.append(pix.copy())
    return video


def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False
    )
    return anim
