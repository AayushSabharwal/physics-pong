import mujoco_pong
from stable_baselines3 import A2C, PPO
import sys

assert len(sys.argv) == 2
assert sys.argv[1] in ["ppo", "a2c"]

algo = None
if sys.argv[1] == "ppo":
    algo = PPO
elif sys.argv[1] == "a2c":
    algo = A2C

env = mujoco_pong.env(1)
model = algo.load(f"models/pong_simple_{sys.argv[1]}.zip")

obs = env.reset()
video = []
for i in range(900):
    action, _states = model.predict(obs)
    # print(action)
    obs, rewards, dones, info = env.step(action)
    
    if dones.any():
        obs = env.reset()
    video.append(env.render())

anim = mujoco_pong.display_video(video)
anim.save("pong.mp4")
