import mujoco_pong
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback
import os
import sys

assert len(sys.argv) == 2
assert sys.argv[1] in ["ppo", "a2c"]

algo = None
if sys.argv[1] == "ppo":
    algo = PPO
elif sys.argv[1] == "a2c":
    algo = A2C

env = mujoco_pong.env(32)
if os.path.isfile(f"models/pong_simple_{sys.argv[1]}"):
    model = algo.load(f"models/pong_simple_{sys.argv[1]}.zip")
else:
    model = algo("MlpPolicy", env, verbose=1)

eval_env = mujoco_pong.env(32)
callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=1000,
    deterministic=True,
    render=False,
)

model.learn(total_timesteps=1000000, callback=callback)
model.save(f"models/pong_simple_{sys.argv[1]}")
print("done")
exit()
# obs = env.reset()
# video = []
# for i in range(300):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     video.append(env.render())

# anim = mujoco_pong.display_video(video)
# anim.save("pong.mp4")
