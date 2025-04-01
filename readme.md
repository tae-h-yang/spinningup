**Status:** Maintenance (expect bug fixes and minor updates)

Welcome to Spinning Up in Deep RL! 
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!

---

## ✅ Apple Silicon (M1/M2) Compatibility Setup

This fork has been updated and tested to run successfully on **MacBooks with Apple Silicon (M1/M2 chips)**, including PPO training, MuJoCo environments, policy evaluation, and plotting.

### 🔧 Installation Instructions (Tested on macOS, with MuJoCo)

```bash
# 1. Create environment
conda create -n spinup python=3.8
conda activate spinup

# 2. Install core dependencies for Apple Silicon
pip install tensorflow-macos
pip install tensorflow-metal

# 3. Install gymnasium and environments
pip install gymnasium
pip install "gymnasium[mujoco]"
pip install "gymnasium[box2d]"

# 4. Clone and prepare repo
git clone https://github.com/tae-h-yang/spinningup.git
cd spinningup
git checkout mujoco
brew install open-mpi

# 5. Install the repo
pip install -e .
```

> `setup.py` has been updated to support Apple Silicon by using:
> - `tensorflow-macos` & `tensorflow-metal`
> - `gymnasium` (instead of `gym`)
> - `mujoco` (instead of `mujoco-py`)
> - Compatible versions of `torch`, `seaborn`, and other libraries

---

### ✅ Check Your Install (MuJoCo + Box2D)

```bash
# Test Box2D (LunarLander-v3)
python -m spinup.run ppo --hid "[32,32]" --env LunarLander-v3 --exp_name installtest --gamma 0.999
python -m spinup.run test_policy data/installtest/installtest_s0
python -m spinup.run plot data/installtest/installtest_s0

# Test MuJoCo (Walker2d-v5)
python -m spinup.run ppo --hid "[32,32]" --env Walker2d-v5 --exp_name mujocotest
python -m spinup.run test_policy data/mujocotest/mujocotest_s0
python -m spinup.run plot data/mujocotest/mujocotest_s0
```

✅ All tests should run successfully on M1/M2 MacBooks.

---

Citing Spinning Up
------------------

If you reference or use Spinning Up in your research, please cite:

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```