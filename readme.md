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

This fork has been updated and tested to run successfully on **MacBooks with Apple Silicon (M1/M2 chips)**, including PPO training, policy evaluation, and plotting.

For optional MuJoCo installation, refer to the MuJoCo branch’s installation instructions.

### 🔧 Installation Instructions (Tested on macOS)

```bash
# 1. Create environment
conda create -n spinup python=3.8
conda activate spinup

# 2. Install updated dependencies
pip install tensorflow-macos
pip install tensorflow-metal
pip install box2d-py
pip install "gym[classic_control]"

# 3. Clone and prepare repo
git clone https://github.com/tae-h-yang/spinningup.git
cd spinningup
brew install open-mpi

# 4. Install the repo
pip install -e .
```

> `setup.py` was updated to use modern, ARM-compatible versions of:
> - `tensorflow` (via `tensorflow-macos`)
> - `torch`
> - `gym` (>= 0.26)
> - `seaborn` (>= 0.11)

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