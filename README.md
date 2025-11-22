# ARLForge  
Adaptive Reinforcement Learning Library for Model-Based Autonomous Control

ARLForge is a research-grade reinforcement learning framework that combines real environment experience with learned dynamics models to improve sample efficiency, stability, and adaptability in complex continuous-control tasks like rockets, drones, and robotics.

This project is designed as a modular library — clean, extendable, and suitable for research or production.


## Architecture

ARLForge uses a hybrid reinforcement learning loop where a learned dynamics model generates additional synthetic experience to improve SAC training. The system works as a closed learning loop:



 Environment
     │ real transitions
     ▼
 Replay Buffer (real)
     │ sample batches
     ▼
 SAC Agent  ←──────────────────────────┐
     │ policy update                   │ synthetic transitions
     ▼                                 │
 Model Rollouts  →  Replay Buffer (synthetic)
     ▲
     │ predicted transitions
     ▼
 Dynamics Model (learned environment)


## Key Features

- Adaptive model-based + model-free reinforcement learning  
- Learned internal dynamics model for predictive rollouts  
- Synthetic experience generation for higher sample efficiency  
- Modular architecture: core agents, dynamics, utils, envs  
- Replay buffers for both real and model-generated transitions  
- Custom environments (e.g., RocketEnv for continuous control)  
- SAC backbone with actor + twin critics  
- Clean, extendable code designed for research and real-world control  



## Installation

Clone the repository:

```bash
git clone https://github.com/Vardhinedi/arlforge.git
cd arlforge


## Usage

### Train on the custom Rocket environment

```bash
python train_rocket.py


## Project Structure


arlforge/
 ├── arlforge/
 │    ├── core/
 │    │     ├── agents/
 │    │     │      └── sac.py
 │    │     ├── dynamics_model.py
 │    │     └── ...
 │    ├── envs/
 │    │     └── rocket_env.py
 │    ├── utils/
 │    │     ├── replay_buffer.py
 │    │     ├── networks.py
 │    │     └── ...
 │    └── adaptive_agent.py
 ├── train_cartpole.py
 ├── train_rocket.py
 └── README.md


## Research Background

ARLForge is inspired by modern research in model-based reinforcement learning.  
It merges ideas from several influential algorithms:

- **Model-Based Policy Optimization (MBPO)**  
- **Dreamer (latent dynamics learning)**  
- **PETS (probabilistic ensembles of trajectories)**  
- **Soft Actor-Critic (SAC)**  
- **Hybrid offline + online learning methods**

By combining real transitions with synthetic transitions produced by the learned dynamics model, ARLForge significantly improves:

- sample efficiency  
- stability  
- generalization  
- robustness in unstable physics environments  

This makes it suitable for autonomous rockets, drones, robotic arms, and simulation-heavy research.


## Roadmap

### Short Term
- Improve dynamics model accuracy  
- Add model regularization and uncertainty estimation  
- Better rollout scheduling (adaptive synthetic generation)  

### Medium Term
- Domain randomization support  
- Multi-agent adaptive RL  
- Hardware-in-the-loop compatibility  

### Long Term
- Advanced aerospace simulations  
- Autonomous flight control stack  
- Industry-grade ARL training pipeline  


## License

This project is released under the MIT License.  
You are free to use it for academic, personal, and commercial work.

## Citation

If you use ARLForge in research:

@software{arlforge2025,
author = {Vardhinedi},
title = {ARLForge: Adaptive Reinforcement Learning Library},
year = {2025},
url = {https://github.com/Vardhinedi/arlforge}

}