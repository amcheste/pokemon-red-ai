# 🎮 Pokemon Red AI

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyBoy](https://img.shields.io/badge/PyBoy-2.0+-green.svg)](https://github.com/Baekalfen/PyBoy)

A comprehensive reinforcement learning toolkit for training AI agents to play Pokemon Red using PyBoy emulation and Stable-Baselines3.

**🚀 Now with improved exploration-focused training for better results!**

## 🌟 Features

- **🤖 Complete RL Environment**: Full Gymnasium-compatible environment for Pokemon Red
- **🎯 Improved Exploration Training**: Enhanced reward system for better map discovery and exploration
- **📊 Live Training Visualization**: Real-time plots and metrics during training
- **🔧 Modular Architecture**: Clean, maintainable codebase with separated concerns
- **💾 Automatic State Management**: Handles game initialization, resets, and save states
- **🎨 Rich CLI**: Beautiful command-line interface powered by Click and Rich
- **📈 TensorBoard Integration**: Detailed training logs and metrics
- **🧪 Testing & Evaluation**: Comprehensive model testing and performance analysis
- **⚡ Anti-Stuck Mechanisms**: Prevents agents from getting trapped in small areas
- **🎮 Extended Episodes**: 3x longer episodes (15,000 steps) for better exploration

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Training Strategies](#training-strategies)
- [Advanced Usage](#advanced-usage)
- [Performance Tips](#performance-tips)
- [What's Improved](#whats-improved)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Pokemon Red ROM file (`.gb` format)
- 4GB+ RAM recommended
- CUDA-capable GPU (optional, for faster training)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pokemon-red-ai.git
cd pokemon-red-ai

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Install with editable mode and development dependencies
pip install -e .
pip install -r requirements-dev.txt  # If available
```

### Verify Installation

```bash
# Check system requirements
pokemon-ai doctor

# Find ROM files in current directory
pokemon-ai find-roms .
```

## 🎯 Quick Start

### Training Your First Agent (Improved Settings)

```bash
# Basic training with improved exploration-focused defaults
pokemon-ai train --rom PokemonRed.gb --monitor-mode

# Training with custom timesteps but keeping improvements
pokemon-ai train --rom PokemonRed.gb --timesteps 200000 --monitor-mode

# Fast training without monitoring (headless)
pokemon-ai train --rom PokemonRed.gb --timesteps 100000
```

### Testing a Trained Model

```bash
# Test with improved episode length
pokemon-ai test --rom PokemonRed.gb --model models/best_model.zip --episodes 10
```

### Using the Python API

```python
from pokemon_red_ai import PokemonTrainer

# Create trainer with improved defaults (exploration-focused)
trainer = PokemonTrainer(
    rom_path="PokemonRed.gb",
    save_dir="./training_output/"
)

# Train game with improved settings
trainer.train(
    total_timesteps=500000,  # 5x more training
    show_plots=True
)

# Test trained model
results = trainer.test(
    model_path="./training_output/models/best_model.zip",
    episodes=10
)

print(f"Average reward: {results['avg_reward']:.2f}")
print(f"Max maps discovered: {results['max_maps_visited']}")
print(f"Exploration efficiency: {results['avg_exploration_efficiency']:.4f}")
```

## 📖 Usage

### Command Line Interface

The package provides a comprehensive CLI with multiple commands:

#### Initialize a New Project

```bash
# Create project structure
pokemon-ai init --save-dir ./my_pokemon_project
```

#### Training Commands

```bash
# 🚀 IMPROVED: Training with new defaults (recommended)
pokemon-ai train --rom PokemonRed.gb --monitor-mode

# Advanced training with all monitoring
pokemon-ai train \
  --rom PokemonRed.gb \
  --timesteps 1000000 \
  --reward-strategy exploration \
  --max-episode-steps 20000 \
  --learning-rate 0.0001 \
  --batch-size 32 \
  --monitor-mode

# Quick training without monitoring (fastest)
pokemon-ai train --rom PokemonRed.gb --timesteps 100000
```

**Key Improvements in Default Settings:**
- 📈 **500k timesteps** (5x increase from 100k)
- 🎯 **Exploration rewards** (5x higher than standard)
- ⏱️ **15k step episodes** (3x longer for better exploration)
- 🧠 **Optimized learning** (lower learning rate, smaller batches)
- 🔧 **Anti-stuck mechanisms** (prevents getting trapped)

#### Testing and Evaluation

```bash
# Test model with improved episode length
pokemon-ai test \
  --rom PokemonRed.gb \
  --model models/best_model.zip \
  --episodes 10 \
  --render

# Save detailed test results
pokemon-ai test \
  --rom PokemonRed.gb \
  --model models/best_model.zip \
  --save-results results.json \
  --max-episode-steps 20000
```

#### Project Management

```bash
# View project information
pokemon-ai info ./training_output/

# Create/validate configuration
pokemon-ai config ./config.yaml

# View configuration template
pokemon-ai config --template

# System health check
pokemon-ai doctor
```

### Python API

#### Basic Training

```python
from pokemon_red_ai import create_trainer

# Quick training with improved defaults
trainer = create_trainer("PokemonRed.gb")
trainer.train(timesteps=500000)  # Uses exploration-focused rewards
```

#### Custom Environment

```python
from pokemon_red_ai import PokemonRedGymEnv, RewardConfig

# Create enhanced reward configuration
reward_config = RewardConfig(
    exploration_reward=5.0,        # 5x higher exploration rewards
    new_map_reward=100.0,         # Major bonuses for new areas
    time_penalty=-0.001,          # Reduced time pressure
    level_reward_multiplier=25.0   # Balanced progress rewards
)

# Create environment with improved settings
env = PokemonRedGymEnv(
    rom_path="PokemonRed.gb",
    reward_strategy="exploration",  # Exploration-focused by default
    reward_config=reward_config,
    max_episode_steps=15000        # 3x longer episodes
)

# Use with any RL library
from stable_baselines3 import PPO

model = PPO("MultiInputPolicy", env, verbose=1,
           learning_rate=1e-4,     # More stable learning
           batch_size=32,          # Better gradients
           ent_coef=0.02)          # Higher exploration
model.learn(total_timesteps=500000)
```

#### Advanced Training Setup

```python
from pokemon_red_ai import PokemonTrainer, load_config

# Load configuration from file
config = load_config("config.yaml")

# Create trainer with improved exploration focus
trainer = PokemonTrainer(
    rom_path="PokemonRed.gb",
    save_dir="./advanced_training/",
    reward_strategy="exploration",    # Default improved strategy
    observation_type="multi_modal"
)

# Train with optimized hyperparameters
trainer.train(
    total_timesteps=1000000,         # Extended training
    algorithm="PPO",
    learning_rate=1e-4,              # More stable
    batch_size=32,                   # Better gradients
    n_epochs=5,                      # Prevent overfitting
    gamma=0.995,                     # Long-term thinking
    ent_coef=0.02,                   # Higher exploration
    max_episode_steps=15000,         # 3x longer episodes
    show_plots=True,
    save_freq=25000                  # More frequent saves
)

# Get detailed training statistics
stats = trainer.get_training_summary()
print(f"Best exploration: {stats['training_stats']['best_exploration']}")
```

## 🏗️ Architecture

The project is organized into modular components:

```
pokemon_red_ai/
├── game/               # Game interface components
│   ├── agent.py       # Main Pokemon Red agent
│   ├── controls.py    # Input controls and screen detection
│   └── memory.py      # Memory addresses and state reading
├── environment/        # RL environment components
│   ├── gym_env.py     # Gymnasium environment wrapper
│   ├── rewards.py     # Reward calculation strategies
│   └── observations.py # Observation processing
├── training/          # Training infrastructure
│   ├── trainer.py     # Main trainer class
│   ├── callbacks.py   # Training callbacks
│   └── models.py      # Model creation utilities
├── utils/             # Utility functions
│   ├── config.py      # Configuration management
│   └── file_utils.py  # File operations
└── cli/               # Command-line interface
    └── commands.py    # CLI commands
```

### Key Components

- **PokemonRedAgent**: Core game interface handling PyBoy emulation
- **PokemonRedGymEnv**: Gymnasium-compatible RL environment
- **RewardCalculator**: Flexible reward calculation system
- **PokemonTrainer**: High-level training orchestration
- **Callbacks**: Training monitoring and visualization

## ⚙️ Configuration

### YAML Configuration

Create a `config.yaml` file for reproducible training with improved defaults:

```yaml
# Training Configuration (IMPROVED DEFAULTS)
training:
  total_timesteps: 500000          # 5x increase
  algorithm: "PPO"
  max_episode_steps: 15000         # 3x longer episodes
  reward_strategy: "exploration"   # Exploration-focused
  observation_type: "multi_modal"
  learning_rate: 0.0001           # More stable
  batch_size: 32                  # Better gradients
  n_epochs: 5                     # Prevent overfitting
  gamma: 0.995                    # Long-term thinking
  ent_coef: 0.02                  # Higher exploration
  show_plots: true

# Improved Reward Configuration
rewards:
  exploration_reward: 5.0         # 5x higher
  new_map_reward: 100.0          # 5x higher
  time_penalty: -0.001           # 10x lower pressure
  level_reward_multiplier: 25.0   # Balanced
  badge_reward_multiplier: 150.0  # Balanced
  death_penalty: -50.0           # Less harsh

# Environment Configuration
environment:
  rom_path: "PokemonRed.gb"
  headless: true
  max_episode_steps: 15000        # Extended episodes
  screen_size: [80, 72]
```

### Environment Variables

Configure via environment variables:

```bash
# Example environment variable overrides
export POKEMON_AI_TRAINING__LEARNING_RATE=0.0001
export POKEMON_AI_REWARDS__EXPLORATION_REWARD=5.0
export POKEMON_AI_ENVIRONMENT__MAX_EPISODE_STEPS=15000
```

## 🎯 Training Strategies

### Exploration-Focused (Default & Recommended)
Optimized for discovering new areas and preventing getting stuck:
```python
trainer = PokemonTrainer(rom_path="PokemonRed.gb", reward_strategy="exploration")
```

**Features:**
- 5x higher exploration rewards
- Massive bonuses for new map discovery
- Anti-stuck mechanisms
- Distance-based bonuses
- Progressive milestone rewards

### Standard (Balanced)
Balanced approach for general gameplay:
```python
trainer = PokemonTrainer(rom_path="PokemonRed.gb", reward_strategy="standard")
```

### Progress-Focused
Optimized for story progression and badges:
```python
trainer = PokemonTrainer(rom_path="PokemonRed.gb", reward_strategy="progress")
```

### Sparse
Only rewards major achievements:
```python
trainer = PokemonTrainer(rom_path="PokemonRed.gb", reward_strategy="sparse")
```

### Observation Types

#### Multi-Modal (Default)
Combines screen, position, stats, and exploration data:
```python
env = PokemonRedGymEnv(rom_path="PokemonRed.gb", observation_type="multi_modal")
```

#### Screen-Only
Only uses visual screen information:
```python
env = PokemonRedGymEnv(rom_path="PokemonRed.gb", observation_type="screen_only")
```

#### Minimal
Compact feature vector for faster training:
```python
env = PokemonRedGymEnv(rom_path="PokemonRed.gb", observation_type="minimal")
```

## 🔬 Advanced Usage

### Custom Reward Function

```python
from pokemon_red_ai.environment import BaseRewardCalculator, RewardConfig

class CustomRewardCalculator(BaseRewardCalculator):
    def calculate_reward(self, current_state):
        reward = 0.0
        
        # Your custom reward logic here
        if current_state['stats']['badges'] > 0:
            reward += 1000
            
        return reward

# Use custom calculator
from pokemon_red_ai import PokemonRedGymEnv

env = PokemonRedGymEnv(
    rom_path="PokemonRed.gb",
    reward_calculator=CustomRewardCalculator()
)
```

### Hyperparameter Optimization

```python
from pokemon_red_ai.training import optimize_hyperparameters

# Optimize hyperparameters using Optuna
best_params = optimize_hyperparameters(
    env=env,
    algorithm='PPO',
    n_trials=50
)

print(f"Best hyperparameters: {best_params}")
```

### Multi-Environment Training

```python
from pokemon_red_ai.environment import PokemonRedVecEnv

# Create vectorized environment
vec_env = PokemonRedVecEnv(
    rom_paths=["PokemonRed.gb"] * 4,  # 4 parallel environments
    headless=True
)

# Train with parallel environments
model = PPO("MultiInputPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1000000)
```

### Training Callbacks

```python
from pokemon_red_ai.training import EnhancedTrainingCallback, EarlyStopping

# Create custom callbacks
callbacks = [
    EnhancedTrainingCallback(show_plots=True),
    EarlyStopping(patience=10, min_delta=1.0)
]

trainer.train(
    total_timesteps=1000000,
    callbacks=callbacks
)
```

## 🐛 Troubleshooting

### Common Issues

#### ROM Not Found
```bash
# Ensure ROM file exists and path is correct
pokemon-ai find-roms .
```

#### PyBoy Compatibility Issues
```python
# Try different PyBoy initialization methods (handled automatically)
# Check PyBoy version
pip show pyboy
```

#### Memory Issues
```bash
# Reduce batch size or episode length
pokemon-ai train --rom PokemonRed.gb --batch-size 16 --max-episode-steps 10000
```

#### Training Not Progressing
```bash
# Use improved exploration strategy (now default)
pokemon-ai train --rom PokemonRed.gb --reward-strategy exploration

# Adjust learning rate if needed
pokemon-ai train --rom PokemonRed.gb --learning-rate 0.00005
```

### Debug Mode

```bash
# Run with verbose logging
pokemon-ai train --rom PokemonRed.gb -vv
```

## 📊 Performance Tips

### Better Exploration Results

1. **Use exploration strategy** (now default): `--reward-strategy exploration`
2. **Longer episodes**: `--max-episode-steps 15000` (now default)
3. **Extended training**: `--timesteps 500000` (now default)
4. **Monitor progress**: `--monitor-mode` to watch training
5. **Stable learning**: Lower learning rates and smaller batches (now default)

### Faster Training

1. **Use headless mode**: No `--monitor-mode` (trains ~2x faster)
2. **Reduce episode length**: `--max-episode-steps 10000` (if needed)
3. **Use GPU**: Ensure PyTorch CUDA is available
4. **Smaller batch size**: `--batch-size 16` (if memory limited)

### Better Results

1. **Default settings**: Just use `pokemon-ai train --rom PokemonRed.gb --monitor-mode`
2. **Patient training**: Run for 500k+ timesteps
3. **Multiple runs**: Train several models and pick the best
4. **Save frequently**: `--save-freq 25000` (now default)

## 🔬 What's Improved

### Key Changes from Previous Versions

**🎯 Exploration Focus:**
- Exploration rewards increased 5x (1.0 → 5.0)
- New map rewards increased 5x (20.0 → 100.0) 
- Time penalty reduced 10x (-0.01 → -0.001)

**⏱️ Episode Length:**
- Extended from 5,000 to 15,000 steps (3x longer)
- Allows much more exploration time

**🧠 Learning Stability:**
- Learning rate: 3e-4 → 1e-4 (more stable)
- Batch size: 64 → 32 (better gradients)
- Entropy coefficient: 0.01 → 0.02 (more exploration)

**🔧 Anti-Stuck Features:**
- Detects when agent is stuck in same area
- Penalties for frequent location revisits
- Progressive bonuses for consecutive exploration
- Distance bonuses for exploring far from start

**💾 Better Defaults:**
- Training: 100k → 500k timesteps
- Saves: Every 25k steps (more frequent)
- Strategy: Exploration-focused by default

## 🤝 Contributing

Contributions are welcome! Please see our [Developer Guide](DEVELOPER_GUIDE.md) for detailed information about:

- Project architecture and design patterns
- Development setup and workflow
- Testing strategies and conventions
- How to add new features

For quick contributions, check out our [Contributing Guidelines](#contributing-guidelines) below.

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{pokemon_red_ai,
  author = {Alan Chester},
  title = {Pokemon Red AI: Reinforcement Learning for Pokemon Red},
  year = {2025},
  url = {https://github.com/amcheste/pokemon-red-ai}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyBoy](https://github.com/Baekalfen/PyBoy) - Game Boy emulator in Python
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) - RL environment standard
- Pokemon Red ROM hacking community for memory address documentation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/pokemon-red-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pokemon-red-ai/discussions)
- **Email**: amcheste@gmail.com

## 🗺️ Roadmap

### Completed ✅
- Exploration-focused training improvements
- Extended episode lengths for better exploration
- Anti-stuck mechanisms and adaptive rewards
- Optimized learning parameters
- Enhanced CLI with better defaults

### Coming Soon 🚀
- Curriculum learning for progressive difficulty
- Intrinsic motivation and curiosity-driven exploration
- Multi-agent training environments
- Advanced save state management
- Automated hyperparameter optimization

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Disclaimer**: This project is for educational and research purposes only. You must own a legal copy of Pokemon Red to use this software.