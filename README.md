# 🎮 Pokemon Red AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyBoy](https://img.shields.io/badge/PyBoy-2.0+-green.svg)](https://github.com/Baekalfen/PyBoy)

A comprehensive reinforcement learning toolkit for training AI agents to play Pokemon Red using PyBoy emulation and Stable-Baselines3.

## 🌟 Features

- **🤖 Complete RL Environment**: Full Gymnasium-compatible environment for Pokemon Red
- **🎯 Multiple Reward Strategies**: Standard, exploration-focused, progress-focused, and sparse reward systems
- **📊 Live Training Visualization**: Real-time plots and metrics during training
- **🔧 Modular Architecture**: Clean, maintainable codebase with separated concerns
- **💾 Automatic State Management**: Handles game initialization, resets, and save states
- **🎨 Rich CLI**: Beautiful command-line interface powered by Click and Rich
- **📈 TensorBoard Integration**: Detailed training logs and metrics
- **🧪 Testing & Evaluation**: Comprehensive model testing and performance analysis

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Training Strategies](#training-strategies)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
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

### Training Your First Agent

```bash
# Basic training (100k timesteps)
pokemon-ai train --rom PokemonRed.gb --timesteps 100000

# Training with visualization
pokemon-ai train --rom PokemonRed.gb --timesteps 100000 --monitor-mode

# Training with custom settings
pokemon-ai train \
  --rom PokemonRed.gb \
  --timesteps 500000 \
  --reward-strategy exploration \
  --show-plots
```

### Testing a Trained Model

```bash
# Test trained model
pokemon-ai test --rom PokemonRed.gb --model models/best_model.zip --episodes 10
```

### Using the Python API

```python
from pokemon_red_ai import PokemonTrainer

# Create trainer
trainer = PokemonTrainer(
    rom_path="PokemonRed.gb",
    save_dir="./training_output/"
)

# Train agent
trainer.train(
    total_timesteps=100000,
    show_plots=True
)

# Test trained model
results = trainer.test(
    model_path="./training_output/models/best_model.zip",
    episodes=10
)

print(f"Average reward: {results['avg_reward']:.2f}")
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
# Standard training
pokemon-ai train --rom PokemonRed.gb --timesteps 100000

# With custom configuration
pokemon-ai train --rom PokemonRed.gb --config config.yaml

# Advanced training options
pokemon-ai train \
  --rom PokemonRed.gb \
  --timesteps 1000000 \
  --algorithm PPO \
  --reward-strategy progress \
  --observation-type multi_modal \
  --learning-rate 0.0003 \
  --batch-size 128 \
  --save-freq 50000 \
  --show-game \
  --show-plots
```

#### Testing and Evaluation

```bash
# Test model with visualization
pokemon-ai test \
  --rom PokemonRed.gb \
  --model models/best_model.zip \
  --episodes 10 \
  --render

# Save test results
pokemon-ai test \
  --rom PokemonRed.gb \
  --model models/best_model.zip \
  --save-results results.json
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

# Quick training with defaults
trainer = create_trainer("PokemonRed.gb")
trainer.train(timesteps=100000)
```

#### Custom Environment

```python
from pokemon_red_ai import PokemonRedGymEnv, RewardConfig

# Create custom reward configuration
reward_config = RewardConfig(
    exploration_reward=2.0,
    new_map_reward=50.0,
    level_reward_multiplier=100.0
)

# Create environment
env = PokemonRedGymEnv(
    rom_path="PokemonRed.gb",
    reward_strategy="exploration",
    reward_config=reward_config,
    max_episode_steps=10000
)

# Use with any RL library
from stable_baselines3 import PPO

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

#### Advanced Training Setup

```python
from pokemon_red_ai import PokemonTrainer, load_config

# Load configuration from file
config = load_config("config.yaml")

# Create trainer with custom settings
trainer = PokemonTrainer(
    rom_path="PokemonRed.gb",
    save_dir="./advanced_training/",
    reward_strategy="progress",
    observation_type="multi_modal"
)

# Train with custom hyperparameters
trainer.train(
    total_timesteps=1000000,
    algorithm="PPO",
    learning_rate=0.0003,
    batch_size=128,
    n_epochs=15,
    show_plots=True,
    save_freq=50000
)

# Get training statistics
stats = trainer.get_training_summary()
print(stats)
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

Create a `config.yaml` file for reproducible training:

```yaml
# Training Configuration
training:
  total_timesteps: 100000
  algorithm: "PPO"
  max_episode_steps: 5000
  reward_strategy: "standard"
  observation_type: "multi_modal"
  learning_rate: 0.0003
  batch_size: 64
  show_plots: true

# Reward Configuration
rewards:
  exploration_reward: 1.0
  new_map_reward: 20.0
  level_reward_multiplier: 50.0
  badge_reward_multiplier: 200.0
  death_penalty: -100.0

# Environment Configuration
environment:
  rom_path: "PokemonRed.gb"
  headless: true
  max_episode_steps: 5000
  screen_size: [80, 72]
```

### Environment Variables

Configure via environment variables:

```bash
# Example environment variable overrides
export POKEMON_AI_TRAINING__LEARNING_RATE=0.0001
export POKEMON_AI_REWARDS__EXPLORATION_REWARD=2.0
export POKEMON_AI_ENVIRONMENT__MAX_EPISODE_STEPS=10000
```

## 🎯 Training Strategies

### Reward Strategies

#### Standard (Default)
Balanced approach for general gameplay:
```python
trainer = PokemonTrainer(rom_path="PokemonRed.gb", reward_strategy="standard")
```

#### Exploration-Focused
Emphasizes map coverage and discovery:
```python
trainer = PokemonTrainer(rom_path="PokemonRed.gb", reward_strategy="exploration")
```

#### Progress-Focused
Optimized for story progression and badges:
```python
trainer = PokemonTrainer(rom_path="PokemonRed.gb", reward_strategy="progress")
```

#### Sparse
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
pokemon-ai train --rom PokemonRed.gb --batch-size 32 --max-episode-steps 2000
```

#### Training Not Progressing
```bash
# Try different reward strategy
pokemon-ai train --rom PokemonRed.gb --reward-strategy exploration

# Adjust learning rate
pokemon-ai train --rom PokemonRed.gb --learning-rate 0.0001
```

### Debug Mode

```bash
# Run with verbose logging
pokemon-ai train --rom PokemonRed.gb -vv
```

## 📊 Performance Tips

### Faster Training

1. **Use headless mode** (default): `--headless`
2. **Reduce observation resolution**: Smaller screen size
3. **Use minimal observations**: `--observation-type minimal`
4. **Increase batch size**: `--batch-size 128` (if you have enough RAM)
5. **Use GPU**: Ensure PyTorch CUDA is available

### Better Results

1. **Longer training**: `--timesteps 1000000`
2. **Appropriate reward strategy**: Choose based on your goal
3. **Tune hyperparameters**: Use `optimize_hyperparameters()`
4. **Multi-environment training**: Use vectorized environments
5. **Save frequency**: `--save-freq 10000` to keep best models

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/pokemon-red-ai.git
cd pokemon-red-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

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

Coming Soon...

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Disclaimer**: This project is for educational and research purposes only. You must own a legal copy of Pokemon Red to use this software.