"""
Command-line interface for Pokemon Red RL.

This module provides a comprehensive CLI for training, testing, and managing
Pokemon Red RL projects using Click framework with improved defaults.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

# Import package components
from ..training import PokemonTrainer
from ..utils import (
    load_config, create_default_config, validate_config,
    create_directories, find_rom_files, validate_rom_file,
    get_project_info, cleanup_rom_save_files, get_config_template
)
from .. import __version__

# Rich console for pretty output
console = Console()

# Common options
rom_option = click.option(
    '--rom', '-r', 'rom_path',
    required=True,
    help='Path to Pokemon Red ROM file',
    type=click.Path(exists=True, readable=True)
)

config_option = click.option(
    '--config', '-c', 'config_path',
    help='Path to configuration file',
    type=click.Path(exists=True, readable=True)
)

save_dir_option = click.option(
    '--save-dir', '-s', 'save_dir',
    default='./pokemon_training/',
    help='Directory to save training artifacts',
    type=click.Path()
)

verbose_option = click.option(
    '--verbose', '-v',
    count=True,
    help='Increase verbosity (-v for INFO, -vv for DEBUG)'
)


def setup_logging(verbose_count: int = 0) -> None:
    """Setup logging based on verbosity level."""
    if verbose_count == 0:
        level = logging.WARNING
    elif verbose_count == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def validate_rom_with_feedback(rom_path: str) -> bool:
    """Validate ROM file and provide user feedback."""
    console.print(f"🎮 Validating ROM file: {rom_path}")

    rom_info = validate_rom_file(rom_path)

    if not rom_info['valid']:
        console.print("❌ [red]ROM validation failed:[/red]")
        if 'error' in rom_info:
            console.print(f"   Error: {rom_info['error']}")
        else:
            console.print(f"   File exists: {rom_info['exists']}")
            console.print(f"   Readable: {rom_info['readable']}")
            console.print(f"   Size: {rom_info['size_mb']:.1f}MB")
        return False

    console.print("✅ [green]ROM validation passed[/green]")
    console.print(f"   Size: {rom_info['size_mb']:.1f}MB")
    if rom_info['is_pokemon_red']:
        console.print("   🎯 [green]Detected as Pokemon Red ROM[/green]")

    return True


@click.group()
@click.version_option(version=__version__, prog_name='pokemon-rl')
@click.pass_context
def main(ctx):
    """
    🎮 Pokemon Red RL - Train AI agents to play Pokemon Red

    A comprehensive toolkit for training reinforcement learning agents
    to play Pokemon Red using PyBoy emulation with improved defaults.
    """
    ctx.ensure_object(dict)

    # Display banner
    if not ctx.resilient_parsing:
        console.print(Panel.fit(
            f"[bold blue]Pokemon Red RL v{__version__}[/bold blue]\n"
            "[dim]Train AI agents to play Pokemon Red[/dim]\n"
            "[green]🚀 Now with improved exploration training![/green]",
            border_style="blue"
        ))


@main.command()
@rom_option
@config_option
@save_dir_option
@verbose_option
@click.option('--timesteps', '-t', default=500000, help='Training timesteps (IMPROVED: 5x increase)', type=int)
@click.option('--algorithm', '-a', default='RecurrentPPO', help='RL algorithm',
              type=click.Choice(['PPO', 'RecurrentPPO', 'A2C', 'DQN']))
@click.option('--reward-strategy', default='events', help='Reward strategy (events = event-flag milestones)',
              type=click.Choice(['standard', 'exploration', 'progress', 'sparse', 'events']))
@click.option('--observation-type', default='multi_modal', help='Observation type',
              type=click.Choice(['multi_modal', 'screen_only', 'minimal', 'pixel', 'symbolic', 'hybrid']))
@click.option('--show-game/--no-show-game', default=False, help='Show game window during training')
@click.option('--show-plots/--no-show-plots', default=False, help='Show live training plots')
@click.option('--monitor-mode', is_flag=True, help='Enable monitoring (game + plots)')
@click.option('--learning-rate', type=float, default=1e-4, help='Learning rate (IMPROVED: more stable)')
@click.option('--batch-size', type=int, default=32, help='Batch size (IMPROVED: better gradients)')
@click.option('--save-freq', default=25000, type=int, help='Model save frequency (IMPROVED: more frequent)')
@click.option('--max-episode-steps', default=15000, type=int, help='Maximum steps per episode (IMPROVED: 3x longer)')
@click.option('--clean-start/--no-clean-start', default=True, help='Clean ROM save files before training')
@click.option('--save-state', type=click.Path(exists=True), default=None,
              help='PyBoy .state file for fast episode resets')
@click.option('--wandb-project', default='pokemon-red-ai', help='W&B project name')
@click.option('--no-wandb', is_flag=True, help='Disable Weights & Biases logging')
@click.option('--seed', type=int, default=None, help='Random seed for reproducibility')
def train(rom_path, config_path, save_dir, verbose, timesteps, algorithm, reward_strategy,
          observation_type, show_game, show_plots, monitor_mode, learning_rate, batch_size,
          save_freq, max_episode_steps, clean_start, save_state, wandb_project, no_wandb, seed):
    """🚀 Train a Pokemon Red RL game with improved exploration-focused settings."""
    setup_logging(verbose)

    if monitor_mode:
        show_game = show_plots = True

    # Validate ROM
    if not validate_rom_with_feedback(rom_path):
        sys.exit(1)

    # Load configuration
    config = None
    if config_path:
        console.print(f"📄 Loading configuration from: {config_path}")
        try:
            config = load_config(config_path)
            if not validate_config(config):
                console.print("❌ [red]Configuration validation failed[/red]")
                sys.exit(1)
        except Exception as e:
            console.print(f"❌ [red]Failed to load configuration: {e}[/red]")
            sys.exit(1)

    # Clean ROM save files if requested
    if clean_start:
        console.print("🧹 Cleaning ROM save files...")
        removed_files = cleanup_rom_save_files(rom_path)
        if removed_files:
            console.print(f"   Removed {len(removed_files)} save files")

    # Create trainer
    console.print("🎯 Initializing improved trainer...")
    try:
        trainer = PokemonTrainer(
            rom_path=rom_path,
            save_dir=save_dir,
            reward_strategy=reward_strategy,
            observation_type=observation_type
        )
    except Exception as e:
        console.print(f"❌ [red]Failed to initialize trainer: {e}[/red]")
        sys.exit(1)

    # Build training parameters
    train_params = {
        'total_timesteps': timesteps,
        'algorithm': algorithm,
        'show_game': show_game,
        'show_plots': show_plots,
        'max_episode_steps': max_episode_steps,
        'save_freq': save_freq,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        # Additional improved parameters
        'n_epochs': 5,          # Reduced from default 10
        'gamma': 0.995,         # Increased from default 0.99
        'gae_lambda': 0.98,     # Improved from default 0.95
        'clip_range': 0.15,     # Reduced from default 0.2
        'ent_coef': 0.02,       # Increased from default 0.01
        'vf_coef': 0.25         # Reduced from default 0.5
    }

    # Display training info with improvements highlighted
    info_table = Table(title="🚀 Improved Training Configuration", show_header=True)
    info_table.add_column("Parameter", style="cyan")
    info_table.add_column("Value", style="magenta")
    info_table.add_column("Improvement", style="green")

    info_table.add_row("ROM File", os.path.basename(rom_path), "")
    info_table.add_row("Algorithm", algorithm, "")
    info_table.add_row("Timesteps", f"{timesteps:,}", "5x increase" if timesteps >= 500000 else "")
    info_table.add_row("Reward Strategy", reward_strategy, "🎯 Exploration focus" if reward_strategy == "exploration" else "")
    info_table.add_row("Episode Length", f"{max_episode_steps:,}", "3x longer" if max_episode_steps >= 15000 else "")
    info_table.add_row("Learning Rate", f"{learning_rate:.0e}", "⚡ More stable" if learning_rate <= 1e-4 else "")
    info_table.add_row("Batch Size", str(batch_size), "🎯 Better gradients" if batch_size <= 32 else "")
    info_table.add_row("Save Frequency", f"{save_freq:,}", "💾 More frequent" if save_freq <= 25000 else "")
    info_table.add_row("Observation Type", observation_type, "")
    info_table.add_row("Show Game", "Yes" if show_game else "No", "")
    info_table.add_row("Show Plots", "Yes" if show_plots else "No", "")
    info_table.add_row("Save Directory", save_dir, "")

    console.print(info_table)

    # Show key improvements
    console.print("\n✨ [bold green]Key Improvements:[/bold green]")
    console.print("   🚀 Episodes 3x longer (15,000 steps) for better exploration")
    console.print("   🎯 Exploration rewards 5x higher to encourage map discovery")
    console.print("   ⏰ Time penalty 10x lower to reduce pressure")
    console.print("   🧠 Learning parameters optimized for stability")
    console.print("   🔧 Anti-stuck mechanisms to prevent getting trapped")

    if not click.confirm("\n🤔 Start training with these improved settings?"):
        console.print("Training cancelled by user")
        sys.exit(0)

    # Start training
    console.print("\n🎮 [bold green]Starting Improved Pokemon Red RL Training![/bold green]")
    try:
        trainer.train(**train_params)
        console.print("🎉 [bold green]Training completed successfully![/bold green]")
    except KeyboardInterrupt:
        console.print("\n⏹️  [yellow]Training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"❌ [red]Training failed: {e}[/red]")
        sys.exit(1)


@main.command()
@rom_option
@verbose_option
@click.option('--model', '-m', required=True, help='Path to trained model', type=click.Path(exists=True))
@click.option('--episodes', '-e', default=10, help='Number of test episodes', type=int)
@click.option('--render/--no-render', default=True, help='Show game window during testing')
@click.option('--max-episode-steps', default=15000, type=int, help='Maximum steps per episode (IMPROVED: longer)')
@click.option('--save-results', help='Save test results to file', type=click.Path())
def test(rom_path, verbose, model, episodes, render, max_episode_steps, save_results):
    """🧪 Test a trained Pokemon Red RL model with improved episode length."""
    setup_logging(verbose)

    # Validate ROM
    if not validate_rom_with_feedback(rom_path):
        sys.exit(1)

    # Validate model file
    if not os.path.exists(model):
        console.print(f"❌ [red]Model file not found: {model}[/red]")
        sys.exit(1)

    console.print(f"🧪 Testing model: {os.path.basename(model)}")
    console.print(f"Episodes: {episodes}")
    console.print(f"Episode length: {max_episode_steps:,} steps")
    console.print(f"Render: {'Yes' if render else 'No'}")

    # Create trainer
    try:
        trainer = PokemonTrainer(rom_path=rom_path)
    except Exception as e:
        console.print(f"❌ [red]Failed to initialize trainer: {e}[/red]")
        sys.exit(1)

    # Run testing
    try:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
        ) as progress:
            task = progress.add_task("Testing model...", total=None)

            results = trainer.test(
                model_path=model,
                episodes=episodes,
                render=render,
                max_episode_steps=max_episode_steps
            )

        # Display results with improved metrics
        results_table = Table(title="🧪 Test Results", show_header=True)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="magenta")

        results_table.add_row("Episodes Tested", str(results['episodes_tested']))
        results_table.add_row("Average Reward", f"{results['avg_reward']:.2f}")
        results_table.add_row("Average Steps", f"{results['avg_steps']:.0f}")
        results_table.add_row("Average Maps Visited", f"{results['avg_maps_visited']:.1f}")
        results_table.add_row("Maximum Maps Visited", str(results.get('max_maps_visited', 'N/A')))
        results_table.add_row("Average Locations", f"{results.get('avg_locations_visited', 0):.0f}")
        results_table.add_row("Maximum Locations", str(results.get('max_locations_visited', 'N/A')))
        results_table.add_row("Exploration Efficiency", f"{results.get('avg_exploration_efficiency', 0):.4f}")
        results_table.add_row("Maximum Badges", str(results['max_badges']))
        results_table.add_row("Average Badges", f"{results['avg_badges']:.1f}")

        console.print(results_table)

        # Save results if requested
        if save_results:
            import json
            with open(save_results, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"✅ Results saved to: {save_results}")

        console.print("🎉 [bold green]Testing completed![/bold green]")

    except Exception as e:
        console.print(f"❌ [red]Testing failed: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument('project_dir', type=click.Path(), default='./pokemon_training/')
@verbose_option
def info(project_dir, verbose):
    """📊 Show information about a Pokemon Red RL project."""
    setup_logging(verbose)

    console.print(f"📊 Analyzing project: {project_dir}")

    try:
        project_info = get_project_info(project_dir)

        if not project_info['exists']:
            console.print(f"❌ [red]Project directory not found: {project_dir}[/red]")
            sys.exit(1)

        # Basic info
        info_table = Table(title="Project Information", show_header=True)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="magenta")

        info_table.add_row("Project Directory", project_info['project_dir'])
        info_table.add_row("Total Size", f"{project_info['total_size_mb']:.1f} MB")
        info_table.add_row("File Count", str(project_info['file_count']))

        if project_info.get('created_at'):
            info_table.add_row("Created", project_info['created_at'][:19])
        if project_info.get('last_modified'):
            info_table.add_row("Last Modified", project_info['last_modified'][:19])

        console.print(info_table)

        # Directory breakdown
        if project_info['directories']:
            dir_table = Table(title="Directory Breakdown", show_header=True)
            dir_table.add_column("Directory", style="cyan")
            dir_table.add_column("Exists", style="green")
            dir_table.add_column("Size (MB)", style="magenta", justify="right")
            dir_table.add_column("Files", style="yellow", justify="right")

            for dir_name, dir_info in project_info['directories'].items():
                exists = "✅" if dir_info['exists'] else "❌"
                size = f"{dir_info.get('size_mb', 0):.1f}" if dir_info['exists'] else "—"
                files = str(dir_info.get('file_count', 0)) if dir_info['exists'] else "—"
                dir_table.add_row(dir_name, exists, size, files)

            console.print(dir_table)

        # Models, configs, etc.
        if project_info['models']:
            console.print(f"🤖 [bold]Models found:[/bold] {len(project_info['models'])}")
            for model in project_info['models'][-5:]:  # Show last 5
                console.print(f"   • {os.path.basename(model)}")

        if project_info['configs']:
            console.print(f"⚙️  [bold]Configs found:[/bold] {len(project_info['configs'])}")
            for config in project_info['configs']:
                console.print(f"   • {os.path.basename(config)}")

        if project_info['rom_files']:
            console.print(f"🎮 [bold]ROM files found:[/bold] {len(project_info['rom_files'])}")
            for rom in project_info['rom_files']:
                console.print(f"   • {os.path.basename(rom)}")

    except Exception as e:
        console.print(f"❌ [red]Failed to analyze project: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument('config_path', type=click.Path(), default='./pokemon_config.yaml')
@verbose_option
@click.option('--template', is_flag=True, help='Show configuration template')
def config(config_path, verbose, template):
    """⚙️  Create or validate configuration files."""
    setup_logging(verbose)

    if template:
        console.print("📄 [bold]Configuration Template:[/bold]")
        template_content = get_config_template()
        console.print(Panel(template_content, border_style="green"))
        return

    if not os.path.exists(config_path):
        console.print(f"📄 Creating default configuration: {config_path}")
        try:
            config_obj = create_default_config(config_path)
            console.print("✅ [green]Configuration file created successfully![/green]")
            console.print(f"   Edit {config_path} to customize your settings")
        except Exception as e:
            console.print(f"❌ [red]Failed to create configuration: {e}[/red]")
            sys.exit(1)
    else:
        console.print(f"📄 Validating configuration: {config_path}")
        try:
            config_obj = load_config(config_path)
            if validate_config(config_obj):
                console.print("✅ [green]Configuration is valid![/green]")
            else:
                console.print("❌ [red]Configuration validation failed[/red]")
                sys.exit(1)
        except Exception as e:
            console.print(f"❌ [red]Failed to load configuration: {e}[/red]")
            sys.exit(1)


@main.command()
@click.argument('search_path', type=click.Path(exists=True), default='.')
@verbose_option
def find_roms(search_path, verbose):
    """🔍 Find Pokemon ROM files in a directory."""
    setup_logging(verbose)

    console.print(f"🔍 Searching for ROM files in: {search_path}")

    try:
        rom_files = find_rom_files(search_path)

        if not rom_files:
            console.print("❌ [yellow]No ROM files found[/yellow]")
            return

        rom_table = Table(title=f"ROM Files Found ({len(rom_files)})", show_header=True)
        rom_table.add_column("File", style="cyan")
        rom_table.add_column("Size (MB)", style="magenta", justify="right")
        rom_table.add_column("Valid", style="green")
        rom_table.add_column("Pokemon Red", style="yellow")

        for rom_file in rom_files:
            rom_info = validate_rom_file(rom_file)
            valid_icon = "✅" if rom_info['valid'] else "❌"
            pokemon_icon = "🎯" if rom_info.get('is_pokemon_red') else "❓"

            rom_table.add_row(
                str(rom_file.name),
                f"{rom_info['size_mb']:.1f}",
                valid_icon,
                pokemon_icon
            )

        console.print(rom_table)

    except Exception as e:
        console.print(f"❌ [red]ROM search failed: {e}[/red]")
        sys.exit(1)


@main.command()
@save_dir_option
@verbose_option
@click.option('--force', is_flag=True, help='Create directories without confirmation')
def init(save_dir, verbose, force):
    """🏗️  Initialize a new Pokemon Red RL project with improved defaults."""
    setup_logging(verbose)

    save_path = Path(save_dir)

    if save_path.exists() and any(save_path.iterdir()) and not force:
        console.print(f"⚠️  [yellow]Directory {save_dir} already exists and is not empty[/yellow]")
        if not click.confirm("Continue anyway?"):
            console.print("Project initialization cancelled")
            sys.exit(0)

    console.print(f"🏗️  Initializing improved Pokemon Red RL project in: {save_dir}")

    try:
        # Create directory structure
        dirs = create_directories(save_dir)

        # Create default config with improved settings
        config_path = save_path / 'pokemon_config.yaml'
        create_default_config(config_path)

        # Show what was created
        console.print("✅ [green]Project initialized successfully![/green]")
        console.print("\n📁 [bold]Created directories:[/bold]")
        for name, path in dirs.items():
            if name != 'base':
                console.print(f"   • {name}/")

        console.print(f"\n⚙️  [bold]Created configuration:[/bold] {config_path.name}")
        console.print("\n🚀 [bold]Next steps:[/bold]")
        console.print("   1. Place your Pokemon Red ROM in the project directory")
        console.print(f"   2. Edit {config_path.name} to customize training settings")
        console.print("   3. Run: pokemon-ai train --rom PokemonRed.gb")
        console.print("\n✨ [bold green]Improvements included:[/bold green]")
        console.print("   • Exploration-focused rewards by default")
        console.print("   • 3x longer episodes (15,000 steps)")
        console.print("   • Optimized learning parameters")
        console.print("   • Enhanced monitoring capabilities")

    except Exception as e:
        console.print(f"❌ [red]Project initialization failed: {e}[/red]")
        sys.exit(1)


@main.command()
@verbose_option
def doctor(verbose):
    """🩺 Check system requirements and package health."""
    setup_logging(verbose)

    console.print("🩺 [bold]Pokemon Red RL Health Check[/bold]")

    health_issues = []

    # Check Python version
    console.print("\n📋 [bold]System Requirements:[/bold]")

    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    console.print(f"   Python: {python_version}", end="")

    if sys.version_info >= (3, 8):
        console.print(" ✅")
    else:
        console.print(" ❌ (Requires Python 3.8+)")
        health_issues.append("Python version too old")

    # Check dependencies
    dependencies = [
        ('gymnasium', 'Gymnasium'),
        ('stable_baselines3', 'Stable Baselines3'),
        ('pyboy', 'PyBoy'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('matplotlib', 'Matplotlib'),
        ('yaml', 'PyYAML'),
        ('click', 'Click'),
        ('rich', 'Rich')
    ]

    console.print("\n📦 [bold]Dependencies:[/bold]")

    for module, name in dependencies:
        try:
            __import__(module)
            console.print(f"   {name}: ✅")
        except ImportError:
            console.print(f"   {name}: ❌ (Missing)")
            health_issues.append(f"Missing dependency: {name}")

    # Check GPU availability
    console.print("\n🖥️  [bold]Hardware:[/bold]")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            console.print(f"   GPU: ✅ ({gpu_count} device(s) - {gpu_name})")
        else:
            console.print("   GPU: ⚠️  (Not available - will use CPU)")
    except ImportError:
        console.print("   GPU: ❓ (PyTorch not available for detection)")

    # Overall health
    console.print(f"\n🎯 [bold]Overall Health:[/bold]")
    if not health_issues:
        console.print("   ✅ [green]All systems operational![/green]")
        console.print("   🚀 [green]Ready for improved Pokemon Red RL training![/green]")
    else:
        console.print("   ❌ [red]Issues found:[/red]")
        for issue in health_issues:
            console.print(f"      • {issue}")

        console.print("\n💡 [bold]Recommended fixes:[/bold]")
        console.print("   pip install pokemon-red-rl[all]")


if __name__ == '__main__':
    main()