"""
Training callbacks for Pokemon Red RL.

This module provides callback classes for monitoring and controlling
the training process, including model saving, progress tracking, and
live visualization.
"""

import os
import numpy as np
import logging
from collections import deque
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class TrainingCallback(BaseCallback):
    """
    Standard callback for Pokemon Red training with basic functionality.

    Provides model saving, progress tracking, and basic logging without
    the overhead of live plotting.
    """

    def __init__(self,
                 save_freq: int = 10000,
                 save_path: str = './models/',
                 verbose: int = 1):
        """
        Initialize training callback.

        Args:
            save_freq: Frequency of model saves (in timesteps)
            save_path: Path to save models and logs
            verbose: Verbosity level (0=none, 1=info, 2=debug)
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.save_path_models = os.path.join(save_path, 'models')
        self.save_path_logs = os.path.join(save_path, 'logs')

        # Create directories
        os.makedirs(self.save_path_models, exist_ok=True)
        os.makedirs(self.save_path_logs, exist_ok=True)

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.exploration_progress = []
        self.best_reward = -float('inf')
        self.best_exploration = 0

        # Performance tracking
        self.total_episodes = 0
        self.model_saves = 0

        if self.verbose >= 1:
            logger.info(f"TrainingCallback initialized")
            logger.info(f"  Save frequency: {save_freq} timesteps")
            logger.info(f"  Save path: {save_path}")

    def _on_step(self) -> bool:
        """Called after each environment step."""
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        # Log training progress with error handling
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            try:
                # Convert to list if needed and get recent episodes
                ep_buffer = list(self.model.ep_info_buffer)
                recent_episodes = ep_buffer[-10:] if len(ep_buffer) >= 10 else ep_buffer

                # Extract data safely
                rewards = []
                lengths = []
                exploration_data = []

                for ep in recent_episodes:
                    if 'r' in ep:
                        rewards.append(ep['r'])
                    if 'l' in ep:
                        lengths.append(ep['l'])

                    # Extract exploration metrics if available
                    if 'maps_visited' in ep:
                        exploration_data.append(ep['maps_visited'])

                if rewards and lengths:  # Only proceed if we have data
                    mean_reward = np.mean(rewards)
                    mean_length = np.mean(lengths)
                    mean_exploration = np.mean(exploration_data) if exploration_data else 0

                    # Store statistics
                    self.episode_rewards.extend(rewards)
                    self.episode_lengths.extend(lengths)
                    if exploration_data:
                        self.exploration_progress.extend(exploration_data)

                    self.total_episodes += len(rewards)

                    # Track best performance
                    if mean_reward > self.best_reward:
                        self.best_reward = mean_reward
                        # Save best model
                        best_model_path = os.path.join(self.save_path_models, 'best_model')
                        self.model.save(best_model_path)
                        self.model_saves += 1

                        if self.verbose >= 1:
                            logger.info(f"🎉 New best model saved! Reward: {mean_reward:.2f}")

                    # Track best exploration
                    if mean_exploration > self.best_exploration:
                        self.best_exploration = mean_exploration

                    # Periodic detailed logging
                    if self.verbose >= 1 and self.num_timesteps % 5000 == 0:
                        logger.info(f"Step {self.num_timesteps:,}")
                        logger.info(f"  Reward: {mean_reward:.2f} (best: {self.best_reward:.2f})")
                        logger.info(f"  Length: {mean_length:.0f} steps")
                        logger.info(f"  Episodes: {self.total_episodes}")
                        if exploration_data:
                            logger.info(f"  Maps: {mean_exploration:.1f} (best: {self.best_exploration:.1f})")

                    # Brief periodic logging
                    elif self.verbose >= 1 and self.num_timesteps % 1000 == 0:
                        logger.info(f"Step {self.num_timesteps:,}: "
                                    f"Reward={mean_reward:.2f}, Length={mean_length:.0f}")

            except (TypeError, IndexError, KeyError, AttributeError) as e:
                if self.verbose >= 2:
                    logger.warning(f"Episode buffer processing failed: {e}")

        # Periodic model saving
        if self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(self.save_path_models, f'model_{self.num_timesteps}')
            self.model.save(model_path)
            self.model_saves += 1

            if self.verbose >= 1:
                logger.info(f"📦 Checkpoint saved: model_{self.num_timesteps}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        stats = {
            'num_timesteps': self.num_timesteps,
            'total_episodes': self.total_episodes,
            'model_saves': self.model_saves,
            'best_reward': self.best_reward,
            'best_exploration': self.best_exploration
        }

        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 100 else self.episode_rewards
            stats.update({
                'recent_avg_reward': np.mean(recent_rewards),
                'recent_reward_std': np.std(recent_rewards),
                'total_recorded_episodes': len(self.episode_rewards)
            })

        if self.episode_lengths:
            recent_lengths = self.episode_lengths[-100:] if len(self.episode_lengths) > 100 else self.episode_lengths
            stats.update({
                'recent_avg_length': np.mean(recent_lengths),
                'recent_length_std': np.std(recent_lengths)
            })

        return stats


class EnhancedTrainingCallback(TrainingCallback):
    """
    Enhanced callback with live visualizations and detailed monitoring.

    Extends TrainingCallback with real-time plotting capabilities and
    more detailed statistics tracking.
    """

    def __init__(self,
                 save_freq: int = 10000,
                 save_path: str = './models/',
                 show_plots: bool = True,
                 plot_freq: int = 1000,
                 verbose: int = 1):
        """
        Initialize enhanced training callback.

        Args:
            save_freq: Frequency of model saves
            save_path: Path to save models and logs
            show_plots: Whether to show live plots
            plot_freq: Frequency of plot updates (in timesteps)
            verbose: Verbosity level
        """
        super().__init__(save_freq, save_path, verbose)

        self.show_plots = show_plots
        self.plot_freq = plot_freq

        # Extended statistics with limited history for plotting
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.maps_discovered = deque(maxlen=1000)
        self.badges_earned = deque(maxlen=1000)
        self.hp_ratios = deque(maxlen=1000)

        # Timestep tracking for x-axis
        self.timesteps_history = deque(maxlen=1000)

        # Live plotting setup
        if self.show_plots:
            self.setup_live_plots()

        if self.verbose >= 1:
            logger.info(f"EnhancedTrainingCallback initialized")
            logger.info(f"  Live plots: {'enabled' if show_plots else 'disabled'}")
            logger.info(f"  Plot frequency: {plot_freq} timesteps")

    def setup_live_plots(self) -> None:
        """Setup matplotlib for live plotting."""
        try:
            plt.ion()  # Interactive mode
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle('Pokemon Red RL Training Progress', fontsize=14, fontweight='bold')

            # Configure subplots
            self.axes[0, 0].set_title('Episode Rewards')
            self.axes[0, 0].set_xlabel('Timestep')
            self.axes[0, 0].set_ylabel('Reward')
            self.axes[0, 0].grid(True, alpha=0.3)

            self.axes[0, 1].set_title('Episode Length')
            self.axes[0, 1].set_xlabel('Timestep')
            self.axes[0, 1].set_ylabel('Steps')
            self.axes[0, 1].grid(True, alpha=0.3)

            self.axes[1, 0].set_title('Maps Discovered')
            self.axes[1, 0].set_xlabel('Timestep')
            self.axes[1, 0].set_ylabel('Unique Maps')
            self.axes[1, 0].grid(True, alpha=0.3)

            self.axes[1, 1].set_title('Badges Earned')
            self.axes[1, 1].set_xlabel('Timestep')
            self.axes[1, 1].set_ylabel('Badge Count')
            self.axes[1, 1].grid(True, alpha=0.3)

            # Initialize empty line objects
            self.lines = {}
            colors = ['blue', 'green', 'red', 'purple']
            for i, (ax, color) in enumerate(zip(self.axes.flat, colors)):
                line, = ax.plot([], [], color=color, alpha=0.7, linewidth=1.5)
                self.lines[i] = line

            plt.tight_layout()
            plt.show(block=False)

            if self.verbose >= 1:
                logger.info("Live plotting initialized successfully")

        except Exception as e:
            logger.error(f"Failed to setup live plots: {e}")
            self.show_plots = False

    def update_plots(self) -> None:
        """Update the live plots with current data."""
        if not self.show_plots or len(self.timesteps_history) == 0:
            return

        try:
            timesteps = list(self.timesteps_history)

            # Update reward plot
            if len(self.episode_rewards) > 0:
                rewards = list(self.episode_rewards)
                x_data = timesteps[-len(rewards):]
                self.lines[0].set_data(x_data, rewards)
                self.axes[0, 0].relim()
                self.axes[0, 0].autoscale_view()

            # Update length plot
            if len(self.episode_lengths) > 0:
                lengths = list(self.episode_lengths)
                x_data = timesteps[-len(lengths):]
                self.lines[1].set_data(x_data, lengths)
                self.axes[0, 1].relim()
                self.axes[0, 1].autoscale_view()

            # Update maps plot
            if len(self.maps_discovered) > 0:
                maps = list(self.maps_discovered)
                x_data = timesteps[-len(maps):]
                self.lines[2].set_data(x_data, maps)
                self.axes[1, 0].relim()
                self.axes[1, 0].autoscale_view()

            # Update badges plot
            if len(self.badges_earned) > 0:
                badges = list(self.badges_earned)
                x_data = timesteps[-len(badges):]
                self.lines[3].set_data(x_data, badges)
                self.axes[1, 1].relim()
                self.axes[1, 1].autoscale_view()

            # Refresh the plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        except Exception as e:
            if self.verbose >= 2:
                logger.warning(f"Plot update failed: {e}")

    def _on_rollout_end(self) -> None:
        """Enhanced rollout end processing with plotting."""
        # Call parent method for basic functionality
        super()._on_rollout_end()

        # Process episode info for enhanced statistics
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            try:
                recent_episodes = list(self.model.ep_info_buffer)

                for episode_info in recent_episodes:
                    reward = episode_info.get('r', 0)
                    length = episode_info.get('l', 0)

                    # Extract Pokemon-specific metrics
                    maps = episode_info.get('maps_visited', 0)
                    badges = episode_info.get('badges_earned', 0)
                    hp_ratio = episode_info.get('hp_ratio', 1.0)

                    # Store enhanced data
                    self.episode_rewards.append(reward)
                    self.episode_lengths.append(length)
                    self.maps_discovered.append(maps)
                    self.badges_earned.append(badges)
                    self.hp_ratios.append(hp_ratio)
                    self.timesteps_history.append(self.num_timesteps)

                # Update plots periodically
                if self.show_plots and self.num_timesteps % self.plot_freq == 0:
                    self.update_plots()

                # Enhanced logging
                if len(self.episode_rewards) >= 10 and self.num_timesteps % 2000 == 0:
                    recent_rewards = list(self.episode_rewards)[-10:]
                    recent_maps = list(self.maps_discovered)[-10:]
                    recent_badges = list(self.badges_earned)[-10:]
                    recent_hp = list(self.hp_ratios)[-10:]

                    if self.verbose >= 1:
                        logger.info(f"📊 Enhanced Stats (Step {self.num_timesteps:,}):")
                        logger.info(f"   Reward: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
                        logger.info(f"   Maps: {np.mean(recent_maps):.1f} (max: {max(recent_maps)})")
                        logger.info(f"   Badges: {max(recent_badges)} (avg: {np.mean(recent_badges):.1f})")
                        logger.info(f"   HP: {np.mean(recent_hp):.2f}")

            except Exception as e:
                if self.verbose >= 2:
                    logger.warning(f"Enhanced episode processing failed: {e}")

    def save_plots(self, filename: Optional[str] = None) -> None:
        """Save current plots to file."""
        if not self.show_plots:
            return

        try:
            if filename is None:
                filename = os.path.join(self.save_path_logs, f'training_plots_{self.num_timesteps}.png')

            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            if self.verbose >= 1:
                logger.info(f"Training plots saved: {filename}")

        except Exception as e:
            logger.error(f"Failed to save plots: {e}")

    def cleanup(self) -> None:
        """Clean up plotting resources."""
        if self.show_plots and hasattr(self, 'fig'):
            try:
                # Save final plots
                self.save_plots()
                # Close figure
                plt.close(self.fig)
                plt.ioff()  # Turn off interactive mode
                if self.verbose >= 1:
                    logger.info("Live plotting cleanup completed")
            except Exception as e:
                logger.warning(f"Plot cleanup failed: {e}")

    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced training statistics."""
        base_stats = super().get_statistics()

        enhanced_stats = base_stats.copy()

        if len(self.maps_discovered) > 0:
            recent_maps = list(self.maps_discovered)[-100:]
            enhanced_stats.update({
                'recent_avg_maps': np.mean(recent_maps),
                'max_maps_discovered': max(self.maps_discovered),
                'recent_maps_std': np.std(recent_maps)
            })

        if len(self.badges_earned) > 0:
            recent_badges = list(self.badges_earned)[-100:]
            enhanced_stats.update({
                'recent_avg_badges': np.mean(recent_badges),
                'max_badges_earned': max(self.badges_earned),
                'recent_badges_std': np.std(recent_badges)
            })

        if len(self.hp_ratios) > 0:
            recent_hp = list(self.hp_ratios)[-100:]
            enhanced_stats.update({
                'recent_avg_hp_ratio': np.mean(recent_hp),
                'recent_hp_ratio_std': np.std(recent_hp)
            })

        return enhanced_stats


class EarlyStopping(BaseCallback):
    """
    Early stopping callback for Pokemon Red training.

    Stops training if no improvement in reward for a specified number
    of evaluations, helping prevent overfitting and save compute time.
    """

    def __init__(self,
                 check_freq: int = 10000,
                 patience: int = 5,
                 min_delta: float = 1.0,
                 verbose: int = 1):
        """
        Initialize early stopping callback.

        Args:
            check_freq: Frequency of improvement checks (in timesteps)
            patience: Number of checks to wait for improvement
            min_delta: Minimum improvement to reset patience
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.min_delta = min_delta

        self.best_reward = -float('inf')
        self.wait_count = 0
        self.stopped_early = False

    def _on_step(self) -> bool:
        """Check for early stopping condition."""
        if self.num_timesteps % self.check_freq == 0:
            # Get recent performance
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                try:
                    recent_episodes = list(self.model.ep_info_buffer)[-20:]  # Last 20 episodes
                    rewards = [ep['r'] for ep in recent_episodes if 'r' in ep]

                    if rewards:
                        current_reward = np.mean(rewards)

                        # Check for improvement
                        if current_reward > self.best_reward + self.min_delta:
                            self.best_reward = current_reward
                            self.wait_count = 0
                            if self.verbose >= 1:
                                logger.info(f"✅ Performance improved to {current_reward:.2f}")
                        else:
                            self.wait_count += 1
                            if self.verbose >= 1:
                                logger.info(f"⏳ No improvement for {self.wait_count}/{self.patience} checks")

                            # Check if we should stop
                            if self.wait_count >= self.patience:
                                self.stopped_early = True
                                if self.verbose >= 1:
                                    logger.info(f"🛑 Early stopping triggered after {self.num_timesteps} timesteps")
                                    logger.info(f"   Best reward: {self.best_reward:.2f}")
                                    logger.info(f"   Current reward: {current_reward:.2f}")
                                return False  # Stop training

                except Exception as e:
                    if self.verbose >= 2:
                        logger.warning(f"Early stopping check failed: {e}")

        return True  # Continue training


class PerformanceMonitor(BaseCallback):
    """
    Performance monitoring callback for system resource tracking.

    Monitors training performance metrics like FPS, memory usage,
    and environment step times.
    """

    def __init__(self,
                 monitor_freq: int = 1000,
                 verbose: int = 1):
        """
        Initialize performance monitor.

        Args:
            monitor_freq: Frequency of performance checks
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.monitor_freq = monitor_freq

        # Performance tracking
        self.step_times = deque(maxlen=1000)
        self.last_time = None

        try:
            import psutil
            self.psutil_available = True
            self.process = psutil.Process()
        except ImportError:
            self.psutil_available = False
            if verbose >= 1:
                logger.warning("psutil not available - memory monitoring disabled")

    def _on_step(self) -> bool:
        """Monitor performance metrics."""
        import time

        current_time = time.time()
        if self.last_time is not None:
            step_time = current_time - self.last_time
            self.step_times.append(step_time)
        self.last_time = current_time

        # Periodic performance reporting
        if self.num_timesteps % self.monitor_freq == 0 and len(self.step_times) > 0:
            avg_step_time = np.mean(self.step_times)
            fps = 1.0 / avg_step_time if avg_step_time > 0 else 0

            perf_info = f"🔧 Performance (Step {self.num_timesteps:,}):"
            perf_info += f"\n   FPS: {fps:.1f}"
            perf_info += f"\n   Avg step time: {avg_step_time * 1000:.2f}ms"

            if self.psutil_available:
                try:
                    memory_mb = self.process.memory_info().rss / 1024 / 1024
                    cpu_percent = self.process.cpu_percent()
                    perf_info += f"\n   Memory: {memory_mb:.1f}MB"
                    perf_info += f"\n   CPU: {cpu_percent:.1f}%"
                except Exception:
                    pass

            if self.verbose >= 1:
                logger.info(perf_info)

        return True

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        if not self.step_times:
            return {}

        step_times = list(self.step_times)
        avg_step_time = np.mean(step_times)

        stats = {
            'avg_step_time_ms': avg_step_time * 1000,
            'fps': 1.0 / avg_step_time if avg_step_time > 0 else 0,
            'step_time_std_ms': np.std(step_times) * 1000
        }

        if self.psutil_available:
            try:
                stats.update({
                    'memory_mb': self.process.memory_info().rss / 1024 / 1024,
                    'cpu_percent': self.process.cpu_percent()
                })
            except Exception:
                pass

        return stats