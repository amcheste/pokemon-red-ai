"""
Training callbacks for Pokemon Red RL.

This module provides callback classes for monitoring and controlling
the training process, including model saving, progress tracking,
live visualization, and Weights & Biases experiment tracking.

macOS-compatible version that saves plots to files instead of displaying them live.
"""

import json
import os
import numpy as np
import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional
import matplotlib
# Use Agg backend (no GUI) to avoid conflicts with SDL on macOS
matplotlib.use('Agg')
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


class EnhancedTrainingCallback(BaseCallback):
    """
    Enhanced callback with plot saving and detailed monitoring.

    macOS-compatible version that saves plots to files instead of displaying them live.
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
            show_plots: Whether to save plots to files
            plot_freq: Frequency of plot updates (in timesteps)
            verbose: Verbosity level
        """
        super().__init__(verbose)

        self.save_freq = save_freq
        self.save_path = save_path
        self.show_plots = show_plots
        self.plot_freq = plot_freq

        # Create directories
        self.save_path_models = os.path.join(save_path, 'models')
        self.save_path_logs = os.path.join(save_path, 'logs')
        os.makedirs(self.save_path_models, exist_ok=True)
        os.makedirs(self.save_path_logs, exist_ok=True)

        # Extended statistics with limited history for plotting
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.maps_discovered = deque(maxlen=1000)
        self.badges_earned = deque(maxlen=1000)
        self.hp_ratios = deque(maxlen=1000)

        # Timestep tracking for x-axis
        self.timesteps_history = deque(maxlen=1000)

        self.best_reward = -float('inf')
        self.best_exploration = 0

        if self.show_plots:
            print("📊 macOS-compatible plotting enabled - plots will be saved to files")
            print(f"   Plot files will be saved to: {self.save_path_logs}")
            print("   Use 'open <filename>' or your image viewer to see plots")

        if self.verbose >= 1:
            logger.info(f"EnhancedTrainingCallback initialized (macOS mode)")
            logger.info(f"  Plot saving: {'enabled' if show_plots else 'disabled'}")
            logger.info(f"  Plot frequency: {plot_freq} timesteps")

    def create_and_save_plots(self) -> None:
        """Create plots and save them to files (no GUI display)"""
        if not self.show_plots:
            return

        try:
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Pokemon Red RL Training Progress (Step {self.num_timesteps})',
                         fontsize=14, fontweight='bold')

            # Get data - ENSURE we have data to plot
            timesteps = list(self.timesteps_history)
            rewards = list(self.episode_rewards)
            lengths = list(self.episode_lengths)
            maps = list(self.maps_discovered)
            badges = list(self.badges_earned)

            # DEBUG: Print what data we have
            print(f"🔍 DEBUG PLOTTING DATA:")
            print(
                f"   Timesteps: {len(timesteps)} points, range: {timesteps[:3] if timesteps else 'empty'}...{timesteps[-3:] if len(timesteps) > 3 else ''}")
            print(
                f"   Rewards: {len(rewards)} points, range: {rewards[:3] if rewards else 'empty'}...{rewards[-3:] if len(rewards) > 3 else ''}")
            print(f"   Lengths: {len(lengths)} points")
            print(f"   Maps: {len(maps)} points")
            print(f"   Badges: {len(badges)} points")

            # Plot 1: Episode Rewards - ALWAYS plot something, even if fake data
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Timestep')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)

            if len(rewards) > 0 and len(timesteps) >= len(rewards):
                x_data = timesteps[-len(rewards):]
                axes[0, 0].plot(x_data, rewards, 'b-o', alpha=0.7, linewidth=1.5, markersize=4)
                print(f"   ✅ Plotted {len(rewards)} reward points")
            else:
                # Add fake data so we can see something
                fake_x = [self.num_timesteps - 2000, self.num_timesteps - 1000, self.num_timesteps]
                fake_rewards = [-15, -10, -5]
                axes[0, 0].plot(fake_x, fake_rewards, 'r--o', alpha=0.5, label='No real data yet')
                axes[0, 0].legend()
                print(f"   ⚠️ No reward data, plotted fake data")

            # Plot 2: Episode Length
            axes[0, 1].set_title('Episode Length')
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].grid(True, alpha=0.3)

            if len(lengths) > 0 and len(timesteps) >= len(lengths):
                x_data = timesteps[-len(lengths):]
                axes[0, 1].plot(x_data, lengths, 'g-o', alpha=0.7, linewidth=1.5, markersize=4)
                print(f"   ✅ Plotted {len(lengths)} length points")
            else:
                fake_x = [self.num_timesteps - 2000, self.num_timesteps - 1000, self.num_timesteps]
                fake_lengths = [200, 180, 150]
                axes[0, 1].plot(fake_x, fake_lengths, 'orange', linestyle='--', marker='o', alpha=0.5,
                                label='No real data yet')
                axes[0, 1].legend()
                print(f"   ⚠️ No length data, plotted fake data")

            # Plot 3: Maps Discovered
            axes[1, 0].set_title('Maps Discovered')
            axes[1, 0].set_xlabel('Timestep')
            axes[1, 0].set_ylabel('Unique Maps')
            axes[1, 0].grid(True, alpha=0.3)

            if len(maps) > 0 and len(timesteps) >= len(maps):
                x_data = timesteps[-len(maps):]
                axes[1, 0].plot(x_data, maps, 'r-o', alpha=0.7, linewidth=1.5, markersize=4)
                print(f"   ✅ Plotted {len(maps)} map points")
            else:
                fake_x = [self.num_timesteps - 2000, self.num_timesteps - 1000, self.num_timesteps]
                fake_maps = [1, 1, 2]
                axes[1, 0].plot(fake_x, fake_maps, 'purple', linestyle='--', marker='o', alpha=0.5,
                                label='No real data yet')
                axes[1, 0].legend()
                print(f"   ⚠️ No map data, plotted fake data")

            # Plot 4: Badges Earned
            axes[1, 1].set_title('Badges Earned')
            axes[1, 1].set_xlabel('Timestep')
            axes[1, 1].set_ylabel('Badge Count')
            axes[1, 1].grid(True, alpha=0.3)

            if len(badges) > 0 and len(timesteps) >= len(badges):
                x_data = timesteps[-len(badges):]
                axes[1, 1].plot(x_data, badges, 'm-o', alpha=0.7, linewidth=1.5, markersize=4)
                print(f"   ✅ Plotted {len(badges)} badge points")
            else:
                fake_x = [self.num_timesteps - 2000, self.num_timesteps - 1000, self.num_timesteps]
                fake_badges = [0, 0, 0]
                axes[1, 1].plot(fake_x, fake_badges, 'brown', linestyle='--', marker='o', alpha=0.5,
                                label='No real data yet')
                axes[1, 1].legend()
                print(f"   ⚠️ No badge data, plotted fake data")

            # Add statistics text box
            stats_text = f"Latest Stats (Step {self.num_timesteps}):\n"
            stats_text += f"Episodes: {len(rewards)}\n"
            if len(rewards) > 0:
                stats_text += f"Avg Reward (last 10): {np.mean(rewards[-10:]):.2f}\n"
                stats_text += f"Best Reward: {self.best_reward:.2f}\n"
            if len(maps) > 0:
                stats_text += f"Max Maps: {max(maps)}\n"
            if len(badges) > 0:
                stats_text += f"Max Badges: {max(badges)}"

            # AMC
            #fig.text(0.98, 0.02, stats_text, fontsize=10,
            #         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
            #         horizontalalignment='right', verticalalignment='bottom')

            plt.tight_layout()

            # Save timestamped plot
            filename = os.path.join(self.save_path_logs, f'training_plots_{self.num_timesteps}.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')

            # Save "latest" version that overwrites
            latest_filename = os.path.join(self.save_path_logs, 'latest_training_plots.png')
            plt.savefig(latest_filename, dpi=150, bbox_inches='tight')

            plt.close(fig)  # Important: close figure to free memory

            print(f"📊 Training plots saved:")
            print(f"   Timestamped: {filename}")
            print(f"   Latest: {latest_filename}")

        except Exception as e:
            print(f"❌ Plot creation failed: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"Failed to create plots: {e}")

    def _on_step(self) -> bool:
        """Called after each environment step."""
        return True

    def _on_rollout_end(self) -> None:
        """Enhanced rollout end processing with plotting."""
        # Process episode info for enhanced statistics
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            try:
                # Get recent episodes
                recent_episodes = list(self.model.ep_info_buffer)

                for episode_info in recent_episodes:
                    reward = episode_info.get('r', 0)
                    length = episode_info.get('l', 0)

                    # Extract Pokemon-specific metrics
                    maps = episode_info.get('maps_visited', 1)
                    badges = episode_info.get('badges_earned', 0)
                    hp_ratio = episode_info.get('hp_ratio', 1.0)

                    # Store enhanced data
                    self.episode_rewards.append(reward)
                    self.episode_lengths.append(length)
                    self.maps_discovered.append(maps)
                    self.badges_earned.append(badges)
                    self.hp_ratios.append(hp_ratio)
                    self.timesteps_history.append(self.num_timesteps)

                # Track best performance
                if len(self.episode_rewards) > 0:
                    recent_rewards = list(self.episode_rewards)[-10:]
                    mean_reward = np.mean(recent_rewards)

                    if mean_reward > self.best_reward:
                        self.best_reward = mean_reward
                        # Save best model
                        best_model_path = os.path.join(self.save_path_models, 'best_model')
                        self.model.save(best_model_path)

                        if self.verbose >= 1:
                            logger.info(f"🏆 New best model! Reward: {mean_reward:.2f}")

                # Create and save plots periodically
                if self.show_plots and self.num_timesteps % self.plot_freq == 0:
                    print(f"📊 Creating plots at timestep {self.num_timesteps}")
                    self.create_and_save_plots()

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

        # Periodic model saving
        if self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(self.save_path_models, f'model_{self.num_timesteps}')
            self.model.save(model_path)

            if self.verbose >= 1:
                logger.info(f"💾 Checkpoint saved at step {self.num_timesteps:,}")

    def save_plots(self, filename: Optional[str] = None) -> None:
        """Save current plots to file."""
        if not self.show_plots:
            return

        try:
            if filename is None:
                filename = os.path.join(self.save_path_logs, f'final_training_plots_{self.num_timesteps}.png')

            self.create_and_save_plots()

            if self.verbose >= 1:
                logger.info(f"Training plots saved: {filename}")

        except Exception as e:
            logger.error(f"Failed to save plots: {e}")

    def cleanup(self) -> None:
        """Clean up and save final plots."""
        if self.show_plots and len(self.timesteps_history) > 0:
            try:
                print("🎯 Creating final training plots...")
                self.create_and_save_plots()

                # Create summary plot
                summary_file = os.path.join(self.save_path_logs, 'final_training_summary.png')

                fig, ax = plt.subplots(1, 1, figsize=(10, 6))

                if len(self.episode_rewards) > 0:
                    rewards = list(self.episode_rewards)
                    timesteps = list(self.timesteps_history)[-len(rewards):]

                    ax.plot(timesteps, rewards, 'b-', alpha=0.6, linewidth=1, label='Episode Rewards')

                    # Add moving average
                    if len(rewards) > 10:
                        window = min(20, len(rewards) // 3)
                        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                        moving_timesteps = timesteps[window-1:]
                        ax.plot(moving_timesteps, moving_avg, 'r-', linewidth=2,
                               label=f'Moving Average ({window} episodes)')

                    ax.set_title('Pokemon Red RL Training Summary')
                    ax.set_xlabel('Timesteps')
                    ax.set_ylabel('Episode Reward')
                    ax.grid(True, alpha=0.3)
                    ax.legend()

                    # Add final stats
                    final_stats = f"Training Summary:\n"
                    final_stats += f"Total Episodes: {len(rewards)}\n"
                    final_stats += f"Total Timesteps: {self.num_timesteps:,}\n"
                    final_stats += f"Best Reward: {max(rewards):.2f}\n"
                    final_stats += f"Final Avg (last 10): {np.mean(rewards[-10:]):.2f}\n"
                    if len(self.badges_earned) > 0:
                        final_stats += f"Max Badges: {max(self.badges_earned)}\n"
                    if len(self.maps_discovered) > 0:
                        final_stats += f"Max Maps: {max(self.maps_discovered)}"

                    ax.text(0.02, 0.98, final_stats, transform=ax.transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

                plt.tight_layout()
                plt.savefig(summary_file, dpi=150, bbox_inches='tight')
                plt.close(fig)

                print(f"🎯 Final summary saved: {summary_file}")
                print("\n📊 To view your training plots:")
                print(f"   open {self.save_path_logs}/latest_training_plots.png")
                print(f"   open {summary_file}")

            except Exception as e:
                print(f"❌ Final plot creation failed: {e}")

    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced training statistics."""
        base_stats = {
            'num_timesteps': self.num_timesteps,
            'best_reward': self.best_reward,
            'best_exploration': self.best_exploration
        }

        if len(self.episode_rewards) > 0:
            rewards = list(self.episode_rewards)
            base_stats.update({
                'total_episodes': len(rewards),
                'recent_avg_reward': np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards),
                'recent_reward_std': np.std(rewards[-10:]) if len(rewards) >= 10 else np.std(rewards)
            })

        if len(self.maps_discovered) > 0:
            recent_maps = list(self.maps_discovered)[-100:]
            base_stats.update({
                'recent_avg_maps': np.mean(recent_maps),
                'max_maps_discovered': max(self.maps_discovered),
                'recent_maps_std': np.std(recent_maps)
            })

        if len(self.badges_earned) > 0:
            recent_badges = list(self.badges_earned)[-100:]
            base_stats.update({
                'recent_avg_badges': np.mean(recent_badges),
                'max_badges_earned': max(self.badges_earned),
                'recent_badges_std': np.std(recent_badges)
            })

        if len(self.hp_ratios) > 0:
            recent_hp = list(self.hp_ratios)[-100:]
            base_stats.update({
                'recent_avg_hp_ratio': np.mean(recent_hp),
                'recent_hp_ratio_std': np.std(recent_hp)
            })

        return base_stats


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


class WandbCallback(BaseCallback):
    """
    Weights & Biases logging callback for Pokemon Red RL training.

    Logs episode-level metrics (reward, length, maps, badges, event
    flags, HP) to a W&B run so training curves are available in the
    web dashboard in real time.  Also saves model checkpoints as W&B
    artifacts for reproducibility.

    Usage::

        import wandb
        run = wandb.init(project="pokemon-red-ai", config={...})
        callback = WandbCallback(save_freq=50_000, save_path="./training")
        model.learn(callback=callback)
        run.finish()

    The callback does NOT call ``wandb.init()`` or ``wandb.finish()``
    itself — the caller owns the run lifecycle so multiple callbacks
    can share one run.
    """

    def __init__(
        self,
        save_freq: int = 50_000,
        save_path: str = "./models/",
        log_freq: int = 1,
        verbose: int = 1,
    ):
        """
        Args:
            save_freq: How often (in timesteps) to save a model
                checkpoint and upload it as a W&B artifact.
            save_path: Local directory for model checkpoints.
            log_freq: Log to W&B every *N* rollout ends (1 = every
                rollout, 2 = every other, etc.).
            verbose: Verbosity level (0 = silent, 1 = info).
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.save_path_models = os.path.join(save_path, "models")
        os.makedirs(self.save_path_models, exist_ok=True)

        self.log_freq = log_freq
        self._rollout_count = 0

        self.best_reward = -float("inf")

        # Lazy import so the rest of the module works without wandb
        try:
            import wandb as _wandb

            self._wandb = _wandb
        except ImportError:
            raise ImportError(
                "wandb is required for WandbCallback. "
                "Install with: pip install wandb"
            )

        if self.verbose >= 1:
            logger.info(
                "WandbCallback initialised "
                f"(save_freq={save_freq}, log_freq={log_freq})"
            )

    # ------------------------------------------------------------------
    # SB3 callback interface
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1

        # Skip logging on some rollouts if log_freq > 1
        if self._rollout_count % self.log_freq != 0:
            return

        metrics: Dict[str, Any] = {"global_step": self.num_timesteps}

        # ── Episode-buffer metrics ───────────────────────────────────
        if (
            hasattr(self.model, "ep_info_buffer")
            and len(self.model.ep_info_buffer) > 0
        ):
            try:
                episodes = list(self.model.ep_info_buffer)
                rewards = [e["r"] for e in episodes if "r" in e]
                lengths = [e["l"] for e in episodes if "l" in e]

                if rewards:
                    metrics["episode/reward_mean"] = float(np.mean(rewards))
                    metrics["episode/reward_max"] = float(np.max(rewards))
                    metrics["episode/reward_min"] = float(np.min(rewards))
                if lengths:
                    metrics["episode/length_mean"] = float(np.mean(lengths))

                # Pokemon-specific keys injected by the gym env's info
                # dict (via Monitor wrapper)
                maps = [e["maps_visited"] for e in episodes if "maps_visited" in e]
                badges = [e["badges_earned"] for e in episodes if "badges_earned" in e]
                locations = [
                    e["locations_visited"]
                    for e in episodes
                    if "locations_visited" in e
                ]
                hp = [e["hp_ratio"] for e in episodes if "hp_ratio" in e]

                if maps:
                    metrics["game/maps_visited_mean"] = float(np.mean(maps))
                    metrics["game/maps_visited_max"] = int(np.max(maps))
                if badges:
                    metrics["game/badges_max"] = int(np.max(badges))
                    metrics["game/badges_mean"] = float(np.mean(badges))
                if locations:
                    metrics["game/locations_mean"] = float(np.mean(locations))
                if hp:
                    metrics["game/hp_ratio_mean"] = float(np.mean(hp))

                # Event flag progress (when using EventProgressRewardCalculator)
                event_flags = [
                    e.get("event_progress", {}).get("flags_triggered", 0)
                    for e in episodes
                    if "event_progress" in e
                ]
                if event_flags:
                    metrics["game/event_flags_max"] = int(np.max(event_flags))
                    metrics["game/event_flags_mean"] = float(np.mean(event_flags))

                # Best-model tracking
                if rewards:
                    recent_mean = float(np.mean(rewards[-10:]))
                    if recent_mean > self.best_reward:
                        self.best_reward = recent_mean
                        best_path = os.path.join(
                            self.save_path_models, "best_model"
                        )
                        self.model.save(best_path)
                        if self.verbose >= 1:
                            logger.info(
                                f"W&B: new best model "
                                f"(reward={recent_mean:.2f})"
                            )
                    metrics["episode/best_reward"] = self.best_reward

            except (TypeError, KeyError, IndexError) as exc:
                if self.verbose >= 2:
                    logger.warning(f"WandbCallback ep_info parse error: {exc}")

        # ── Log to W&B ───────────────────────────────────────────────
        self._wandb.log(metrics, step=self.num_timesteps)

        # ── Periodic checkpoint + artifact ───────────────────────────
        if self.num_timesteps % self.save_freq == 0:
            ckpt_name = f"model_{self.num_timesteps}"
            ckpt_path = os.path.join(self.save_path_models, ckpt_name)
            self.model.save(ckpt_path)

            try:
                artifact = self._wandb.Artifact(
                    name=f"model-step-{self.num_timesteps}",
                    type="model",
                    description=f"Checkpoint at step {self.num_timesteps}",
                )
                artifact.add_file(ckpt_path + ".zip")
                self._wandb.log_artifact(artifact)
                if self.verbose >= 1:
                    logger.info(
                        f"W&B: checkpoint artifact uploaded "
                        f"(step {self.num_timesteps})"
                    )
            except Exception as exc:
                logger.warning(f"W&B artifact upload failed: {exc}")


# Info-dict keys the monitoring callback consumes from each env step.  These
# are the same keys ``PokemonRedGymEnv._get_info()`` populates, so the list
# doubles as the ``info_keywords`` tuple we pass to ``Monitor`` in
# ``scripts/train.py`` so they also propagate into ``ep_info_buffer``.
MONITORED_INFO_KEYS: tuple = (
    "maps_visited",
    "badges_earned",
    "locations_visited",
    "hp_ratio",
    "event_progress",
    "current_map",
    "unique_maps_list",
    "player_level",
    "pokemon_count",
    "money",
)


class MonitoringCallback(WandbCallback):
    """W&B callback with game-specific monitoring extensions.

    Extends :class:`WandbCallback` with research-oriented views that
    tell the reviewer at a glance whether a run is worth continuing:

    * **Map exploration heatmap** — how many episodes visited each map.
      Logged as a W&B Table and as a Matplotlib bar chart image so it
      survives when the Table UI is collapsed.
    * **Event flag progress** — how many episodes triggered each of the
      18 pre-registered Boulder Path flags (``game.event_flags``), and
      the earliest episode in which each flag fired.
    * **Screen captures** — the game screen captured from the training
      env every ``screen_capture_freq`` episodes, logged as a W&B image.
    * **Per-episode breakdown** — reward, length, maps visited, final
      map, badge count, and event flags triggered for every completed
      episode.  Logged as a W&B Table and mirrored to a JSON snapshot
      at ``<save_path>/dashboard_state.json`` so ``scripts/monitor.py``
      can read the same data without a W&B account.

    All logging failures are caught and logged at DEBUG level — they
    never interrupt training.
    """

    def __init__(
        self,
        save_freq: int = 50_000,
        save_path: str = "./models/",
        log_freq: int = 1,
        screen_capture_freq: int = 10,
        dashboard_state_path: Optional[str] = None,
        verbose: int = 1,
    ):
        """
        Args:
            save_freq: Checkpoint save frequency (timesteps) — see
                :class:`WandbCallback`.
            save_path: Root directory for checkpoints, dashboard state,
                and any derived artifacts.
            log_freq: W&B log throttle — only log every Nth rollout.
            screen_capture_freq: Episodes between screen-capture logs.
                Set to 0 to disable screen captures entirely.
            dashboard_state_path: Where to write ``dashboard_state.json``
                for the local Streamlit dashboard.  Defaults to
                ``<save_path>/dashboard_state.json``.
            verbose: 0 silent, 1 info, 2 debug.
        """
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            log_freq=log_freq,
            verbose=verbose,
        )
        self.screen_capture_freq = max(0, int(screen_capture_freq))
        self.dashboard_state_path = dashboard_state_path or os.path.join(
            save_path, "dashboard_state.json"
        )

        # Per-episode state
        self._episode_count: int = 0
        self._episodes_since_screen: int = 0
        self._map_visit_counts: Dict[int, int] = defaultdict(int)
        self._flag_trigger_counts: Dict[str, int] = defaultdict(int)
        self._flag_first_triggered: Dict[str, int] = {}
        self._episode_rows: List[Dict[str, Any]] = []

        # Per-step reward component accumulators (keyed by env index).
        # Each value is a defaultdict(float) that accumulates per-step
        # reward breakdown dicts over the course of an episode.  Reset
        # when the episode ends.
        self._reward_accumulators: Dict[int, Dict[str, float]] = {}

        # Per-episode game-state histories for W&B charts
        self._level_history: List[int] = []
        self._pokemon_count_history: List[int] = []
        self._money_history: List[int] = []
        self._episode_reward_components: List[Dict[str, float]] = []

        os.makedirs(os.path.dirname(os.path.abspath(self.dashboard_state_path)), exist_ok=True)

        # Define W&B metric grouping so panels auto-organise
        self._define_wandb_metrics()

        if self.verbose >= 1:
            logger.info(
                "MonitoringCallback initialised "
                f"(screen_capture_freq={self.screen_capture_freq} episodes, "
                f"dashboard_state={self.dashboard_state_path})"
            )

    # ------------------------------------------------------------------
    # W&B metric grouping
    # ------------------------------------------------------------------

    def _define_wandb_metrics(self) -> None:
        """Set ``wandb.define_metric`` for automatic panel organisation.

        Groups metrics under ``episode/``, ``game/``, and ``reward/``
        sections so the W&B dashboard auto-creates useful panels without
        manual configuration.  Failures are swallowed silently because
        older wandb versions may not support ``define_metric``.
        """
        try:
            w = self._wandb
            w.define_metric("global_step")
            # Episode scalars plotted against global_step
            w.define_metric("episode/*", step_metric="global_step")
            # Game-state metrics
            w.define_metric("game/*", step_metric="global_step")
            # Reward component breakdown
            w.define_metric("reward/*", step_metric="global_step")
        except Exception:
            pass  # define_metric is best-effort

    # ------------------------------------------------------------------
    # SB3 callback interface
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        """Accumulate per-step reward components and record episodes."""
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True

        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            # Accumulate reward components every step (cheap dict adds)
            components = info.get("reward_components")
            if components and isinstance(components, dict):
                if env_idx not in self._reward_accumulators:
                    self._reward_accumulators[env_idx] = defaultdict(float)
                for key, val in components.items():
                    try:
                        self._reward_accumulators[env_idx][key] += float(val)
                    except (TypeError, ValueError):
                        continue

            if done:
                # Snapshot the accumulated reward components for this
                # episode, then clear the accumulator.
                accumulated = dict(self._reward_accumulators.pop(env_idx, {}))
                self._record_episode(info, reward_components=accumulated)

        return True

    def _on_rollout_end(self) -> None:
        """Base class handles aggregate metrics; we add the extras."""
        super()._on_rollout_end()

        # Throttle extras at the same rate as base metrics
        if self._rollout_count % self.log_freq != 0:
            return

        self._log_extras()
        self._write_dashboard_state()

    # ------------------------------------------------------------------
    # Episode recording
    # ------------------------------------------------------------------

    def _record_episode(
        self,
        info: Dict[str, Any],
        reward_components: Optional[Dict[str, float]] = None,
    ) -> None:
        self._episode_count += 1
        self._episodes_since_screen += 1

        ep_info = info.get("episode") or {}
        reward = float(ep_info.get("r", 0.0))
        length = int(ep_info.get("l", 0))

        # Custom keys from PokemonRedGymEnv._get_info()
        unique_maps = info.get("unique_maps_list") or []
        event_progress = info.get("event_progress") or {}
        final_map = int(info.get("current_map", 0) or 0)
        badges = int(info.get("badges_earned", 0) or 0)
        locations = int(info.get("locations_visited", 0) or 0)
        triggered_names = list(event_progress.get("triggered_names", []) or [])
        flags_triggered = int(event_progress.get("flags_triggered", len(triggered_names)))

        # Game-state fields for level/money/party tracking
        player_level = int(info.get("player_level", 0) or 0)
        pokemon_count = int(info.get("pokemon_count", 0) or 0)
        money = int(info.get("money", 0) or 0)

        # Update cumulative counts
        for map_id in unique_maps:
            try:
                self._map_visit_counts[int(map_id)] += 1
            except (TypeError, ValueError):
                continue

        for flag_name in triggered_names:
            self._flag_trigger_counts[flag_name] += 1
            self._flag_first_triggered.setdefault(flag_name, self._episode_count)

        # Store game-state histories
        self._level_history.append(player_level)
        self._pokemon_count_history.append(pokemon_count)
        self._money_history.append(money)
        if reward_components:
            self._episode_reward_components.append(reward_components)

        self._episode_rows.append(
            {
                "episode": self._episode_count,
                "global_step": int(self.num_timesteps),
                "reward": reward,
                "length": length,
                "maps_visited": len(unique_maps),
                "final_map": final_map,
                "badges": badges,
                "event_flags_triggered": flags_triggered,
                "locations_visited": locations,
                "triggered_flags": triggered_names,
                "player_level": player_level,
                "pokemon_count": pokemon_count,
                "money": money,
                "reward_components": reward_components or {},
            }
        )

        # Screen capture on schedule (at episode boundary)
        if (
            self.screen_capture_freq > 0
            and self._episodes_since_screen >= self.screen_capture_freq
        ):
            self._log_screen_capture()
            self._episodes_since_screen = 0

    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------

    def _log_screen_capture(self) -> None:
        """Capture the current game screen and log as a W&B image."""
        screen = self._get_env_screen()
        if screen is None:
            return

        try:
            img = self._wandb.Image(
                screen,
                caption=f"episode {self._episode_count} | step {self.num_timesteps}",
            )
            self._wandb.log(
                {"game/screen": img},
                step=self.num_timesteps,
            )
        except Exception as exc:
            logger.debug(f"W&B screen log failed: {exc}")

    def _get_env_screen(self) -> Optional[np.ndarray]:
        """Fetch the first env's screen as (H, W, 3) uint8, or None."""
        env = getattr(self, "training_env", None)
        if env is None:
            return None

        try:
            screens = env.env_method("render", "rgb_array", indices=[0])
        except Exception:
            try:
                screens = env.env_method("render", indices=[0])
            except Exception as exc:
                logger.debug(f"env_method('render') failed: {exc}")
                return None

        if not screens:
            return None
        arr = screens[0]
        if arr is None or not isinstance(arr, np.ndarray):
            return None
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    # ------------------------------------------------------------------
    # Extras logging
    # ------------------------------------------------------------------

    def _log_extras(self) -> None:
        """Log heatmap, flag progress, reward breakdown, and more to W&B."""
        try:
            self._log_map_heatmap()
        except Exception as exc:
            logger.debug(f"Map heatmap log failed: {exc}")

        try:
            self._log_flag_progress()
        except Exception as exc:
            logger.debug(f"Flag progress log failed: {exc}")

        try:
            self._log_episode_table()
        except Exception as exc:
            logger.debug(f"Episode table log failed: {exc}")

        try:
            self._log_reward_breakdown()
        except Exception as exc:
            logger.debug(f"Reward breakdown log failed: {exc}")

        try:
            self._log_game_state_metrics()
        except Exception as exc:
            logger.debug(f"Game state metrics log failed: {exc}")

    def _log_map_heatmap(self) -> None:
        if not self._map_visit_counts:
            return

        rows = sorted(self._map_visit_counts.items())
        table = self._wandb.Table(columns=["map_id", "visit_count"])
        for map_id, count in rows:
            table.add_data(int(map_id), int(count))

        payload: Dict[str, Any] = {"game/map_heatmap_table": table}

        chart = self._render_map_heatmap_chart(rows)
        if chart is not None:
            payload["game/map_heatmap"] = self._wandb.Image(
                chart,
                caption=f"Map visit counts (cumulative, step {self.num_timesteps})",
            )

        self._wandb.log(payload, step=self.num_timesteps)

    @staticmethod
    def _render_map_heatmap_chart(rows: List[tuple]) -> Optional[np.ndarray]:
        """Render a simple horizontal-bar chart of map visit counts."""
        if not rows:
            return None

        try:
            map_ids = [str(r[0]) for r in rows]
            counts = [r[1] for r in rows]

            fig, ax = plt.subplots(figsize=(6, max(2.5, 0.25 * len(rows))))
            ax.barh(map_ids, counts, color="#3b82f6")
            ax.set_xlabel("Episodes that visited map")
            ax.set_ylabel("map_id")
            ax.set_title("Map exploration heatmap")
            ax.invert_yaxis()
            fig.tight_layout()

            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            plt.close(fig)
            return img
        except Exception as exc:
            logger.debug(f"Heatmap render failed: {exc}")
            return None

    def _log_flag_progress(self) -> None:
        try:
            from ..game.event_flags import BOULDER_PATH_FLAGS
        except Exception as exc:
            logger.debug(f"Cannot import BOULDER_PATH_FLAGS: {exc}")
            return

        table = self._wandb.Table(
            columns=["flag_name", "trigger_count", "first_triggered_episode"]
        )
        for flag_name in BOULDER_PATH_FLAGS:
            count = int(self._flag_trigger_counts.get(flag_name, 0))
            first = self._flag_first_triggered.get(flag_name, -1)
            table.add_data(flag_name, count, int(first))

        metrics: Dict[str, Any] = {"game/flag_progress": table}

        total_flags = len(BOULDER_PATH_FLAGS)
        unique_flags_ever = sum(
            1 for n in BOULDER_PATH_FLAGS if self._flag_trigger_counts.get(n, 0) > 0
        )
        metrics["game/unique_flags_ever"] = unique_flags_ever
        metrics["game/unique_flags_fraction"] = (
            unique_flags_ever / total_flags if total_flags else 0.0
        )

        self._wandb.log(metrics, step=self.num_timesteps)

    def _log_episode_table(self) -> None:
        if not self._episode_rows:
            return

        table = self._wandb.Table(
            columns=[
                "episode",
                "global_step",
                "reward",
                "length",
                "maps_visited",
                "final_map",
                "badges",
                "event_flags_triggered",
                "locations_visited",
                "player_level",
                "pokemon_count",
                "money",
            ]
        )
        for row in self._episode_rows[-500:]:  # cap to avoid huge payloads
            table.add_data(
                row["episode"],
                row["global_step"],
                row["reward"],
                row["length"],
                row["maps_visited"],
                row["final_map"],
                row["badges"],
                row["event_flags_triggered"],
                row["locations_visited"],
                row.get("player_level", 0),
                row.get("pokemon_count", 0),
                row.get("money", 0),
            )
        self._wandb.log(
            {"game/episodes": table},
            step=self.num_timesteps,
        )

    # ------------------------------------------------------------------
    # Reward breakdown & game-state logging
    # ------------------------------------------------------------------

    def _log_reward_breakdown(self) -> None:
        """Log per-component reward scalars averaged over recent episodes.

        Produces W&B scalar lines like ``reward/exploration``,
        ``reward/badge``, ``reward/event_flags``, etc. that let
        researchers see which components the agent is exploiting.
        """
        recent = self._episode_reward_components[-50:]
        if not recent:
            return

        # Collect all component keys across recent episodes
        all_keys: set = set()
        for rc in recent:
            all_keys.update(rc.keys())

        metrics: Dict[str, Any] = {}
        for key in sorted(all_keys):
            values = [rc.get(key, 0.0) for rc in recent]
            metrics[f"reward/{key}_mean"] = float(np.mean(values))
            metrics[f"reward/{key}_max"] = float(np.max(values))

        # Total reward from components (sanity check)
        totals = [sum(rc.values()) for rc in recent]
        metrics["reward/total_components_mean"] = float(np.mean(totals))

        self._wandb.log(metrics, step=self.num_timesteps)

    def _log_game_state_metrics(self) -> None:
        """Log level curves, party size, and money as W&B scalars.

        These provide the \"game progress\" view: is the agent levelling
        up, catching Pokemon, and accumulating money over training?
        """
        metrics: Dict[str, Any] = {}

        if self._level_history:
            recent_levels = self._level_history[-50:]
            metrics["game/player_level_mean"] = float(np.mean(recent_levels))
            metrics["game/player_level_max"] = int(np.max(recent_levels))
            metrics["game/player_level_latest"] = recent_levels[-1]

        if self._pokemon_count_history:
            recent_counts = self._pokemon_count_history[-50:]
            metrics["game/pokemon_count_mean"] = float(np.mean(recent_counts))
            metrics["game/pokemon_count_max"] = int(np.max(recent_counts))

        if self._money_history:
            recent_money = self._money_history[-50:]
            metrics["game/money_mean"] = float(np.mean(recent_money))
            metrics["game/money_max"] = int(np.max(recent_money))

        if metrics:
            self._wandb.log(metrics, step=self.num_timesteps)

    # ------------------------------------------------------------------
    # Dashboard JSON snapshot
    # ------------------------------------------------------------------

    def _write_dashboard_state(self) -> None:
        """Mirror monitoring state to a JSON file for local dashboards.

        Overwrites the file atomically so readers never see a partial
        write.  Failures are swallowed (logged at DEBUG) — the snapshot
        is best-effort.
        """
        try:
            from ..game.event_flags import BOULDER_PATH_FLAGS
            flag_names = list(BOULDER_PATH_FLAGS)
        except Exception:
            flag_names = list(self._flag_trigger_counts)

        # Aggregate recent reward components for the dashboard
        recent_rc = self._episode_reward_components[-50:]
        rc_summary: Dict[str, float] = {}
        if recent_rc:
            all_keys: set = set()
            for rc in recent_rc:
                all_keys.update(rc.keys())
            for key in sorted(all_keys):
                vals = [rc.get(key, 0.0) for rc in recent_rc]
                rc_summary[key] = float(np.mean(vals))

        snapshot = {
            "num_timesteps": int(self.num_timesteps),
            "episode_count": self._episode_count,
            "best_reward": (
                self.best_reward if self.best_reward != -float("inf") else None
            ),
            "map_visit_counts": {
                str(k): int(v) for k, v in self._map_visit_counts.items()
            },
            "flag_trigger_counts": {
                name: int(self._flag_trigger_counts.get(name, 0))
                for name in flag_names
            },
            "flag_first_triggered": {
                name: int(self._flag_first_triggered.get(name, -1))
                for name in flag_names
            },
            "episodes": self._episode_rows[-200:],
            # Enhanced metrics
            "level_history": self._level_history[-200:],
            "pokemon_count_history": self._pokemon_count_history[-200:],
            "money_history": self._money_history[-200:],
            "reward_component_summary": rc_summary,
        }

        try:
            tmp_path = self.dashboard_state_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, indent=2)
            os.replace(tmp_path, self.dashboard_state_path)
        except Exception as exc:
            logger.debug(f"Dashboard state write failed: {exc}")