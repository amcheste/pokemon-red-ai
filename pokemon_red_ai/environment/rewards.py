"""
Reward calculation system for Pokemon Red RL environment.

This module provides flexible reward calculation with different reward
strategies and component tracking for analysis.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    # Base rewards
    time_penalty: float = -0.01
    exploration_reward: float = 1.0
    new_map_reward: float = 20.0

    # Progress rewards
    level_reward_multiplier: float = 50.0
    badge_reward_multiplier: float = 200.0
    pokemon_reward_multiplier: float = 100.0

    # Health penalties
    low_health_threshold: float = 0.5
    health_penalty_multiplier: float = 10.0
    death_penalty: float = -100.0

    # Money rewards
    money_reward_multiplier: float = 0.01

    # Optional advanced rewards
    battle_victory_reward: float = 10.0
    item_acquisition_reward: float = 5.0
    story_progress_reward: float = 50.0


class BaseRewardCalculator(ABC):
    """Abstract base class for reward calculators."""

    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.reward_components = defaultdict(float)
        self.previous_state: Optional[Dict[str, Any]] = None
        self.visited_locations: Set[Tuple[int, int, int]] = set()
        self.visited_maps: Set[int] = set()

    @abstractmethod
    def calculate_reward(self, current_state: Dict[str, Any]) -> float:
        """Calculate reward for current state."""
        pass

    def reset(self) -> None:
        """Reset calculator state for new episode."""
        self.reward_components.clear()
        self.previous_state = None
        self.visited_locations.clear()
        self.visited_maps.clear()

    def get_reward_breakdown(self) -> Dict[str, float]:
        """Get breakdown of reward components."""
        return dict(self.reward_components)


class StandardRewardCalculator(BaseRewardCalculator):
    """
    Standard reward calculator based on the original implementation.

    Focuses on exploration, progress, and survival.
    """

    def calculate_reward(self, current_state: Dict[str, Any]) -> float:
        """Calculate reward using standard Pokemon Red RL strategy."""
        reward = 0.0
        self.reward_components.clear()

        position = current_state['position']
        stats = current_state['stats']

        # Base time penalty (encourages efficiency)
        time_penalty = self.config.time_penalty
        reward += time_penalty
        self.reward_components['time'] = time_penalty

        # Exploration rewards
        location_key = (position['x'], position['y'], position['map'])
        if location_key not in self.visited_locations:
            self.visited_locations.add(location_key)
            exploration_reward = self.config.exploration_reward
            reward += exploration_reward
            self.reward_components['exploration'] = exploration_reward

        # New map discovery
        if (position['map'] not in self.visited_maps and
                position['map'] != 0):
            self.visited_maps.add(position['map'])
            map_reward = self.config.new_map_reward
            reward += map_reward
            self.reward_components['new_map'] = map_reward

        # Progress rewards (compare with previous state)
        if self.previous_state:
            prev_stats = self.previous_state['stats']

            # Level progression
            level_diff = stats['level'] - prev_stats['level']
            if level_diff > 0:
                level_reward = level_diff * self.config.level_reward_multiplier
                reward += level_reward
                self.reward_components['level'] = level_reward

            # Badge progression (major milestone)
            badge_diff = stats['badges'] - prev_stats['badges']
            if badge_diff > 0:
                badge_reward = badge_diff * self.config.badge_reward_multiplier
                reward += badge_reward
                self.reward_components['badge'] = badge_reward

            # Pokemon acquisition
            party_diff = stats['party_count'] - prev_stats['party_count']
            if party_diff > 0:
                pokemon_reward = party_diff * self.config.pokemon_reward_multiplier
                reward += pokemon_reward
                self.reward_components['pokemon'] = pokemon_reward

        # Health penalty (losing HP is bad)
        if stats['max_hp'] > 0:
            hp_ratio = stats['current_hp'] / stats['max_hp']
            if hp_ratio < self.config.low_health_threshold:
                health_penalty = (-self.config.health_penalty_multiplier *
                                  (self.config.low_health_threshold - hp_ratio))
                reward += health_penalty
                self.reward_components['health'] = health_penalty

        # Death penalty (Pokemon fainting)
        if stats['current_hp'] == 0 and stats['max_hp'] > 0:
            death_penalty = self.config.death_penalty
            reward += death_penalty
            self.reward_components['death'] = death_penalty

        # Store current state for next comparison
        self.previous_state = {
            'position': position.copy(),
            'stats': stats.copy()
        }

        return reward


class ExplorationFocusedCalculator(BaseRewardCalculator):
    """
    Reward calculator that heavily emphasizes exploration and map coverage.

    Good for training agents to explore the world thoroughly.
    """

    def calculate_reward(self, current_state: Dict[str, Any]) -> float:
        """Calculate reward with heavy exploration focus."""
        reward = 0.0
        self.reward_components.clear()

        position = current_state['position']
        stats = current_state['stats']

        # Smaller time penalty to encourage exploration
        time_penalty = self.config.time_penalty * 0.5
        reward += time_penalty
        self.reward_components['time'] = time_penalty

        # Much higher exploration rewards
        location_key = (position['x'], position['y'], position['map'])
        if location_key not in self.visited_locations:
            self.visited_locations.add(location_key)
            exploration_reward = self.config.exploration_reward * 3.0
            reward += exploration_reward
            self.reward_components['exploration'] = exploration_reward

        # Massive bonus for new maps
        if (position['map'] not in self.visited_maps and
                position['map'] != 0):
            self.visited_maps.add(position['map'])
            map_reward = self.config.new_map_reward * 2.0
            reward += map_reward
            self.reward_components['new_map'] = map_reward

        # Bonus for exploration coverage
        map_coverage_bonus = len(self.visited_maps) * 5.0
        reward += map_coverage_bonus
        self.reward_components['coverage'] = map_coverage_bonus

        # Standard progress rewards but smaller
        if self.previous_state:
            prev_stats = self.previous_state['stats']

            level_diff = stats['level'] - prev_stats['level']
            if level_diff > 0:
                level_reward = level_diff * self.config.level_reward_multiplier * 0.5
                reward += level_reward
                self.reward_components['level'] = level_reward

            badge_diff = stats['badges'] - prev_stats['badges']
            if badge_diff > 0:
                badge_reward = badge_diff * self.config.badge_reward_multiplier
                reward += badge_reward
                self.reward_components['badge'] = badge_reward

        # Reduced health penalties to encourage risk-taking for exploration
        if stats['current_hp'] == 0 and stats['max_hp'] > 0:
            death_penalty = self.config.death_penalty * 0.5
            reward += death_penalty
            self.reward_components['death'] = death_penalty

        self.previous_state = {
            'position': position.copy(),
            'stats': stats.copy()
        }

        return reward


class ProgressFocusedCalculator(BaseRewardCalculator):
    """
    Reward calculator focused on game progression and achievements.

    Good for training agents to complete the main storyline efficiently.
    """

    def calculate_reward(self, current_state: Dict[str, Any]) -> float:
        """Calculate reward with focus on game progression."""
        reward = 0.0
        self.reward_components.clear()

        position = current_state['position']
        stats = current_state['stats']

        # Higher time penalty to encourage efficiency
        time_penalty = self.config.time_penalty * 2.0
        reward += time_penalty
        self.reward_components['time'] = time_penalty

        # Moderate exploration rewards
        location_key = (position['x'], position['y'], position['map'])
        if location_key not in self.visited_locations:
            self.visited_locations.add(location_key)
            exploration_reward = self.config.exploration_reward * 0.5
            reward += exploration_reward
            self.reward_components['exploration'] = exploration_reward

        # Progress rewards are much higher
        if self.previous_state:
            prev_stats = self.previous_state['stats']

            # Major level rewards
            level_diff = stats['level'] - prev_stats['level']
            if level_diff > 0:
                level_reward = level_diff * self.config.level_reward_multiplier * 2.0
                reward += level_reward
                self.reward_components['level'] = level_reward

            # Huge badge rewards
            badge_diff = stats['badges'] - prev_stats['badges']
            if badge_diff > 0:
                badge_reward = badge_diff * self.config.badge_reward_multiplier * 3.0
                reward += badge_reward
                self.reward_components['badge'] = badge_reward

            # Pokemon acquisition important for progression
            party_diff = stats['party_count'] - prev_stats['party_count']
            if party_diff > 0:
                pokemon_reward = party_diff * self.config.pokemon_reward_multiplier * 1.5
                reward += pokemon_reward
                self.reward_components['pokemon'] = pokemon_reward

        # Harsh penalties for poor health management
        if stats['max_hp'] > 0:
            hp_ratio = stats['current_hp'] / stats['max_hp']
            if hp_ratio < self.config.low_health_threshold:
                health_penalty = (-self.config.health_penalty_multiplier * 2.0 *
                                  (self.config.low_health_threshold - hp_ratio))
                reward += health_penalty
                self.reward_components['health'] = health_penalty

        # Major death penalty
        if stats['current_hp'] == 0 and stats['max_hp'] > 0:
            death_penalty = self.config.death_penalty * 2.0
            reward += death_penalty
            self.reward_components['death'] = death_penalty

        self.previous_state = {
            'position': position.copy(),
            'stats': stats.copy()
        }

        return reward


class SparseRewardCalculator(BaseRewardCalculator):
    """
    Sparse reward calculator that only gives rewards for major achievements.

    Good for advanced RL algorithms that can handle sparse rewards.
    """

    def calculate_reward(self, current_state: Dict[str, Any]) -> float:
        """Calculate sparse rewards only for major achievements."""
        reward = 0.0
        self.reward_components.clear()

        position = current_state['position']
        stats = current_state['stats']

        # No time penalty in sparse rewards

        # Only reward major map discoveries
        if (position['map'] not in self.visited_maps and
                position['map'] != 0 and
                position['map'] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):  # Major cities only
            self.visited_maps.add(position['map'])
            map_reward = self.config.new_map_reward * 2.0
            reward += map_reward
            self.reward_components['new_map'] = map_reward

        # Only major progress rewards
        if self.previous_state:
            prev_stats = self.previous_state['stats']

            # Only reward every 5 levels
            level_diff = stats['level'] - prev_stats['level']
            if level_diff > 0 and stats['level'] % 5 == 0:
                level_reward = self.config.level_reward_multiplier * 5.0
                reward += level_reward
                self.reward_components['level'] = level_reward

            # Badges are the main reward
            badge_diff = stats['badges'] - prev_stats['badges']
            if badge_diff > 0:
                badge_reward = badge_diff * self.config.badge_reward_multiplier * 5.0
                reward += badge_reward
                self.reward_components['badge'] = badge_reward

        # Only death penalty, no health management rewards
        if stats['current_hp'] == 0 and stats['max_hp'] > 0:
            death_penalty = self.config.death_penalty
            reward += death_penalty
            self.reward_components['death'] = death_penalty

        self.previous_state = {
            'position': position.copy(),
            'stats': stats.copy()
        }

        return reward


def create_reward_calculator(strategy: str = "standard",
                             config: RewardConfig = None) -> BaseRewardCalculator:
    """
    Factory function to create reward calculators.

    Args:
        strategy: Reward strategy ('standard', 'exploration', 'progress', 'sparse')
        config: Optional reward configuration

    Returns:
        Appropriate reward calculator instance
    """
    calculators = {
        'standard': StandardRewardCalculator,
        'exploration': ExplorationFocusedCalculator,
        'progress': ProgressFocusedCalculator,
        'sparse': SparseRewardCalculator
    }

    if strategy not in calculators:
        logger.warning(f"Unknown reward strategy: {strategy}. Using 'standard'.")
        strategy = 'standard'

    return calculators[strategy](config)


def evaluate_reward_strategy(calculator: BaseRewardCalculator,
                             episode_states: list) -> Dict[str, Any]:
    """
    Evaluate a reward strategy on a sequence of episode states.

    Args:
        calculator: Reward calculator to evaluate
        episode_states: List of game states from an episode

    Returns:
        Dictionary with evaluation metrics
    """
    calculator.reset()
    total_reward = 0.0
    reward_history = []
    component_totals = defaultdict(float)

    for state in episode_states:
        reward = calculator.calculate_reward(state)
        total_reward += reward
        reward_history.append(reward)

        # Accumulate component totals
        for component, value in calculator.get_reward_breakdown().items():
            component_totals[component] += value

    return {
        'total_reward': total_reward,
        'mean_reward': np.mean(reward_history) if reward_history else 0.0,
        'reward_std': np.std(reward_history) if reward_history else 0.0,
        'reward_history': reward_history,
        'component_totals': dict(component_totals),
        'final_exploration': len(calculator.visited_locations),
        'maps_discovered': len(calculator.visited_maps)
    }