"""
Reward calculation system for Pokemon Red RL environment.

This module provides flexible reward calculation with different reward
strategies and component tracking for analysis.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    # Base rewards (Enhanced for better exploration)
    time_penalty: float = -0.001          # Reduced from -0.01
    exploration_reward: float = 5.0       # Increased from 1.0
    new_map_reward: float = 100.0         # Increased from 20.0

    # Progress rewards (More balanced)
    level_reward_multiplier: float = 25.0      # Reduced from 50.0
    badge_reward_multiplier: float = 150.0     # Reduced from 200.0
    pokemon_reward_multiplier: float = 75.0    # Reduced from 100.0

    # Health penalties
    low_health_threshold: float = 0.3     # Reduced from 0.5
    health_penalty_multiplier: float = 5.0     # Reduced from 10.0
    death_penalty: float = -50.0         # Reduced from -100.0

    # Money rewards
    money_reward_multiplier: float = 0.005      # Reduced from 0.01

    # Optional advanced rewards
    battle_victory_reward: float = 15.0         # Increased from 10.0
    item_acquisition_reward: float = 8.0        # Increased from 5.0
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
    Enhanced reward calculator that heavily emphasizes exploration.

    Includes anti-stuck mechanisms and adaptive rewards.
    """

    def __init__(self, config: RewardConfig = None):
        super().__init__(config)

        # Enhanced tracking for better exploration
        self.position_history = deque(maxlen=100)
        self.location_visit_counts = defaultdict(int)
        self.consecutive_new_locations = 0
        self.steps_since_exploration = 0
        self.stuck_threshold = 50

    def calculate_reward(self, current_state: Dict[str, Any]) -> float:
        """Calculate reward with heavy exploration focus and anti-stuck mechanisms."""
        reward = 0.0
        self.reward_components.clear()

        position = current_state['position']
        stats = current_state['stats']

        # Adaptive time penalty based on exploration activity
        if self.steps_since_exploration < self.stuck_threshold:
            time_penalty = self.config.time_penalty * 0.1  # Very low when exploring
        else:
            time_penalty = self.config.time_penalty * 2.0  # Higher when stuck

        reward += time_penalty
        self.reward_components['time'] = time_penalty

        # Much higher exploration rewards with bonuses
        location_key = (position['x'], position['y'], position['map'])
        if location_key not in self.visited_locations:
            self.visited_locations.add(location_key)
            self.steps_since_exploration = 0
            self.consecutive_new_locations += 1

            # Base exploration reward (5x higher than standard)
            exploration_reward = self.config.exploration_reward

            # Consecutive exploration bonus
            if self.consecutive_new_locations > 1:
                exploration_reward *= (1.0 + 0.1 * self.consecutive_new_locations)

            # Distance bonus for going far from starting area
            if len(self.position_history) > 0:
                start_pos = self.position_history[0] if self.position_history else (0, 0, 1)
                distance = abs(position['x'] - start_pos[0]) + abs(position['y'] - start_pos[1])
                distance_bonus = min(distance * 0.1, 10.0)
                exploration_reward += distance_bonus
                self.reward_components['distance_bonus'] = distance_bonus

            reward += exploration_reward
            self.reward_components['exploration'] = exploration_reward
        else:
            # Small penalty for frequent revisits
            self.location_visit_counts[location_key] += 1
            if self.location_visit_counts[location_key] > 5:
                revisit_penalty = -0.5
                reward += revisit_penalty
                self.reward_components['revisit_penalty'] = revisit_penalty

            self.consecutive_new_locations = 0
            self.steps_since_exploration += 1

        # Massive bonus for new maps with progressive scaling
        if (position['map'] not in self.visited_maps and
                position['map'] != 0):
            self.visited_maps.add(position['map'])

            # Base map reward (5x higher)
            map_reward = self.config.new_map_reward

            # Progressive bonus for discovering more maps
            num_maps = len(self.visited_maps)
            if num_maps > 1:
                map_reward *= (1.0 + 0.3 * (num_maps - 1))

            reward += map_reward
            self.reward_components['new_map'] = map_reward

            # Milestone rewards
            if num_maps == 3:
                milestone_reward = 200.0
                reward += milestone_reward
                self.reward_components['milestone_3_maps'] = milestone_reward
            elif num_maps == 5:
                milestone_reward = 500.0
                reward += milestone_reward
                self.reward_components['milestone_5_maps'] = milestone_reward

        # Map diversity bonus
        if len(self.visited_maps) > 1:
            diversity_bonus = len(self.visited_maps) * 3.0
            reward += diversity_bonus
            self.reward_components['diversity'] = diversity_bonus

        # Reduced progress rewards to maintain exploration focus
        if self.previous_state:
            prev_stats = self.previous_state['stats']

            level_diff = stats['level'] - prev_stats['level']
            if level_diff > 0:
                level_reward = level_diff * self.config.level_reward_multiplier * 0.3
                reward += level_reward
                self.reward_components['level'] = level_reward

            badge_diff = stats['badges'] - prev_stats['badges']
            if badge_diff > 0:
                badge_reward = badge_diff * self.config.badge_reward_multiplier * 0.8
                reward += badge_reward
                self.reward_components['badge'] = badge_reward

                # Reset exploration tracking on major progress
                self.steps_since_exploration = 0

        # Reduced health penalties to encourage risk-taking
        if stats['current_hp'] == 0 and stats['max_hp'] > 0:
            death_penalty = self.config.death_penalty * 0.3
            reward += death_penalty
            self.reward_components['death'] = death_penalty

        # Anti-stuck penalty
        if self.steps_since_exploration > self.stuck_threshold:
            stuck_penalty = -2.0 * (self.steps_since_exploration - self.stuck_threshold) / 10.0
            reward += stuck_penalty
            self.reward_components['stuck_penalty'] = stuck_penalty

        # Update tracking
        self.position_history.append((position['x'], position['y'], position['map']))

        self.previous_state = {
            'position': position.copy(),
            'stats': stats.copy()
        }

        return reward

    def reset(self) -> None:
        """Reset calculator state for new episode."""
        super().reset()
        self.position_history.clear()
        self.location_visit_counts.clear()
        self.consecutive_new_locations = 0
        self.steps_since_exploration = 0


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