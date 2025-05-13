"""
ASCII-ME Emitter Implementation for Quality Diversity.

This module implements the ASCII-ME (Action Sequence Crossover with
performance-Informed Interpolation for MAP-Elites) emitter, which combines
genetic algorithm (GA) emitters with policy gradient emitters for quality diversity
optimization. This hybrid approach allows balancing between exploration (via GA)
and exploitation (via policy gradients).
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import flax.linen as nn

from qdax.core.emitters.ascii_emitter import ASCIIConfig, ASCIIEmitter
from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.custom_types import Params, RNGKey
from qdax.environments.base_wrappers import QDEnv


@dataclass
class ASCIIMEConfig:
    """Configuration for ASCII-ME Algorithm.

    This class defines the parameters for the ASCII-ME hybrid emitter,
    controlling both the GA and policy gradient components.

    Attributes:
        proportion_mutation_ga: Fraction of agents using genetic algorithm mutation.
        no_agents: Total number of agents/policies to maintain.
        buffer_sample_batch_size: Size of batches sampled from replay buffer.
        no_epochs: Number of optimization epochs per iteration.
        learning_rate: Learning rate for policy gradient updates.
        clip_param: Clipping parameter for PPO-style updates.
        discount_rate: Discount factor for reward calculation.
        cosine_similarity: Whether to use cosine similarity for behavior comparison.
        std: Standard deviation for policy exploration.
    """

    proportion_mutation_ga: float = 0.5
    no_agents: int = 512
    buffer_sample_batch_size: int = 2
    no_epochs: int = 16
    learning_rate: float = 3e-4
    clip_param: float = 0.2
    discount_rate: float = 0.99
    cosine_similarity: bool = True
    std: float = 0.5


class ASCIIMEEmitter(MultiEmitter):
    """ASCII-ME Emitter that combines GA mutation and policy gradients.

    This emitter implements the hybrid ASCII-ME approach, which dynamically balances
    between genetic algorithm variation (for exploration) and policy gradient learning
    (for exploitation) based on the configured proportion.
    """

    def __init__(
        self,
        config: ASCIIMEConfig,
        policy_network: nn.Module,
        env: QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:
        """Initialize the ASCII-ME emitter.

        Args:
            config: Configuration parameters for the emitter.
            policy_network: Neural network model used for policies.
            env: Environment wrapper for QD algorithms.
            variation_fn: Function for genetic variation operations.
        """
        # Store configuration and components
        self._config = config
        self._policy_network = policy_network
        self._env = env
        self._variation_fn = variation_fn

        # Calculate number of agents for each approach based on proportion
        ga_no_agents = int(self._config.proportion_mutation_ga * config.no_agents)
        mcpg_no_agents = config.no_agents - ga_no_agents

        # Handle three cases:
        # 1. Pure GA approach (proportion_mutation_ga = 1.0)
        if mcpg_no_agents == 0:
            ga_emitter = MixingEmitter(
                mutation_fn=None,
                variation_fn=variation_fn,
                variation_percentage=1.0,
                batch_size=ga_no_agents,
            )
            super().__init__(emitters=(ga_emitter,))

        # 2. Pure policy gradient approach (proportion_mutation_ga = 0.0)
        elif ga_no_agents == 0:
            ascii_config = ASCIIConfig(
                no_agents=mcpg_no_agents,
                buffer_sample_batch_size=config.buffer_sample_batch_size,
                buffer_add_batch_size=config.no_agents,
                no_epochs=config.no_epochs,
                learning_rate=config.learning_rate,
                discount_rate=config.discount_rate,
                clip_param=config.clip_param,
                std=config.std,
            )

            ascii_emitter = ASCIIEmitter(
                config=ascii_config, policy_net=policy_network, env=env
            )

            super().__init__(emitters=(ascii_emitter,))

        # 3. Hybrid approach combining both emitters
        else:
            # Configure the ASCII component (policy gradient)
            ascii_config = ASCIIConfig(
                no_agents=mcpg_no_agents,
                buffer_sample_batch_size=config.buffer_sample_batch_size,
                buffer_add_batch_size=config.no_agents,
                no_epochs=config.no_epochs,
                learning_rate=config.learning_rate,
                discount_rate=config.discount_rate,
                clip_param=config.clip_param,
                std=config.std,
            )

            # Configure the GA component
            ga_emitter = MixingEmitter(
                mutation_fn=None,
                variation_fn=variation_fn,
                variation_percentage=1.0,
                batch_size=ga_no_agents,
            )

            # Create the ASCII emitter
            ascii_emitter = ASCIIEmitter(
                config=ascii_config, policy_net=policy_network, env=env
            )

            # Initialize the multi-emitter with both components
            super().__init__(emitters=(ascii_emitter, ga_emitter))
