"""
Action Sequence Crossover with performance-Informed Interpolation (ASCII)
Emitter for Quality-Diversity optimization.

This module implements the ASCII emitter for Quality-Diversity optimization.
It uses policy gradient techniques to optimize policies within a MAP-Elites
framework. The emitter maintains a trajectory buffer and uses a policy
optimization approach to improve sampled policies.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple

import flashbax as fbx
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import ArrayTree
from utils import compute_cosine_similarity

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from qdax.environments.base_wrappers import QDEnv

# Constants
EPS = 1e-8


@dataclass
class ASCIIConfig:
    """Configuration parameters for the ASCII emitter.

    This class defines hyperparameters used by the ASCII emitter for policy optimization
    and buffer management.

    Attributes:
        no_agents: Number of policies/agents to maintain and optimize.
        buffer_sample_batch_size: Size of batches sampled from the trajectory buffer.
        buffer_add_batch_size: Size of batches added to the trajectory buffer.
        no_epochs: Number of optimization epochs per iteration.
        learning_rate: Learning rate for policy optimization.
        discount_rate: Discount factor for computing returns.
        clip_param: Clipping parameter.
        std: Standard deviation for policy exploration.
    """

    no_agents: int = 256
    buffer_sample_batch_size: int = 32
    buffer_add_batch_size: int = 256
    no_epochs: int = 16
    learning_rate: float = 3e-4
    discount_rate: float = 0.99
    clip_param: float = 0.2
    std: float = 0.5


class ASCIIEmitterState(EmitterState):
    """State maintained by the ASCII emitter.

    Tracks the trajectory buffer state and random key for reproducibility.

    Attributes:
        buffer_state: State of the trajectory buffer.
        random_key: JAX random key for stochastic operations.
    """

    buffer_state: Any
    random_key: RNGKey


class ASCIIEmitter(Emitter):
    """ASCII Emitter for QD optimization.

    This emitter uses policy gradient methods to optimize policies within a
    quality-diversity framework. It maintains a buffer of trajectories and uses
    a policy optimization approach to improve policies.
    """

    def __init__(
        self,
        config: ASCIIConfig,
        policy_net: nn.Module,
        env: QDEnv,
    ) -> None:
        """Initialize the ASCII emitter.

        Args:
            config: Configuration parameters for the emitter.
            policy_net: Neural network model used for policies.
            env: Environment wrapper for QD algorithms.
        """
        # Store configuration and components
        self._config = config
        self._policy = policy_net
        self._env = env

        # Initialize the policy optimizer
        self._policy_opt = optax.adam(learning_rate=self._config.learning_rate)

        # Create a trajectory buffer for experience storage
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=self._env.episode_length,
            min_length_time_axis=self._env.episode_length,
            sample_batch_size=self._config.buffer_sample_batch_size,
            add_batch_size=self._config.buffer_add_batch_size,
            sample_sequence_length=self._env.episode_length,
            period=self._env.episode_length,
        )
        self._buffer = buffer

    @property
    def batch_size(self) -> int:
        """Returns the batch size of solutions emitted by this emitter.

        Returns:
            Number of solutions generated per emit call.
        """
        return self._config.no_agents

    @property
    def use_all_data(self) -> bool:
        """Whether this emitter should use all available data.

        Indicates if the emitter should use all data when used alongside other emitters
        in a MultiEmitter setup.

        Returns:
            True if all data should be used, False otherwise.
        """
        return True

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        random_key: RNGKey,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> Tuple[ASCIIEmitterState, RNGKey]:
        """Initialize the emitter state.

        Sets up the initial trajectory buffer and random state.

        Args:
            random_key: JAX PRNG key for stochastic operations.
            repertoire: Initial repertoire of solutions.
            genotypes: Initial population of solutions.
            fitnesses: Fitness scores of initial solutions.
            descriptors: Behavioral descriptors of initial solutions.
            extra_scores: Additional metrics or information about solutions.

        Returns:
            A tuple containing the initialized emitter state and an updated random key.
        """
        # Get environment dimensions
        obs_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length

        # Initialize trajectory buffer with a dummy transition
        dummy_transition = QDTransition.init_dummy(
            observation_dim=obs_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )

        buffer_state = self._buffer.init(dummy_transition)

        # Create the initial emitter state
        random_key, subkey = jax.random.split(random_key)
        emitter_state = ASCIIEmitterState(
            buffer_state=buffer_state,
            random_key=subkey,
        )

        return emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: ASCIIEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, Any, RNGKey]:
        """Generate a batch of new candidate solutions.

        Samples parents from the repertoire and applies the ASCII mutation to create
        improved offspring policies.

        Args:
            repertoire: Current repertoire of solutions.
            emitter_state: Current state of the emitter.
            random_key: JAX PRNG key for stochastic operations.

        Returns:
            A tuple containing the generated solutions, auxiliary information
            (empty dict), and an updated random key.
        """
        # Determine how many agents to generate
        no_agents = self._config.no_agents

        # Create separate random keys for sampling and mutation
        random_keys = jax.random.split(random_key, no_agents + 2)

        # Sample parents from the repertoire
        parents, returns, random_key, trajectories = repertoire.sample(
            random_key=random_keys[-1],
            num_samples=no_agents,
        )

        # Generate offspring through ASCII mutation
        offsprings_ascii = self.emit_ascii(
            emitter_state, parents, returns, trajectories, random_keys[:no_agents]
        )

        return offsprings_ascii, {}, random_keys[-2]

    @partial(jax.jit, static_argnames=("self",))
    def emit_ascii(
        self,
        emitter_state: ASCIIEmitterState,
        parents: Genotype,
        returns: Any,
        trajectories: Any,
        random_keys: ArrayTree,
    ) -> Genotype:
        """Generate offspring through ASCII mutation.

        Applies the ASCII mutation operator to each parent solution in parallel
        using JAX's vectorized mapping (vmap).

        Args:
            emitter_state: Current state of the emitter.
            parents: Parent solutions to mutate.
            returns: Return values associated with parents.
            trajectories: Trajectories associated with parents.
            random_keys: Random keys for stochastic mutation.

        Returns:
            Mutated offspring solutions.
        """
        # Apply mutation function to each parent in parallel
        offsprings = jax.vmap(
            self._mutation_function_ascii, in_axes=(0, 0, 0, None, 0)
        )(parents, returns, trajectories, emitter_state, random_keys)

        return offsprings

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: ASCIIEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> ASCIIEmitterState:
        """Update the emitter state with new experience.

        Updates the trajectory buffer with new transitions and refreshes the random key.

        Args:
            emitter_state: Current emitter state to update.
            repertoire: Current repertoire (not used in this implementation).
            genotypes: Current genotypes (not used in this implementation).
            fitnesses: Current fitness scores (not used in this implementation).
            descriptors: Current descriptors (not used in this implementation).
            extra_scores: Additional metrics containing transitions to add to buffer.

        Returns:
            Updated emitter state.
        """
        # Return the emitter state unchanged if no genotypes
        if genotypes is None:
            return emitter_state

        # Update random key
        random_key, subkey = jax.random.split(emitter_state.random_key)

        # Extract transitions and add to buffer
        transitions = extra_scores["transitions"]
        new_buffer_state = self._buffer.add(emitter_state.buffer_state, transitions)

        # Create updated emitter state
        new_emitter_state = ASCIIEmitterState(
            random_key=random_key, buffer_state=new_buffer_state
        )

        return new_emitter_state

    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_ascii(
        self,
        policy_params: Genotype,
        returns: Any,
        trajectories: Any,
        emitter_state: ASCIIEmitterState,
        random_key: RNGKey,
    ) -> Genotype:
        """Apply the ASCII mutation to a single policy.

        This function performs policy gradient updates to improve a policy based on
        stored experiences from the trajectory buffer.

        Args:
            policy_params: Parameters of the policy to mutate.
            returns: Return values associated with the policy.
            trajectories: Trajectories associated with the policy.
            emitter_state: Current emitter state with buffer.
            random_key: Random key for stochastic operations.

        Returns:
            Updated policy parameters after mutation.
        """
        # Initialize optimizer state for this policy
        policy_opt_state = self._policy_opt.init(policy_params)

        # Sample experiences from buffer
        batch = self._buffer.sample(emitter_state.buffer_state, random_key)
        trans = batch.experience

        # Calculate state cosine weights
        scale_returns = compute_cosine_similarity(
            trajectories.obs, jnp.squeeze(trans.obs, axis=0)
        )
        # Reward-to-go difference times cosine similarity
        standardized_returns = scale_returns * (trans.rewards - returns)

        # Training function for policy updates
        def scan_train_policy(
            carry: Tuple[Genotype, optax.OptState],
            unused: Any,
        ) -> Tuple[Tuple[Genotype, optax.OptState], Any]:
            """Inner loop function for policy training."""
            policy_params, policy_opt_state = carry

            # Perform one update step on the policy
            new_policy_params, new_policy_opt_state = self._train_policy_(
                policy_params,
                policy_opt_state,
                trans.obs,
                trans.actions,
                standardized_returns,
                trans.dones,
            )
            return (new_policy_params, new_policy_opt_state), None

        # Run multiple epochs of policy updates using JAX scan
        (policy_params, policy_opt_state), _ = jax.lax.scan(
            scan_train_policy,
            (policy_params, policy_opt_state),
            None,
            length=self._config.no_epochs,
        )

        return policy_params

    @partial(jax.jit, static_argnames=("self",))
    def _train_policy_(
        self,
        policy_params: Genotype,
        policy_opt_state: optax.OptState,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        standardized_returns: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[Genotype, optax.OptState]:
        """Perform one policy gradient update step.

        Updates the policy parameters using the computed loss gradient.

        Args:
            policy_params: Current policy parameters.
            policy_opt_state: Current optimizer state.
            obs: Observations from sampled trajectories.
            actions: Actions from sampled trajectories.
            standardized_returns: Weighted reward-to-go differences.
            mask: Binary mask indicating end of episodes.

        Returns:
            Tuple of updated policy parameters and optimizer state.
        """
        # Compute gradients of the loss with respect to policy parameters
        grads = jax.grad(self.loss)(
            policy_params, obs, actions, standardized_returns, mask
        )

        # Apply updates using the optimizer
        updates, new_policy_opt_state = self._policy_opt.update(grads, policy_opt_state)
        new_policy_params = optax.apply_updates(policy_params, updates)

        return new_policy_params, new_policy_opt_state

    @partial(jax.jit, static_argnames=("self",))
    def loss(
        self,
        params: Genotype,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        standardized_returns: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the clipped policy gradient loss.

        This loss encourages the policy to take actions that lead to higher returns
        while preventing too large policy updates.

        Args:
            params: Policy parameters.
            obs: Batch of observations.
            actions: Batch of actions.
            standardized_returns: Weighted reward-to-go differences.
            mask: Binary mask for episode boundaries.

        Returns:
            Scalar loss value.
        """
        # Compute action probabilities from the current policy
        pi, _ = self._policy.apply(params, obs)
        logps_ = pi.log_prob(actions)
        log_factor = jnp.log(self._config.std) + 0.5 * jnp.log(2.0 * jnp.pi)

        ratio = jnp.exp(logps_ + self._env.action_size * log_factor)

        # Compute losses
        pg_loss_1 = jnp.multiply(
            ratio, jax.lax.stop_gradient(standardized_returns * (1.0 - mask))
        )
        pg_loss_2 = jax.lax.stop_gradient(
            standardized_returns * (1.0 - mask)
        ) * jnp.maximum(ratio, 1.0 - self._config.clip_param)

        # Take the minimum (clipped) loss
        return -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))
