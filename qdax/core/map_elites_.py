"""
Core components of the MAP-Elites algorithm.

MAP-Elites is a Quality-Diversity (QD) algorithm that maintains a collection of
high-performing solutions across a repertoire of behaviors defined by descriptors.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax

from qdax.core.containers.ascii_me_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


class MAPElites:
    """Core implementation of the MAP-Elites algorithm.

    MAP-Elites maintains a collection of diverse, high-performing solutions organized
    by their behavioral characteristics. The algorithm uses emitters to generate new
    candidate solutions and evaluates them based on both performance (fitness) and
    behavioral characteristics (descriptors).

    Args:
        scoring_function: Function that evaluates genotypes to determine their
            fitness scores and behavioral descriptors.
        emitter: Strategy for generating new candidate solutions from the current
            repertoire. Handles both creation of new solutions and updating internal
            state based on evaluation results.
        metrics_function: Function that computes metrics to track the evolution and
            performance of the repertoire.
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey]:
        """
        Initialize a MAP-Elites repertoire with an initial population of genotypes.

        This method evaluates the initial genotypes, creates a repertoire organized
        around the provided centroids, and initializes the emitter state.

        Args:
            genotypes: Initial population of solutions, where each leaf in the pytree
                has shape (batch_size, num_features).
            centroids: Tessellation centroids defining the behavioral space
                partitioning, with shape (num_centroids, num_descriptors).
            random_key: JAX PRNG key for stochastic operations.

        Returns:
            A tuple containing:
            - An initialized MAP-Elites repertoire
            - The initial state of the emitter
            - An updated random key
        """
        # Score initial genotypes to get fitness and descriptors
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # Initialize the repertoire with initial genotypes and evaluations
        repertoire = MapElitesRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
        )

        # Initialize the emitter state
        emitter_state, random_key = self._emitter.init(
            random_key=random_key,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # Update the emitter state with initial data
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        """
        Perform one iteration of the MAP-Elites algorithm.

        The update process follows these steps:
        1. Generate new candidate solutions (offspring) using the emitter
        2. Evaluate the offspring to determine fitness and descriptors
        3. Add successful candidates to the repertoire
        4. Update the emitter state based on evaluation results
        5. Compute metrics on the updated repertoire

        Args:
            repertoire: Current MAP-Elites repertoire
            emitter_state: Current state of the emitter
            random_key: JAX PRNG key for stochastic operations

        Returns:
            A tuple containing:
            - The updated MAP-Elites repertoire
            - The updated emitter state
            - Metrics about the repertoire and update
            - An updated random key
        """
        # Generate offspring using the emitter
        genotypes, update_info, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # Evaluate the offspring
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # Add evaluated offspring to the repertoire
        repertoire, is_offspring_added = repertoire.add(
            genotypes, descriptors, fitnesses, extra_scores
        )

        # Update emitter state based on evaluation results
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # Compute metrics on the updated repertoire
        metrics = self._metrics_function(repertoire)
        metrics["is_offspring_added"] = is_offspring_added

        return repertoire, emitter_state, metrics, random_key

    @partial(jax.jit, static_argnames=("self",))
    def scan_update(
        self,
        carry: Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey],
        unused: Any,
    ) -> Tuple[Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey], Metrics]:
        """
        Update function compatible with JAX's lax.scan for efficient iteration.

        This function reformats the update method to work with JAX's scan primitive,
        enabling more efficient multiple updates through JIT compilation.

        Args:
            carry: A tuple containing (repertoire, emitter_state, random_key)
            unused: Unused element required by the lax.scan API

        Returns:
            A tuple containing:
            - The updated carry tuple (repertoire, emitter_state, random_key)
            - Metrics from the current update
        """
        repertoire, emitter_state, random_key = carry
        repertoire, emitter_state, metrics, random_key = self.update(
            repertoire,
            emitter_state,
            random_key,
        )

        return (repertoire, emitter_state, random_key), metrics
