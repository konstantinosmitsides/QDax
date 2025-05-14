""" Implements neural networks models that are commonly found in the RL literature."""

from typing import Any, Callable, Optional, Sequence, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    """MLP module."""

    layer_sizes: Tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = None

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        hidden = obs
        for i, hidden_size in enumerate(self.layer_sizes):

            if i != len(self.layer_sizes) - 1:
                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(hidden)
                hidden = self.activation(hidden)  # type: ignore

            else:
                if self.kernel_init_final is not None:
                    kernel_init = self.kernel_init_final
                else:
                    kernel_init = self.kernel_init

                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=kernel_init,
                    use_bias=self.bias,
                )(hidden)

                if self.final_activation is not None:
                    hidden = self.final_activation(hidden)

        return hidden


class MLPDC(nn.Module):
    """Descriptor-conditioned MLP module."""

    layer_sizes: Tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = None

    @nn.compact
    def __call__(self, obs: jnp.ndarray, desc: jnp.ndarray) -> jnp.ndarray:
        hidden = jnp.concatenate([obs, desc], axis=-1)
        for i, hidden_size in enumerate(self.layer_sizes):

            if i != len(self.layer_sizes) - 1:
                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(hidden)
                hidden = self.activation(hidden)  # type: ignore

            else:
                if self.kernel_init_final is not None:
                    kernel_init = self.kernel_init_final
                else:
                    kernel_init = self.kernel_init

                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=kernel_init,
                    use_bias=self.bias,
                )(hidden)

                if self.final_activation is not None:
                    hidden = self.final_activation(hidden)

        return hidden


class QModule(nn.Module):
    """Q Module."""

    hidden_layer_sizes: Tuple[int, ...]
    n_critics: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        hidden = jnp.concatenate([obs, actions], axis=-1)
        res = []
        for _ in range(self.n_critics):
            q = MLP(
                layer_sizes=self.hidden_layer_sizes + (1,),
                activation=nn.relu,
                kernel_init=jax.nn.initializers.lecun_uniform(),
            )(hidden)
            res.append(q)
        return jnp.concatenate(res, axis=-1)


class QModuleDC(nn.Module):
    """Q Module."""

    hidden_layer_sizes: Tuple[int, ...]
    n_critics: int = 2

    @nn.compact
    def __call__(
        self, obs: jnp.ndarray, actions: jnp.ndarray, desc: jnp.ndarray
    ) -> jnp.ndarray:
        hidden = jnp.concatenate([obs, actions], axis=-1)
        res = []
        for _ in range(self.n_critics):
            q = MLPDC(
                layer_sizes=self.hidden_layer_sizes + (1,),
                activation=nn.relu,
                kernel_init=jax.nn.initializers.lecun_uniform(),
            )(hidden, desc)
            res.append(q)
        return jnp.concatenate(res, axis=-1)


class MLPMCPG(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    no_neurons: int = 64
    kernel_init: Callable[..., Any] = jax.nn.initializers.orthogonal(scale=jnp.sqrt(2))
    final_init: Callable[..., Any] = jax.nn.initializers.orthogonal(scale=0.01)
    std: float = 0.5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[distrax.Distribution, jnp.ndarray]:
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            self.no_neurons,
            kernel_init=self.kernel_init,
            bias_init=nn.initializers.constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.no_neurons,
            kernel_init=self.kernel_init,
            bias_init=nn.initializers.constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=self.final_init,
            bias_init=nn.initializers.constant(0.0),
        )(actor_mean)
        pi = distrax.MultivariateNormalDiag(
            loc=actor_mean, scale_diag=jnp.ones_like(actor_mean) * self.std
        )

        return pi, actor_mean
