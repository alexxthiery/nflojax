from __future__ import annotations

from .shared import *

# ===================================================================
# Affine Coupling Layer
# ===================================================================
@dataclass
class AffineCoupling:
    """
    RealNVP-style affine coupling layer.

    This layer splits the input vector x into two parts using a binary mask m.
    Masked dimensions (where m = 1) pass through unchanged. Unmasked dimensions
    (where m = 0) are transformed using parameters produced by a conditioner
    network.

    It roughly works as follows:
    * Split x into x1 = x * m and x2 = x * (1 - m)
    * Use x1 as input to a conditioner network to produce shift and log_scale for transforming x2.
    * Apply the elementwise affine transform on x2: y2 = x2 * exp(log_scale) + shift.
      Note that the shift and log_scale are zeroed out on the masked dimensions.
    * Combine y1 = x1 and y2 to produce output y = y1 + y2. This only modifies the unmasked dimensions.

    Forward transformation y = T(x):
    x1 = x * m
    x2 = x * (1 - m)
    (shift, log_scale) = conditioner(x1) * (1 - m)
    y1 = x1
    y2 = (x2 * exp(log_scale) + shift)
    y = y1 + y2
    The returned log_det is log |det ∂y/∂x|, equal to the sum of log_scale on the
    unmasked coordinates.

    Inverse transformation x = T^{-1}(y):
    y1 = y * m
    y2 = y * (1 - m)
    (shift, log_scale) = conditioner(y1) * (1 - m)
    x2 = (y2 - shift) * exp(-log_scale)
    x = y1 + x2
    The returned log_det is log |det ∂x/∂y| = -sum(log_scale).

    Parameters:
    params["mlp"]: PyTree containing the Flax parameters of the conditioner.

    All operations act along the last dimension. The mask must be one-dimensional
    with the same length as the feature dimension.

    Note:
    In this implementation, the conditioner network is typically initialized
    such that its output is identically zero at initialization. In that case,
    shift = 0 and log_scale = 0, so this layer is exactly the identity map
    at the start of training.

    Conditional flows:
      The optional `context` argument enables conditional density estimation p(x|c).
      When provided, context is concatenated to the masked input before being passed
      to the conditioner network. The conditioner MLP must be initialized with
      `context_dim` matching the size of the context vector.

      Context shape: (batch, context_dim) or (context_dim,) for a single sample.
      The same context is used for all coupling layers in a flow.

    References:
      - Dinh, Krueger, Bengio (2017). "NICE: Non-linear Independent Components Estimation"
      - Dinh, Sohl-Dickstein, Bengio (2017). "Density estimation using Real NVP"
    """
    mask: Array          # shape (dim,), values 0 or 1
    conditioner: MLP     # Flax MLP module (definition, no params inside)
    max_log_scale: float = 5.0
    max_shift: float | None = None  # Default: exp(max_log_scale)

    def __post_init__(self):
        # Ensure mask is a 1D array.
        self.mask = jnp.asarray(self.mask)
        if self.mask.ndim != 1:
            raise ValueError(
                f"AffineCoupling mask must be 1D, got shape {self.mask.shape}."
            )
        # Validate conditioner interface.
        validate_conditioner(self.conditioner, name="AffineCoupling.conditioner")

    @property
    def dim(self) -> int:
        return int(self.mask.shape[0])

    @staticmethod
    def required_out_dim(dim: int) -> int:
        """
        Return required conditioner output dimension for AffineCoupling.

        The conditioner must output shift and log_scale for each dimension,
        so out_dim = 2 * dim.

        Arguments:
            dim: Input/output dimensionality.

        Returns:
            Required output dimension for conditioner (2 * dim).
        """
        return 2 * dim

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        dim: int,
        mask: Array,
        hidden_dim: int,
        n_hidden_layers: int,
        *,
        context_dim: int = 0,
        activation: Callable[[Array], Array] = nn.elu,
        res_scale: float = 0.1,
        max_log_scale: float = 5.0,
        max_shift: float | None = None,
    ) -> Tuple["AffineCoupling", dict]:
        """
        Factory method to create AffineCoupling with properly configured MLP.

        This handles the output dimension calculation internally and initializes
        parameters, returning both the coupling and its params ready to use.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            dim: Input/output dimensionality.
            mask: Binary mask of shape (dim,). 1 = frozen, 0 = transformed.
            hidden_dim: Width of hidden layers in conditioner MLP.
            n_hidden_layers: Number of residual blocks in conditioner MLP.
            context_dim: Context dimension (0 for unconditional).
            activation: Activation function for MLP (default: elu).
            res_scale: Residual connection scale (default: 0.1).
            max_log_scale: Bound on |log_scale| via tanh (default: 5.0).
            max_shift: Bound on |shift| via tanh (default: exp(max_log_scale)).

        Returns:
            Tuple of (coupling, params) ready to use.

        Raises:
            ValueError: If mask length doesn't match dim, or dim <= 0.

        Example:
            >>> coupling, params = AffineCoupling.create(
            ...     key, dim=4, mask=jnp.array([1, 0, 1, 0]),
            ...     hidden_dim=64, n_hidden_layers=2
            ... )
            >>> y, log_det = coupling.forward(params, x)
        """
        # Validate inputs
        if dim <= 0:
            raise ValueError(f"AffineCoupling.create: dim must be positive, got {dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"AffineCoupling.create: hidden_dim must be positive, got {hidden_dim}.")

        mask = jnp.asarray(mask)
        if mask.shape != (dim,):
            raise ValueError(
                f"AffineCoupling.create: mask shape {mask.shape} doesn't match (dim,) = ({dim},)."
            )

        # Create MLP with correct output dimension
        out_dim = cls.required_out_dim(dim)
        mlp = MLP(
            x_dim=dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            out_dim=out_dim,
            activation=activation,
            res_scale=res_scale,
        )

        # Create coupling
        coupling = cls(
            mask=mask,
            conditioner=mlp,
            max_log_scale=max_log_scale,
            max_shift=max_shift,
        )

        # Initialize params
        params = coupling.init_params(key, context_dim=context_dim)

        return coupling, params

    def _condition(
        self,
        params: dict,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Run the conditioner network and produce shift and log_scale.

        params:
          dict with key "mlp" containing the conditioner parameters.
        x:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor of shape (..., context_dim) or (context_dim,).
        g_value:
          optional gate value from identity_gate(context). When g_value=0, the
          transform should be identity, so shift and log_scale are zeroed.
        """
        if "mlp" not in params:
            raise KeyError(
                "AffineCoupling expected params to contain key 'mlp'."
            )

        if x.shape[-1] != self.dim:
            raise ValueError(
                f"AffineCoupling expected input with last dimension {self.dim}, "
                f"got {x.shape[-1]}."
            )

        # Use only the masked part as input to the conditioner.
        # Broadcasting: mask has shape (dim,), x has shape (..., dim).
        x_masked = x * self.mask

        # Apply the MLP. We expect output of size 2 * dim
        # which we split into shift and log_scale_raw.
        mlp_params = params["mlp"]
        out = self.conditioner.apply({"params": mlp_params}, x_masked, context)

        if out.shape[-1] != 2 * self.dim:
            raise ValueError(
                f"Conditioner output last dimension should be 2 * dim = {2 * self.dim}, "
                f"got {out.shape[-1]}."
            )

        shift, log_scale_raw = jnp.split(out, 2, axis=-1)

        # Only transform the unmasked part: zero out contributions on masked dims.
        # (1 - mask) has 1 for transformed dims, 0 otherwise.
        m_unmasked = 1.0 - self.mask

        # Bound both shift and log_scale to avoid numerical explosions.
        # Default max_shift = exp(max_log_scale) matches the maximum scale factor.
        max_shift = self.max_shift if self.max_shift is not None else jnp.exp(self.max_log_scale)
        shift = jnp.tanh(shift / max_shift) * max_shift * m_unmasked
        log_scale = jnp.tanh(log_scale_raw / self.max_log_scale) * self.max_log_scale * m_unmasked

        # Apply identity gate: when g_value=0, shift=0 and log_scale=0 => identity.
        if g_value is not None:
            g = g_value[..., None]  # broadcast to (..., 1) for element-wise multiply
            shift = g * shift
            log_scale = g * log_scale

        return shift, log_scale

    def forward(
        self,
        params: dict,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Forward transform: x -> y, returning (y, log_det).

        params:
          dict with key "mlp" for conditioner parameters.
        x:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor passed to the conditioner.
        g_value:
          optional gate value for identity_gate. When g_value=0, returns identity.

        Returns:
          y: transformed tensor of shape (..., dim).
          log_det: log |det J| with shape x.shape[:-1].
        """
        shift, log_scale = self._condition(params, x, context, g_value=g_value)

        x1 = x * self.mask
        x2 = x * (1.0 - self.mask)

        y2 = x2 * jnp.exp(log_scale) + shift
        y = x1 + y2

        # Sum log_scale over transformed dimensions.
        log_det = jnp.sum(log_scale, axis=-1)
        return y, log_det

    def inverse(
        self,
        params: dict,
        y: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Inverse transform: y -> x, returning (x, log_det).

        params:
          dict with key "mlp" for conditioner parameters.
        y:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor passed to the conditioner.
        g_value:
          optional gate value for identity_gate. When g_value=0, returns identity.

        Returns:
          x: inverse-transformed tensor of shape (..., dim).
          log_det: log |det d x / d y| with shape y.shape[:-1].
        """
        shift, log_scale = self._condition(params, y, context, g_value=g_value)

        y1 = y * self.mask
        y2 = y * (1.0 - self.mask)

        x2 = (y2 - shift) * jnp.exp(-log_scale)
        x = y1 + x2

        # Inverse log-det is negative of forward log-det.
        log_det = -jnp.sum(log_scale, axis=-1)
        return x, log_det

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """
        Initialize parameters for this transform.

        Uses Flax init to create MLP parameters. With zero-initialized final layer,
        the transform starts at identity.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            context_dim: Context dimension (0 for unconditional).

        Returns:
            Dict with key 'mlp' containing MLP parameters.
        """
        dummy_x = jnp.zeros((1, self.dim), dtype=jnp.float32)
        dummy_context = jnp.zeros((1, context_dim), dtype=jnp.float32) if context_dim > 0 else None
        variables = self.conditioner.init(key, dummy_x, dummy_context)
        mlp_params = variables["params"]

        # Zero-init final layer for identity-start (if conditioner supports it).
        if hasattr(self.conditioner, "get_output_layer") and hasattr(self.conditioner, "set_output_layer"):
            out_layer = self.conditioner.get_output_layer(mlp_params)
            kernel = jnp.zeros_like(out_layer["kernel"])
            bias = jnp.zeros_like(out_layer["bias"])
            mlp_params = self.conditioner.set_output_layer(mlp_params, kernel, bias)

        return {"mlp": mlp_params}
