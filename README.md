# nflojax

A minimal, hackable normalizing flows library in JAX.

## Installation

nflojax is used as an editable source package in downstream research repos:

```bash
git clone https://github.com/alexxthiery/nflojax.git
cd nflojax
pip install -e ".[test]"
```

`nflojax.__init__` intentionally has no top-level exports. Import public
objects from their submodules, for example `nflojax.builders`,
`nflojax.transforms`, and `nflojax.distributions`.

## Quick Start

```python
import jax
from nflojax.builders import build_realnvp

key = jax.random.PRNGKey(0)
flow, params = build_realnvp(
    key, dim=4, num_layers=4, hidden_dim=64, n_hidden_layers=2,
)

samples = flow.sample(params, key, shape=(1000,))                      # (1000, 4)
log_prob = flow.log_prob(params, samples)                               # (1000,)
samples, log_prob = flow.sample_and_log_prob(params, key, shape=(1000,))
```

## Documentation

| Document | Content |
|----------|---------|
| [README.md](README.md) | Install, quick start, document map |
| [USAGE.md](USAGE.md) | How-to cookbook with copy-pasteable examples |
| [REFERENCE.md](REFERENCE.md) | API reference: classes, builders, options, param structure |
| [INTERNALS.md](INTERNALS.md) | Math foundations and design decisions |
| [EXTENDING.md](EXTENDING.md) | Recipes for custom transforms, distributions, conditioners |
| [AGENTS.md](AGENTS.md) | Project context for coding agents |
