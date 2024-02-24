
# Perceptual Neural Physical Sound Matching
Han Han, Vincent Lostanlen, Mathieu Lagrange

ICASSP Paper on arxiv: (https://arxiv.org/abs/2301.02886)

TASLP submission on arxiv: (https://arxiv.org/abs/2311.14213)

Sound Matching examples: https://lylyhan.github.io/perceptual_neural_physical/ 

## Installation:

```bash
# clone project   
git clone https://github.com/lylyhan/perceptual_neural_physical.git

# install project   
cd perceptual_neural_physical
python -m pip install .

# install kymatio
git clone https://github.com/cyrusvahidi/jtfs-gpu.git
cd kymatio
pip install -e .

```

## Example Usage:
Compute $S = (\Phi \circ g)(\theta) = \Phi(x)$, where $g$ is a FTM synthesizer, and $\Phi$ is the JTFS coefficients.
```

from pnp_synth.neural import forward
from kymatio.torch import TimeFrequencyScattering1D
from pnp_synth import utils
import torch
import functools

# define synthesizer type
synth_type = "ftm"

# load JTFS hyperparameters based on synthesizer type
jtfs_params = utils.jtfsparam(synth_type)

# define scaling operation on parameter theta
logscale = True
scaler = None


# define JTFS operator 
jtfs_operator = TimeFrequencyScattering1D(**jtfs_params, out_type="list").to("cuda")
jtfs_operator.average_global = True


phi = functools.partial(utils.S_from_x, jtfs_operator=jtfs_operator)
g = functools.partial(utils.x_from_theta, synth_type=synth_type, logscale=logscale)
theta = torch.randn(5)

# Compute S given theta as input
S = forward.pnp_forward(theta[:,None], 
                        Phi=phi,
                        g=g, 
                        scaler=scaler))

```

Compute $\nabla_{(\Phi \circ g)} (\theta)$

```
import taslp23

S_from_nu = taslp23.pnp_forward_factory(scaler, logscale, synth_type)
dS_over_dnu = functorch.jacfwd(S_from_nu)

S = S_from_nu(theta)
# Compute Jacobian: d(S) / d(nu)
J = dS_over_dnu(theta)

# Compute Riemannian metric of a given theta
M = torch.mm(J.T, J)
```
