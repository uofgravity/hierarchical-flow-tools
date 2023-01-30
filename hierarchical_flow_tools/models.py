"""
Provides some out-of-the-box use of hierarchical-flow-tools with other packages.
"""

from nessai.model import Model
from .likelihood import FlowLikelihood
import numpy as np
import torch

#nessai model class
class NessaiModel(Model):
    """Model class for use with the `nessai` package for parameter estimation.

    Parameters
    ----------
    names : list
        A list of label strings for each parameter.
    bounds : dict
        A dict of the prior bounds, assuming uniform priors on all parameters.
    flowlike : FlowLikelihood
        The FlowLikelihood class to use for the log_likelihood calls.
    """
    def __init__(self, names: list, bounds: list, flow_likelihood: FlowLikelihood):
        self.names = names
        self.bounds = bounds
        self.flow_lhood = flow_likelihood
        self._vectorised_likelihood = True  # we force this to be True as there is some minor stochasticity in the flow output on GPU.
        self.device = flow_likelihood.device

    def unpack_live_point(self, x):
        """Unpacks a live point to a torch tensor for use in the log_likelihood.

        Parameters
        ----------
        x : structured_array
            The live points to be unpacked into a torch tensor.

        Returns
        -------
        Tensor
            The unpacked live points.
        """

        start = np.array([x[n] for n in self.names]).T
        if start.ndim == 1:
            start = start[None,:]
        return torch.as_tensor(start, device=self.device).float()

    def log_prior(self, x):
        """Returns log of prior given a live point assuming uniform
        priors on each parameter.

        Parameters
        ----------
        x : structured array
            the live points for which to evaluate the prior probability.

        Returns
        -------
        ndarray
            the log_prior probabilities
        """
        # Check if values are in bounds, returns True/False
        # Then take the log to get 0/-inf and make sure the dtype is float
        log_p = np.log(self.in_bounds(x), dtype="float").astype(np.float32)
        # Iterate through each parameter (x and y)
        # since the live points are a structured array we can
        # get each value using just the name
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
        """Log likelihood wrapper that calls the `FlowLikelihood.log_likelihood` method.

        Parameters
        ----------
        x : structured array
            the live points for which to evaluate the prior probability.

        Returns
        -------
        ndarray
            The resulting log_likelihoods, cast to a numpy array for compatibility with `nessai`.
        """
        conditional = self.unpack_live_point(x)
        out = self.flow_lhood.log_likelihood(conditional).cpu().numpy()
        return out