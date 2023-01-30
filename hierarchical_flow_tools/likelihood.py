"""
Flow likelihood class definition and helper functions.
"""

import numpy as np
import torch
from glasflow.flows.base import Flow

class FlowLikelihood:
    """_summary_

    Parameters
    ----------
    flow : Flow
        A trained normalising flow that will perform the log_likelihood calculation. 
    device : str
        The pytorch device on which to perform all operations
    conditional_rescaling_function : function, optional
        A function for remapping input conditionals to , by default None
    data : Tensor, optional
        The data (rescaled if necessary) to be input to the flow for all log_likelihood evaluations, by default None (data to be provided on log_likelihood call)
    batch_size : int, optional
        Size of each batch to feed into the flow, by default None (in which case, all data is fed through the flow at once - may lead to memory errors for large datasets)
    """
    def __init__(
        self,
        flow: Flow,
        device: str,
        conditional_rescaling_function=None,
        data=None,
        batch_size=None,
    ) -> None:

        if conditional_rescaling_function is None:
            conditional_rescaling_function = lambda x: x
        self.conditional_rescaling_function = conditional_rescaling_function
        self.flow = flow
        self.device = device
        self.data = torch.as_tensor(
            data, device=self.device
        )  # data should be rescaled if required
        if self.data.dim() == 2:
            self.data = self.data.unsqueeze(0)
        self.batch_size = batch_size

    def log_likelihood(self, conditional, data=None):
        """  Get the log_likelihood of the data given a set of M conditional parameters.
        This method supports vectorised evaluation of N sets of conditionals (of shape (N, M)).
        If no data is provided, the method falls back to the data given on instantiation of the parent class.

        Parameters
        ----------
        conditional : Tensor
            The conditional values to evaluate the log_likelihood for. If vectorised, should take the shape (N_points, N_conditional_params)
        data : Tensor, optional
            Data to provide to the flow for the log_likelihood evaluation, by default None (falls back to data supplied on instantiation of this class)

        Returns
        -------
        Tensor
            The combined log_likelihoods for the supplied conditional values over all of the data.

        Raises
        ------
        RuntimeError
            Raises if no data has been provided at any stage for the log_likelihood evaluations.
        """
        if data is None:
            data = self.data
        if data is None:
            raise RuntimeError(
                "No data provided to calculate log-likelihood with respect to"
            )

        rescaled_conditional = self.conditional_rescaling_function(conditional)
        if rescaled_conditional.dim() == 1:
            rescaled_conditional = rescaled_conditional.unsqueeze(0)
        nevents, nsamples, nparams = data.shape
        npoints, nconparams = rescaled_conditional.shape
        if self.batch_size is None:  # do all in one
            data_in = data.unsqueeze(0).expand(npoints, -1, -1, -1).reshape(-1, nparams)
            cond_in = (
                rescaled_conditional.unsqueeze(1)
                .expand(-1, nevents * nsamples, -1)
                .reshape(-1, nconparams)
            )

            with torch.no_grad():
                probs = self.flow.log_prob(data_in, conditional=cond_in).reshape(
                    npoints, nevents, nsamples
                )
                log_prob = torch.sum(torch.logsumexp(probs, dim=-1), dim=-1)

        else:
            log_prob = torch.zeros(npoints)
            nbatches = np.ceil(npoints / self.batch_size).astype(np.int32)
            cond_in = rescaled_conditional.unsqueeze(1).expand(
                -1, nevents * nsamples, -1
            )

            for k in range(nbatches):
                this_cond = cond_in[
                    k * self.batch_size : min((k + 1) * self.batch_size, npoints), :, :
                ]
                data_in = (
                    data.unsqueeze(0)
                    .expand(this_cond.shape[0], -1, -1, -1)
                    .reshape(-1, nparams)
                )

                with torch.no_grad():
                    probs = self.flow.log_prob(
                        data_in, conditional=this_cond.reshape(-1, nconparams)
                    ).reshape(this_cond.shape[0], nevents, nsamples)
                    log_prob[
                        k * self.batch_size : min((k + 1) * self.batch_size, npoints)
                    ] = torch.sum(torch.logsumexp(probs, dim=-1), dim=-1)
        log_prob -= np.log(nsamples) * nevents
        return log_prob
