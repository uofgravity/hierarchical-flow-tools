import numpy as np
import torch

class FlowLikelihood():
    def __init__(self, flow, device, conditional_rescaling_function=lambda x: x, data=None, batch_size=None) -> None:
        self.conditional_rescaling_function = conditional_rescaling_function
        self.flow = flow
        self.device = device
        self.data = torch.as_tensor(data, device=self.device)  # data should be rescaled if required
        if self.data.dim() == 2:
            self.data = self.data.unsqueeze(0)
        self.batch_size = batch_size

    def log_likelihood(self, conditional, data=None):
        """
        Get the log_likelihood of the data given a set of M conditional parameters.
        This method supports vectorised evaluation of N sets of conditionals (of shape (N, M)).
        If no data is provided, the method falls back to the data given on instantiation of the parent class.
        """
        if data is None:
            data = self.data
        if data is None:
            raise Exception("No data provided to calculate log-likelihood with respect to") 
        
        rescaled_conditional = self.conditional_rescaling_function(conditional)
        if rescaled_conditional.dim() == 1:
            rescaled_conditional = rescaled_conditional.unsqueeze(0)
        # conditional has shape Npoints * Nconparams
        # data has shape Nevents * Nsamples * Nparams
        # we want to vectorise over both of these, producing Npoints log-likelihoods
        # logsumexp over each event's probability, then sum over logs for total log-likelihoods

        # the flow takes in 2d

        nevents, nsamples, nparams = data.shape
        npoints, nconparams = rescaled_conditional.shape
        if self.batch_size is None:  # do all in one   
            data_in = data.unsqueeze(0).expand(npoints, -1, -1, -1).reshape(-1, nparams)
            # cond_in = rescaled_conditional.unsqueeze(2).unsqueeze(3).expand(-1, -1, nevents, nsamples).reshape(-1, nconparams)
            cond_in = rescaled_conditional.unsqueeze(1).expand(-1, nevents * nsamples, -1).reshape(-1, nconparams)

            with torch.no_grad():
                probs = self.flow.log_prob(data_in, conditional=cond_in).reshape(npoints, nevents, nsamples)
                log_prob = torch.sum(torch.logsumexp(probs,dim=-1),dim=-1)

        else:
            log_prob = torch.zeros(npoints)
            nbatches = np.ceil(npoints / self.batch_size).astype(np.int32)
            # cond_in = rescaled_conditional.unsqueeze(2).unsqueeze(3).expand(-1, -1, nevents, nsamples)
            cond_in = rescaled_conditional.unsqueeze(1).expand(-1, nevents * nsamples, -1)
            
            for k in range(nbatches):
                this_cond = cond_in[k*self.batch_size:min((k+1)*self.batch_size,npoints),:,:]#,:]
                data_in = data.unsqueeze(0).expand(this_cond.shape[0], -1, -1, -1).reshape(-1, nparams)

                # print(data_in.shape, cond_in.shape)
                with torch.no_grad():
                    # print('here')
                    probs = self.flow.log_prob(data_in, conditional=this_cond.reshape(-1, nconparams)).reshape(this_cond.shape[0], nevents, nsamples)
                    log_prob[k*self.batch_size:min((k+1)*self.batch_size,npoints)] = torch.sum(torch.logsumexp(probs,dim=-1),dim=-1)
                    # print('done')
        log_prob -= np.log(nsamples) * nevents
        return log_prob
