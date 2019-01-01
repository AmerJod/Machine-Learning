import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence
from torch.distributions.kl import _kl_multivariatenormal_multivariatenormal


def loglikelihood(y_pred_mu, y_pred_std, y_target):
    '''
    Calculates the loglikelihood of the target y data under the
    normal distribution parameterised by y_pred_mu and y_pred_std.

    Arguments:
            y_pred_mu: the y predictions mean
            y_pred_std: the y predictions std
            y_target: the y target values.

    Returns:
            log_prob: the loglikelihood of y_target
    '''
    normal = Normal(y_pred_mu, y_pred_std)
    log_prob = normal.log_prob(y_target).sum(dim=1).mean(dim=0)

    return log_prob


def kl_div(mu_context, std_context, mu_all, std_all):
    '''
    Calculates the KL-divergence between two multivariate
    normal distributions parameterised by mu_context, std_context
    and mu_all, std_all. Ultimately, measuring the similarlity
    between using context points and all points for the
    prediction of the y values.

    Arguments:
            mu_context: the mean value of the context normal
            std_context: the std of the context normal
            mu_all: the mu of the normal given all the points
            std_all: the std of the normal given all the points
    '''
    context_dist = MultivariateNormal(mu_context, std_context)
    all_dist = MultivariateNormal(mu_all, std_all)

    return _kl_multivariatenormal_multivariatenormal(all_dist, context_dist)
