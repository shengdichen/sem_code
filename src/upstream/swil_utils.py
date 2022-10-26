import torch

# from gsw import *
from torch import Tensor


def get_random_projections(
    latent_dim: int, num_samples: int, proj_dist="normal"
) -> Tensor:
    """
    Returns random samples from latent distribution's (Gaussian)
    unit sphere for projecting the encoded samples and the
    distribution samples.

    :param latent_dim: (Int) Dimensionality of the latent space (D)
    :param num_samples: (Int) Number of samples required (S)
    :return: Random projections from the latent unit sphere
    """
    if proj_dist == "normal":
        rand_samples = torch.randn(num_samples, latent_dim)
    elif proj_dist == "cauchy":
        rand_samples = (
            torch.distributions.Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
            .sample((num_samples, latent_dim))
            .squeeze()
        )
    else:
        raise ValueError("Unknown projection distribution.")

    rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1, 1)
    return rand_proj  # [S x D]


def get_generalized_random_projections(
    projection_model, latent_dim: int, num_samples: int, proj_dist: str
) -> Tensor:
    """
    Returns random samples from latent distribution's (Gaussian)
    unit sphere for projecting the encoded samples and the
    distribution samples.

    :param latent_dim: (Int) Dimensionality of the latent space (D)
    :param num_samples: (Int) Number of samples required (S)
    :return: Random projections from the latent unit sphere
    """

    # flows?
    rand_samples = projection_model(torch.randn(num_samples, latent_dim))

    rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1, 1)
    return rand_proj  # [S x D]


def compute_swd(
    z_expert: Tensor,
    z_policy: Tensor,
    p: float,
    reg_weight: float,
    n_proj: int,
    latent_dim: int,
) -> Tensor:
    """
    Computes the Sliced Wasserstein Distance (SWD) - which consists of
    randomly projecting the encoded and prior vectors and computing
    their Wasserstein distance along those projections.

    :param z: Latent samples # [N  x D]
    :param p: Value for the p^th Wasserstein distance
    :param reg_weight:
    :return:
    """
    prior_z = z_policy
    device = z_expert.device

    proj_matrix = get_random_projections(latent_dim, n_proj).transpose(0, 1).to(device)
    # gen_proj_matrix = get_generalized_random_projections(latent_dim, ns_proj).transpose(0,1).to(device)

    latent_projections = z_expert.matmul(proj_matrix)  # [N x S]
    prior_projections = prior_z.matmul(proj_matrix)  # [N x S]

    # The Wasserstein distance is computed by sorting the two projections
    # across the batches and computing their element-wise l2 distance
    w_dist = (
        torch.sort(latent_projections.t(), dim=1)[0]
        - torch.sort(prior_projections.t(), dim=1)[0]
    )
    w_dist = w_dist.pow(p)
    return reg_weight * w_dist.mean()


# def _t(arr):
# return torch.FloatTensor(arr)
