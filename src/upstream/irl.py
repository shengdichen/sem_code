import torch.autograd as autograd
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.optim import Adam

from models import MiniGridCNN
from swil_utils import *
from utils import vampprior_kld_twolayervae, vampprior_kld_vae, gaussian_kld


class GAILDiscriminator(nn.Module):
    def __init__(
        self,
        env,
        layer_dims,
        lr,
        use_actions=True,
        use_cnn_base=False,
        l2_coeff=0,
        irm_coeff=0,
        lip_coeff=0,
        bias=False,
    ):
        super(GAILDiscriminator, self).__init__()

        ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        if not ac_shapes:
            ac_shapes = [1]
        if use_actions:
            layer_dims = [ob_shapes[-1] + ac_shapes[-1]] + layer_dims
        else:
            layer_dims = [ob_shapes[-1]] + layer_dims
        ac_len = ac_shapes[0]

        self.layer_dims = layer_dims
        self.lr = lr
        self.use_actions = use_actions
        self.irm_coeff = irm_coeff
        self.lip_coeff = lip_coeff

        self.use_cnn_base = use_cnn_base

        if use_cnn_base:
            self.base = MiniGridCNN(layer_dims, use_actions)
        else:
            self.base = nn.Sequential(
                torch.nn.Linear(self.layer_dims[0], self.layer_dims[1], bias),
                torch.nn.PReLU(),
            )

        self.discriminator_layers = []
        for i in range(2, len(layer_dims)):
            self.discriminator_layers += [
                torch.nn.Linear(
                    in_features=layer_dims[i - 1], out_features=layer_dims[i], bias=bias
                ),
                torch.nn.PReLU(),
            ]

        self.discriminator_layers += [
            torch.nn.Linear(in_features=layer_dims[-1], out_features=1, bias=bias)
        ]

        self.discriminator = nn.Sequential(*self.discriminator_layers)

        if torch.cuda.is_available():
            self.base.cuda()
            self.discriminator.cuda()

        # self.module_list = nn.ModuleList([self.base, self.discriminator])

        base_params = list(self.base.parameters())
        d_params = list(self.discriminator.parameters())
        base_params.extend(d_params)
        self.d_optimizer = Adam(base_params, lr=self.lr, weight_decay=l2_coeff)

    def forward(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        d_out = self.discriminator(base_out)
        return d_out

    def get_reward(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        d_out = self.discriminator(base_out)
        self.reward = -torch.squeeze(torch.log(torch.sigmoid(d_out) + 1e-8))
        return self.reward

    def irm_penalty(self, logits, y):
        scale = torch.tensor(1.0).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def compute_penalty(self, logits, y):
        scale = torch.tensor(1.0).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        g1 = autograd.grad(loss[0::2].mean(), [scale], create_graph=True)[0]
        g2 = autograd.grad(loss[1::2].mean(), [scale], create_graph=True)[0]
        return (g1 * g2).sum()

    # lipschitz penalty
    def lip_penalty(self, update_dict):
        interp_inputs = []
        for policy_input, expert_input in zip(
            update_dict["policy_obs"], update_dict["expert_obs"]
        ):
            obs_epsilon = torch.rand(policy_input.shape)
            interp_input = obs_epsilon * policy_input + (1 - obs_epsilon) * expert_input
            interp_input.requires_grad = True  # For gradient calculation
            interp_inputs.append(interp_input)
        if self.use_actions:
            action_epsilon = torch.rand(update_dict["policy_acs"].shape)

            dones_epsilon = torch.rand(update_dict["policy_dones"].shape)
            action_inputs = torch.cat(
                [
                    action_epsilon * update_dict["policy_acs"]
                    + (1 - action_epsilon) * update_dict["expert_acs"],
                    dones_epsilon * update_dict["policy_dones"]
                    + (1 - dones_epsilon) * update_dict["expert_dones"],
                ],
                dim=1,
            )
            action_inputs.requires_grad = True
            hidden, _ = self.encoder(interp_inputs, action_inputs)
            encoder_input = tuple(interp_inputs + [action_inputs])
        else:
            hidden, _ = self.encoder(interp_inputs)
            encoder_input = tuple(interp_inputs)

        estimate = self.forward(hidden).squeeze(1).sum()
        gradient = torch.autograd.grad(estimate, encoder_input, create_graph=True)[0]
        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = (torch.sum(gradient**2, dim=1) + self.EPSILON).sqrt()
        gradient_mag = torch.mean((safe_norm - 1) ** 2)
        return gradient_mag

    def compute_grad_pen(
        self, expert_state, expert_action, policy_state, policy_action, lambda_=10
    ):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.discriminator(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()

        return grad_pen

    def compute_loss(self, update_dict):
        d_out = self.forward(update_dict["all_obs"], update_dict["all_acs"])
        expert_out, policy_out = torch.chunk(d_out, chunks=2, dim=0)

        expert_loss = F.binary_cross_entropy_with_logits(
            expert_out, torch.ones(expert_out.size())
        )
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_out, torch.zeros(policy_out.size())
        )

        labels = torch.cat(
            [torch.zeros(expert_out.size()), torch.ones(policy_out.size())]
        )

        self.bce_loss = F.binary_cross_entropy_with_logits(d_out, labels)
        # self.grad_penalty = self.irm_penalty(d_out, labels)
        self.grad_penalty = self.irm_penalty(expert_out, torch.zeros(expert_out.size()))
        self.loss = self.bce_loss + self.irm_coeff * self.grad_penalty

        output_dict = {}
        output_dict["d_loss"] = self.bce_loss
        output_dict["grad_penalty"] = self.grad_penalty

        return output_dict

    def update(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()


class VAILDiscriminator(nn.Module):
    def __init__(
        self,
        env,
        layer_dims,
        lr,
        latent_dim,
        use_actions=True,
        use_cnn_base=False,
        use_vampprior=True,
        vae_type="TwoLayer",
        n_pseudo_inputs=50,
        pseudo_inputs_grad=False,
        demos=None,
        i_c=0.5,
        alpha_beta=1e-4,
        irm_coeff=0,
        l2_coeff=0,
        lip_coeff=0,
        bias=False,
    ):
        super(VAILDiscriminator, self).__init__()

        ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        if not ac_shapes:
            ac_shapes = [1]
        if use_actions:
            layer_dims = [ob_shapes[-1] + ac_shapes[-1]] + layer_dims
        else:
            layer_dims = [ob_shapes[-1]] + layer_dims
        ac_len = ac_shapes[0]

        self.layer_dims = layer_dims
        self.lr = lr
        self.latent_dim = latent_dim
        self.vae_type = vae_type
        self.n_pseudo_inputs = n_pseudo_inputs
        self.pseudo_inputs_grad = pseudo_inputs_grad
        self.i_c = i_c
        self.alpha_beta = alpha_beta
        self.use_vampprior = use_vampprior
        self.use_actions = use_actions
        self.irm_coeff = irm_coeff
        self.lip_coeff = lip_coeff

        self.use_cnn_base = use_cnn_base

        if use_cnn_base:
            self.base = MiniGridCNN(layer_dims, use_actions)
        else:
            self.base = nn.Sequential(
                torch.nn.Linear(self.layer_dims[0], self.layer_dims[1], bias),
                torch.nn.PReLU(),
            )

        self.num_inputs = self.layer_dims[1]
        self.nonlinear = torch.nn.PReLU()

        self.vp_dict = {}

        # encoder z_2: q(z1|x,z2)
        self.vp_dict["z_2"] = nn.Linear(self.num_inputs, self.latent_dim)
        self.vp_dict["z_2_mu"] = nn.Linear(self.latent_dim, self.latent_dim)
        self.vp_dict["z_2_logvar"] = nn.Linear(self.latent_dim, self.latent_dim)
        # encoder z_1: q(z1|x,z2)
        self.vp_dict["z_1_x"] = nn.Linear(self.num_inputs, self.latent_dim)
        self.vp_dict["z_1_z2"] = nn.Linear(self.latent_dim, self.latent_dim)
        self.vp_dict["z_1_joint"] = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.vp_dict["z_1_mu"] = nn.Linear(self.latent_dim, self.latent_dim)
        self.vp_dict["z_1_logvar"] = nn.Linear(self.latent_dim, self.latent_dim)
        # decoder z1: p(z1|z2)
        self.vp_dict["p_z1"] = nn.Linear(self.latent_dim, self.latent_dim)
        self.vp_dict["p_z1_mu"] = nn.Linear(self.latent_dim, self.latent_dim)
        self.vp_dict["p_z1_logvar"] = nn.Linear(self.latent_dim, self.latent_dim)
        # decoder x: p(x|z1,z2)
        self.vp_dict["p_x_z1"] = nn.Linear(self.latent_dim, self.latent_dim)
        self.vp_dict["p_x_z2"] = nn.Linear(self.latent_dim, self.latent_dim)
        self.vp_dict["p_x_joint"] = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.vp_dict["final"] = nn.Linear(self.latent_dim, 1)

        if self.use_vampprior:
            self.pseudo_2 = nn.Hardtanh(min_val=0.0, max_val=1.0)
            if demos is not None:
                # make sure that random numbers are distinct and do not repeat themselves
                indices = np.random.choice(
                    demos.shape[0], size=self.n_pseudo_inputs, replace=False
                )
                # indices = np.random.randint(0, self.demonstrations.shape[0], size=self.args.number_pseudo_inputs)
                self.pseudo_inputs = torch.nn.Parameter(
                    demos[indices], requires_grad=False
                )
                self.vp_dict["pseudo_1"] = nn.Linear(self.num_inputs, self.num_inputs)
            else:
                self.pseudo_inputs = torch.nn.Parameter(
                    torch.eye(self.n_pseudo_inputs),
                    requires_grad=self.pseudo_inputs_grad,
                )
                self.vp_dict["pseudo_1"] = nn.Linear(
                    self.n_pseudo_inputs, self.num_inputs
                )

        self._beta = torch.tensor(1.0, dtype=torch.float)

        if torch.cuda.is_available():
            self.base.cuda()
            # self.discriminator.cuda()
            for k, v in self.vp_dict.items():
                v.cuda()

        # self.module_list = nn.ModuleList([self.base, self.discriminator])
        base_params = list(self.base.parameters())
        for k, v in self.vp_dict.items():
            base_params.extend(list(v.parameters()))
        self.d_optimizer = Adam(base_params, lr=self.lr, weight_decay=l2_coeff)

        self.vp_dict["final"].weight.data.mul_(1.0)
        self.vp_dict["final"].bias.data.mul_(0.0)

    def get_reward(self, ob, ac):
        d_out, _ = self.forward(ob, ac)
        self.reward = -torch.squeeze(torch.log(d_out + 1e-8))
        return self.reward

    def normalize(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        x_out = (x - mean) / std
        return x_out

    def encoder_z2(self, x):
        h = self.nonlinear(self.vp_dict["z_2"](x))
        return self.vp_dict["z_2_mu"](h), self.vp_dict["z_2_logvar"](h)

    def encoder_z1(self, x, z2):
        h1 = self.nonlinear(self.vp_dict["z_1_x"](x))
        h2 = self.nonlinear(self.vp_dict["z_1_z2"](z2))
        x_z2 = torch.cat((h1, h2), -1)
        h = self.nonlinear(self.vp_dict["z_1_joint"](x_z2))
        return self.vp_dict["z_1_mu"](h), self.vp_dict["z_1_logvar"](h)

    def irm_penalty(self, logits, y):
        scale = torch.tensor(1.0).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def vail_discriminator(self, z1):
        h1 = self.nonlinear(self.vp_dict["p_z1"](z1))
        z1_p_mu = self.vp_dict["p_z1_mu"](h1)
        z1_p_logvar = self.vp_dict["p_z1_logvar"](h1)

        return torch.sigmoid(self.vp_dict["final"](h1)), z1_p_mu, z1_p_logvar

    def vp_discriminator(self, z1, z2):
        h1 = self.nonlinear(self.vp_dict["p_z1"](z1))  # TODO: z1 or z2?
        z1_p_mu = self.vp_dict["p_z1_mu"](h1)
        z1_p_logvar = self.vp_dict["p_z1_logvar"](h1)
        h2 = self.nonlinear(self.vp_dict["p_x_z1"](z1))
        h3 = self.nonlinear(self.vp_dict["p_x_z2"](z2))
        z1_z2 = torch.cat((h2, h3), -1)
        h = self.nonlinear(self.vp_dict["p_x_joint"](z1_z2))
        return torch.sigmoid(self.vp_dict["final"](h)), z1_p_mu, z1_p_logvar

    def forward(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            base_out = self.base(self.normalize(torch.cat([ob, ac], axis=-1)))
        else:
            base_out = self.base(self.normalize(ob))

        # if self.normalize_input:
        #    x = self.normalize(base_out)
        ## XXX: be careful with exponents in reparametrization -> normalize

        z2_mu, z2_logvar = self.encoder_z2(base_out)
        z2 = self.reparameterize(z2_mu, z2_logvar)
        z1_mu, z1_logvar = self.encoder_z1(base_out, z2)
        z1 = self.reparameterize(z1_mu, z1_logvar)
        if self.use_vampprior:
            z2_mu, z2_logvar = self.encoder_z2(base_out)
            z2 = self.reparameterize(z2_mu, z2_logvar)
            z1_mu, z1_logvar = self.encoder_z1(base_out, z2)
            z1 = self.reparameterize(z1_mu, z1_logvar)
            d_out, z1_p_mu, z1_p_logvar = self.vp_discriminator(z1, z2)
            pseudos_1 = self.vp_dict["pseudo_1"](self.pseudo_inputs)
            pseudos_2 = self.pseudo_2(pseudos_1)
            z2_p_mu, z2_p_logvar = self.encoder_z2(pseudos_2)
        else:
            z2_mu, z2_logvar = self.encoder_z2(base_out)
            z2 = self.reparameterize(z2_mu, z2_logvar)
            d_out, z1_p_mu, z1_p_logvar = self.vail_discriminator(z2)
            z2_p_mu, z2_p_logvar = None, None

        sample_dict = {}
        sample_dict["z1"] = z1
        sample_dict["z1_mu"] = z1_mu
        sample_dict["z1_logvar"] = z1_logvar
        sample_dict["z2"] = z2
        sample_dict["z2_mu"] = z2_mu
        sample_dict["z2_logvar"] = z2_logvar
        sample_dict["z1_p_mu"] = z1_p_mu
        sample_dict["z1_p_logvar"] = z1_p_logvar
        sample_dict["z2_p_mu"] = z2_p_mu
        sample_dict["z2_p_logvar"] = z2_p_logvar

        return d_out, sample_dict

    def redraw_pseudo_inputs_demonstrations(self, demos):
        # redrawing new samples from demonstrations
        indices = np.random.choice(
            demos.shape[0], size=self.opt.number_pseudo_inputs, replace=False
        )
        # indices = np.random.randint(0, self.demonstrations.shape[0], size=self.args.number_pseudo_inputs)
        pseudo_inputs = torch.nn.Parameter(demos[indices], requires_grad=False)
        return pseudo_inputs

    def compute_loss(self, update_dict):
        if self.vae_type == "TwoLayer":
            d_out_policy, policy_sample_dict = self.forward(
                update_dict["policy_obs"], update_dict["policy_acs"]
            )
            d_out_expert, expert_sample_dict = self.forward(
                update_dict["expert_obs"], update_dict["expert_acs"]
            )
            l_kld = vampprior_kld_twolayervae(
                policy_sample_dict, self.n_pseudo_inputs, self.use_vampprior
            )
            e_kld = vampprior_kld_twolayervae(
                expert_sample_dict, self.n_pseudo_inputs, self.use_vampprior
            )
        else:
            d_out_policy, policy_sample_dict = self.forward(
                update_dict["policy_obs"], update_dict["policy_acs"]
            )
            d_out_expert, expert_sample_dict = self.forward(
                update_dict["expert_obs"], update_dict["expert_acs"]
            )

            if self.use_vampprior:
                l_kld = vampprior_kld_vae(policy_sample_dict)
                e_kld = vampprior_kld_vae(
                    expert_sample_dict["z1_mu"],
                    expert_sample_dict["z1_logvar"],
                    expert_sample_dict["z1"],
                    expert_sample_dict["z1_p_mu"],
                    expert_sample_dict["z1_p_logvar"],
                )
            else:
                l_kld = gaussian_kld(
                    policy_sample_dict["z1_mu"], policy_sample_dict["z1_logvar"]
                )
                l_kld = l_kld.mean()
                e_kld = gaussian_kld(
                    expert_sample_dict["z1_mu"], expert_sample_dict["z1_logvar"]
                )
                e_kld = e_kld.mean()

        kld = 0.5 * (l_kld + e_kld)
        bottleneck_loss = kld - self.i_c

        labels = torch.cat(
            [torch.zeros(d_out_expert.size()), torch.ones(d_out_policy.size())]
        )
        d_out = torch.cat([d_out_expert, d_out_policy], dim=0)
        if torch.cuda.is_available():
            labels = labels.cuda()

        discriminator_loss = -(
            torch.log(d_out_expert + 1e-6) + torch.log(1.0 - d_out_policy + 1e-6)
        ).mean()

        with torch.no_grad():
            self._beta = torch.max(
                torch.tensor(0.0), self._beta + self.alpha_beta * bottleneck_loss
            )
        vdb_loss = discriminator_loss + self._beta * bottleneck_loss

        grad_penalty = self.irm_penalty(d_out, labels)

        output_dict = {}
        output_dict["d_loss"] = vdb_loss
        output_dict["grad_penalty"] = grad_penalty
        output_dict["vib_loss"] = bottleneck_loss.item()
        output_dict["vib_beta"] = self._beta.item()

        # log gradients
        # if self.vae_type == 'VAE':
        #     stats_dict['Policy/Norm_grad_last_layer'] = torch.norm(vdb.fc5.weight.grad).item()
        # else:
        #     stats_dict['Policy/Norm_grad_last_layer'] = torch.norm(vdb.final.weight.grad).item()

        # if self.use_vampprior:
        #     if args.pseudo_inputs_grad:
        #         stats_dict['Policy/Norm_grad_pseudoinputs'] = torch.norm(vdb.pseudo_input.grad).item()
        #     if args.attention and args.attention_type == 'nn':
        #         stats_dict['Policy/Norm_grad_attention_nn_parameter'] = torch.norm(vdb.attention_weights.grad).item()

        # if args.change_pseudo_inputs_dem:
        #    vdb.redraw_pseudo_inputs_demonstrations()

        return output_dict

    def update(self, loss):
        self.d_optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward()
        self.d_optimizer.step()

    def save(self, fname):
        torch.save(self.state_dict(), fname)


class AIRLDiscriminator(nn.Module):
    def __init__(
        self,
        env,
        layer_dims,
        lr,
        gamma,
        use_actions=True,
        use_cnn_base=False,
        irm_coeff=0,
        lip_coeff=0,
        l2_coeff=0,
        nonlin=torch.nn.PReLU(),
        bias=False,
    ):
        super(AIRLDiscriminator, self).__init__()

        ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        if not ac_shapes:
            ac_shapes = [1]
        if use_actions:
            layer_dims = [ob_shapes[-1] + ac_shapes[-1]] + layer_dims
        else:
            layer_dims = [ob_shapes[-1]] + layer_dims

        ac_len = ac_shapes[0]

        self.layer_dims = layer_dims
        self.lr = lr
        self.gamma = gamma
        self.use_actions = use_actions
        self.irm_coeff = irm_coeff
        self.lip_coeff = lip_coeff

        self.layer_dims = layer_dims
        self.use_cnn_base = use_cnn_base

        if self.lip_coeff > 0:
            f = lambda x: spectral_norm(x)
        else:
            f = lambda x: x

        # reward function g_\psi
        if use_cnn_base:
            self.base = MiniGridCNN(layer_dims, use_actions)
        else:
            self.base = nn.Sequential(
                f(torch.nn.Linear(self.layer_dims[0], self.layer_dims[1], bias)), nonlin
            )

        self.reward_layers = []
        for i in range(2, len(self.layer_dims)):
            self.reward_layers += [
                f(
                    torch.nn.Linear(
                        in_features=self.layer_dims[i - 1],
                        out_features=self.layer_dims[i],
                        bias=bias,
                    )
                ),
                nonlin,
            ]

        self.reward_layers += [
            f(
                torch.nn.Linear(
                    in_features=self.layer_dims[-1], out_features=1, bias=bias
                )
            )
        ]
        self.reward = nn.Sequential(*self.reward_layers)

        # shaping function h_\phi
        if use_cnn_base:
            self.base_v = MiniGridCNN(layer_dims, use_actions=False)
        else:
            self.base_v = nn.Sequential(
                f(torch.nn.Linear(ob_shapes[-1], self.layer_dims[1], bias)),
                torch.nn.PReLU(),
            )

        self.value_layers = []
        for i in range(2, len(self.layer_dims)):
            self.value_layers += [
                f(
                    torch.nn.Linear(
                        in_features=self.layer_dims[i - 1],
                        out_features=self.layer_dims[i],
                        bias=bias,
                    )
                ),
                nonlin,
            ]

        self.value_layers += [
            f(
                torch.nn.Linear(
                    in_features=self.layer_dims[-1], out_features=1, bias=bias
                )
            )
        ]
        self.value = nn.Sequential(*self.value_layers)

        if torch.cuda.is_available():
            self.base.cuda()
            self.base_v.cuda()
            self.reward.cuda()
            self.value.cuda()

        # self.module_list = nn.ModuleList([self.base, self.base_v, self.reward, self.value])

        self.d_optimizer = Adam(
            list(self.base.parameters())
            + list(self.base_v.parameters())
            + list(self.reward.parameters())
            + list(self.value.parameters()),
            lr=self.lr,
            weight_decay=l2_coeff,
        )

    def forward(self, ob, next_ob, ac, lprobs):
        # forward the nn models
        fitted_value_n = self.value(self.base_v(next_ob))
        fitted_value = self.value(self.base_v(ob))
        reward = self.get_reward(ob, ac)

        # calculate discriminator probability according to AIRL structure
        qfn = reward + self.gamma * fitted_value_n
        log_p_tau = torch.squeeze(qfn - fitted_value)
        # log probabilities of expert actions under policy
        log_q_tau = lprobs

        # log_pq = torch.log(torch.sum(torch.exp(torch.cat([log_p_tau, log_q_tau], dim=0))))
        # d_out = torch.exp(log_p_tau - log_pq)
        # TEST:
        d_out = torch.sigmoid(log_p_tau - log_q_tau)
        # d_out = torch.sigmoid(log_q_tau - log_p_tau)

        return reward, fitted_value, fitted_value_n, d_out

    def get_reward(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        # rew, v, v_n, d_out = self.forward(ob, next_ob, ac, lprobs) TODO??
        return self.reward(base_out)

    def get_value(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        # rew, v, v_n, d_out = self.forward(ob, next_ob, ac, lprobs) TODO??
        return self.value(base_out)

    def irm_penalty(self, logits, y):
        scale = torch.tensor(1.0).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def lip_penalty(self, update_dict):
        obs_epsilon = torch.rand(update_dict["policy_obs"].shape)
        interp_obs = (
            obs_epsilon * update_dict["policy_obs"]
            + (1 - obs_epsilon) * update_dict["expert_obs"]
        )
        interp_obs.requires_grad = True  # For gradient calculation

        obs_epsilon = torch.rand(update_dict["policy_obs_next"].shape)
        interp_obs_next = (
            obs_epsilon * update_dict["policy_obs_next"]
            + (1 - obs_epsilon) * update_dict["expert_obs_next"]
        )
        interp_obs_next.requires_grad = True  # For gradient calculation

        action_epsilon = torch.rand(update_dict["policy_acs"].shape)
        interp_acs = (
            action_epsilon * update_dict["policy_acs"]
            + (1 - action_epsilon) * update_dict["expert_acs"]
        )
        interp_acs.requires_grad = True
        encoder_input = [interp_obs, interp_acs, interp_obs_next]
        _, _, _, estimate = self.forward(
            interp_obs, interp_obs_next, interp_acs, update_dict["policy_lprobs"]
        )

        gradient = torch.autograd.grad(
            estimate.sum(), encoder_input, create_graph=True
        )[0]
        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = (torch.sum(gradient**2, dim=1) + 1e-8).sqrt()
        gradient_mag = torch.mean((safe_norm - 1) ** 2)

        return gradient_mag

    def compute_loss(self, update_dict):
        # Define log p(tau) = r(s,a) + gamma * V(s') - V(s)
        _, _, _, policy_estimate = self.forward(
            update_dict["policy_obs"],
            update_dict["policy_obs_next"],
            update_dict["policy_acs"],
            update_dict["policy_lprobs"],
        )
        _, _, _, expert_estimate = self.forward(
            update_dict["expert_obs"],
            update_dict["expert_obs_next"],
            update_dict["expert_acs"],
            update_dict["expert_lprobs"],
        )

        # label convention: experts 0, policy 1
        labels = torch.cat(
            [torch.zeros(expert_estimate.size()), torch.ones(policy_estimate.size())]
        )
        d_out = torch.cat([expert_estimate, policy_estimate], dim=0)
        if torch.cuda.is_available():
            labels = labels.cuda()

        discriminator_loss = -(
            torch.log(expert_estimate + 1e-6) + torch.log(1.0 - policy_estimate + 1e-6)
        ).mean()

        # loss_pi = -F.logsigmoid(-policy_estimate).mean()
        # loss_exp = -F.logsigmoid(expert_estimate).mean()
        # loss_disc = loss_pi + loss_exp

        # TODO: only compute w.r.t. expert_estimate!
        # what are labels -> need to be expert ids!!
        # -> Jensen Shannon? Mixture of experts?
        grad_penalty = self.irm_penalty(d_out, labels)
        # grad_penalty = self.irm_penalty(expert_estimate, torch.zeros(expert_estimate.size()))

        # lip_penalty = self.lip_penalty(update_dict)
        loss = discriminator_loss + self.irm_coeff * grad_penalty
        if self.irm_coeff > 1.0:
            loss /= self.irm_coeff

        output_dict = {}
        output_dict["total_loss"] = loss
        output_dict["d_loss"] = discriminator_loss
        output_dict["policy_estimate"] = policy_estimate
        output_dict["expert_estimate"] = expert_estimate
        output_dict["grad_penalty"] = grad_penalty
        # output_dict['lip_penalty'] = lip_penalty

        # return self.loss, discriminator_loss, policy_estimate.mean(), expert_estimate.mean(), self.grad_penalty
        return output_dict

    def update(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()

    def save(self, fname):
        torch.save(self.state_dict(), fname)


class WAILDiscriminator(nn.Module):
    def __init__(
        self,
        env,
        layer_dims,
        lr,
        use_actions=True,
        use_cnn_base=False,
        irm_coeff=0,
        lip_coeff=0,
        l2_coeff=0.0,
        epsilon=0.01,
        output_nonlin=torch.nn.Identity(),
        bias=False,
    ):
        super(WAILDiscriminator, self).__init__()

        ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        if not ac_shapes:
            ac_shapes = [1]
        if use_actions:
            layer_dims = [ob_shapes[-1] + ac_shapes[-1]] + layer_dims
        else:
            layer_dims = [ob_shapes[-1]] + layer_dims
        ac_len = ac_shapes[0]

        self.layer_dims = layer_dims
        self.lr = lr
        self.use_actions = use_actions
        self.irm_coeff = irm_coeff
        self.lip_coeff = lip_coeff
        self.epsilon = epsilon

        self.use_cnn_base = use_cnn_base

        if use_cnn_base:
            self.base = MiniGridCNN(layer_dims, use_actions)
        else:
            self.base = nn.Sequential(
                torch.nn.Linear(self.layer_dims[0], self.layer_dims[1], bias),
                torch.nn.LeakyReLU(),
            )

        self.discriminator_layers = []
        for i in range(2, len(layer_dims)):
            self.discriminator_layers += [
                torch.nn.Linear(
                    in_features=layer_dims[i - 1], out_features=layer_dims[i], bias=bias
                ),
                torch.nn.LeakyReLU(),
            ]

        self.discriminator_layers += [
            torch.nn.Linear(in_features=layer_dims[-1], out_features=1, bias=bias),
            output_nonlin,
        ]

        self.discriminator = nn.Sequential(*self.discriminator_layers)

        if torch.cuda.is_available():
            self.base.cuda()
            self.discriminator.cuda()

        # self.module_list = nn.ModuleList([self.base, self.discriminator])

        base_params = list(self.base.parameters())
        d_params = list(self.discriminator.parameters())
        base_params.extend(d_params)
        self.d_optimizer = Adam(base_params, lr=self.lr, weight_decay=l2_coeff)

    def forward(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        d_out = self.discriminator(base_out)
        return d_out

    def get_reward(self, ob, ac):
        self.reward = self.forward(ob, ac)
        return self.reward

    def irm_penalty(self, logits, y):
        scale = torch.tensor(1.0).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def compute_penalty(self, logits, y):
        scale = torch.tensor(1.0).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        g1 = autograd.grad(loss[0::2].mean(), [scale], create_graph=True)[0]
        g2 = autograd.grad(loss[1::2].mean(), [scale], create_graph=True)[0]
        return (g1 * g2).sum()

    # lipschitz penalty
    def lip_penalty(self, update_dict):
        interp_inputs = []
        for policy_input, expert_input in zip(
            update_dict["policy_obs"], update_dict["expert_obs"]
        ):
            obs_epsilon = torch.rand(policy_input.shape)
            interp_input = obs_epsilon * policy_input + (1 - obs_epsilon) * expert_input
            interp_input.requires_grad = True  # For gradient calculation
            interp_inputs.append(interp_input)
        if self.use_actions:
            action_epsilon = torch.rand(update_dict["policy_acs"].shape)

            dones_epsilon = torch.rand(update_dict["policy_dones"].shape)
            action_inputs = torch.cat(
                [
                    action_epsilon * update_dict["policy_acs"]
                    + (1 - action_epsilon) * update_dict["expert_acs"],
                    dones_epsilon * update_dict["policy_dones"]
                    + (1 - dones_epsilon) * update_dict["expert_dones"],
                ],
                dim=1,
            )
            action_inputs.requires_grad = True
            hidden, _ = self.encoder(interp_inputs, action_inputs)
            encoder_input = tuple(interp_inputs + [action_inputs])
        else:
            hidden, _ = self.encoder(interp_inputs)
            encoder_input = tuple(interp_inputs)

        estimate = self.forward(hidden).squeeze(1).sum()
        gradient = torch.autograd.grad(estimate, encoder_input, create_graph=True)[0]
        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = (torch.sum(gradient**2, dim=1) + self.EPSILON).sqrt()
        gradient_mag = torch.mean((safe_norm - 1) ** 2)
        return gradient_mag

    def compute_grad_pen(self, update_dict, lambda_=10):
        expert_state = update_dict["expert_obs"]
        expert_action = update_dict["expert_acs"]
        policy_state = update_dict["policy_obs"]
        policy_action = update_dict["policy_acs"]

        alpha = torch.rand(expert_state.size(0), 1)
        if self.use_actions:
            expert_data = torch.cat([expert_state, expert_action], dim=1)
            policy_data = torch.cat([policy_state, policy_action], dim=1)
        else:
            expert_data = expert_state
            policy_data = policy_state

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.discriminator(self.base(mixup_data))
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()

        return grad_pen

    def compute_loss(self, update_dict):
        expert_out = self.forward(update_dict["expert_obs"], update_dict["expert_acs"])
        policy_out = self.forward(update_dict["policy_obs"], update_dict["policy_acs"])

        lip_penalty = self.compute_grad_pen(update_dict)
        self.ot_loss = torch.sum(
            expert_out
            - policy_out
            - (
                1
                / (4 * self.epsilon)
                * (expert_out - policy_out - torch.norm(expert_out - policy_out))
            )
        )

        # self.grad_penalty = self.irm_penalty(d_out, labels)
        self.grad_penalty = self.irm_penalty(expert_out, torch.zeros(expert_out.size()))
        self.loss = (
            self.ot_loss
            + self.irm_coeff * self.grad_penalty
            + self.lip_coeff * lip_penalty
        )

        output_dict = {}
        output_dict["d_loss"] = self.ot_loss
        output_dict["grad_penalty"] = self.grad_penalty
        output_dict["lipschitz_penalty"] = lip_penalty

        return output_dict

    def update(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()


class SWILDiscriminator(nn.Module):
    def __init__(
        self,
        env,
        layer_dims,
        lr,
        batch_size,
        l2_coeff=0,
        irm_coeff=0,
        use_actions=True,
        use_cnn_base=False,
        n_proj=50,
        bias=False,
    ):
        super(SWILDiscriminator, self).__init__()

        ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        if not ac_shapes:
            ac_shapes = [1]
        if use_actions:
            layer_dims = [ob_shapes[-1] + ac_shapes[-1]] + layer_dims
        else:
            layer_dims = [ob_shapes[-1]] + layer_dims

        ac_len = ac_shapes[0]

        self.layer_dims = layer_dims
        self.lr = lr
        self.use_actions = use_actions

        self.layer_dims = layer_dims
        self.use_cnn_base = use_cnn_base

        if use_cnn_base:
            self.base = MiniGridCNN(layer_dims, use_actions)
        else:
            self.base = nn.Sequential(
                torch.nn.Linear(self.layer_dims[0], self.layer_dims[1], bias),
                torch.nn.PReLU(),
            )

        self.reward_layers = []
        for i in range(2, len(self.layer_dims)):
            self.reward_layers += [
                torch.nn.Linear(
                    in_features=self.layer_dims[i - 1],
                    out_features=self.layer_dims[i],
                    bias=bias,
                ),
                torch.nn.PReLU(),
            ]

        self.reward_layers += [
            torch.nn.Linear(
                in_features=self.layer_dims[-1], out_features=n_proj, bias=bias
            )
        ]
        self.reward = nn.Sequential(*self.reward_layers)

        if torch.cuda.is_available():
            self.base.cuda()
            self.reward.cuda()

        self.policy_obs = torch.randn([batch_size, *ob_shapes], requires_grad=True)
        self.policy_acs = torch.randn([batch_size, *ac_shapes], requires_grad=True)
        # self.module_list = nn.ModuleList([self.base, self.base_v, self.reward, self.value])
        self.d_optimizer = Adam(
            list(self.base.parameters()) + list(self.reward.parameters()),
            lr=self.lr,
            weight_decay=l2_coeff,
        )
        # self.d_optimizer = Adam([self.policy_obs, self.policy_acs], lr=self.lr)

    def forward(self, ob, ac, lprobs):
        # forward the nn models
        reward = self.get_reward(ob, ac)

        # get random projections based on reward
        return reward

    def proj(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        # rew, v, v_n, d_out = self.forward(ob, next_ob, ac, lprobs) TODO??
        # potentially use more sophisticated mechanism
        return self.reward(base_out)

    def get_reward(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        # rew, v, v_n, d_out = self.forward(ob, next_ob, ac, lprobs) TODO??
        # potentially use more sophisticated mechanism for averaging out projections
        return torch.mean(self.reward(base_out), -1)

    def gsw(self, obs_pi, acs_pi, obs_exp, acs_exp, random=True):
        """
        Calculates GSW between two empirical state-action distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        """
        # N,dn = X.shape
        # M,dm = Y.shape
        # assert dn==dm and M==N

        if random:
            self.reward.reset()

        # project slices
        pi_slices = self.proj(obs_pi, acs_pi)
        exp_slices = self.proj(obs_exp, acs_exp)

        # sort slices
        pi_slices_sorted = torch.sort(pi_slices, dim=0)[0]
        exp_slices_sorted = torch.sort(exp_slices, dim=0)[0]

        return torch.sqrt(torch.sum((pi_slices_sorted - exp_slices_sorted) ** 2))

    def max_gsw(self, X, Y, iterations=50, lr=1e-4):
        # N,dn = X.shape
        # M,dm = Y.shape
        # assert dn==dm and M==N

        # self.reward.reset()

        optimizer = torch.optim.Adam(self.reward.parameters(), lr=lr)
        total_loss = np.zeros((iterations,))
        for i in range(iterations):
            optimizer.zero_grad()
            loss = -self.gsw(X.to(self.device), Y.to(self.device), random=False)
            total_loss[i] = loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()

        return self.gsw(X.to(self.device), Y.to(self.device), random=False)

    def compute_loss(self, update_dict):
        # compute sliced Wasserstein distance here
        self.policy_obs = update_dict["policy_obs"]
        self.policy_acs = update_dict["policy_acs"]
        gsw_dist = self.gsw(
            self.policy_obs,
            self.policy_acs,
            update_dict["expert_obs"],
            update_dict["expert_acs"],
            random=False,
        )

        output_dict = {}
        output_dict["d_loss"] = gsw_dist
        output_dict["grad_penalty"] = torch.zeros([0])

        return output_dict

    def update(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()


class MEIRLDiscriminator(nn.Module):
    def __init__(
        self,
        env,
        layer_dims,
        lr,
        use_actions=False,
        use_cnn_base=False,
        l2_coeff=0,
        irm_coeff=0,
        lip_coeff=0,
        clamp_magnitude=10.0,
        bias=False,
    ):
        super(MEIRLDiscriminator, self).__init__()

        ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        if not ac_shapes:
            ac_shapes = [1]
        if use_actions:
            layer_dims = [ob_shapes[-1] + ac_shapes[-1]] + layer_dims
        else:
            layer_dims = [ob_shapes[-1]] + layer_dims
        ac_len = ac_shapes[0]

        self.layer_dims = layer_dims
        self.lr = lr
        self.use_actions = use_actions
        self.irm_coeff = irm_coeff
        self.lip_coeff = lip_coeff
        self.clamp_magnitude = clamp_magnitude

        self.use_cnn_base = use_cnn_base

        if use_cnn_base:
            self.base = MiniGridCNN(layer_dims, use_actions)
        else:
            self.base = nn.Sequential(
                torch.nn.Linear(self.layer_dims[0], self.layer_dims[1], bias),
                torch.nn.PReLU(),
            )

        self.discriminator_layers = []
        for i in range(2, len(layer_dims)):
            self.discriminator_layers += [
                torch.nn.Linear(
                    in_features=layer_dims[i - 1], out_features=layer_dims[i], bias=bias
                ),
                torch.nn.PReLU(),
            ]

        self.phi = nn.Sequential(*self.discriminator_layers)

        self.discriminator_layers += [
            torch.nn.Linear(in_features=layer_dims[-1], out_features=1, bias=bias)
        ]
        self.discriminator = nn.Sequential(*self.discriminator_layers)

        if torch.cuda.is_available():
            self.base.cuda()
            self.phi.cuda()
            self.discriminator.cuda()

        # self.module_list = nn.ModuleList([self.base, self.discriminator])

        base_params = list(self.base.parameters())
        d_params = list(self.discriminator.parameters())
        base_params.extend(d_params)
        self.d_optimizer = Adam(base_params, lr=self.lr, weight_decay=l2_coeff)

    def forward(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        phi = self.phi(base_out)
        d_out = self.discriminator(base_out)
        output = torch.clamp(
            d_out, min=-1.0 * self.clamp_magnitude, max=self.clamp_magnitude
        )
        return output, phi

    def get_reward(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        d_out = self.discriminator(base_out)
        self.reward = torch.clamp(
            d_out, min=-1.0 * self.clamp_magnitude, max=self.clamp_magnitude
        )
        return self.reward

    # lipschitz penalty
    def lip_penalty(self, update_dict):
        interp_inputs = []
        for policy_input, expert_input in zip(
            update_dict["policy_obs"], update_dict["expert_obs"]
        ):
            obs_epsilon = torch.rand(policy_input.shape)
            interp_input = obs_epsilon * policy_input + (1 - obs_epsilon) * expert_input
            interp_input.requires_grad = True  # For gradient calculation
            interp_inputs.append(interp_input)
        if self.use_actions:
            action_epsilon = torch.rand(update_dict["policy_acs"].shape)

            dones_epsilon = torch.rand(update_dict["policy_dones"].shape)
            action_inputs = torch.cat(
                [
                    action_epsilon * update_dict["policy_acs"]
                    + (1 - action_epsilon) * update_dict["expert_acs"],
                    dones_epsilon * update_dict["policy_dones"]
                    + (1 - dones_epsilon) * update_dict["expert_dones"],
                ],
                dim=1,
            )
            action_inputs.requires_grad = True
            hidden, _ = self.encoder(interp_inputs, action_inputs)
            encoder_input = tuple(interp_inputs + [action_inputs])
        else:
            hidden, _ = self.encoder(interp_inputs)
            encoder_input = tuple(interp_inputs)

        estimate = self.forward(hidden).squeeze(1).sum()
        gradient = torch.autograd.grad(estimate, encoder_input, create_graph=True)[0]
        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = (torch.sum(gradient**2, dim=1) + self.EPSILON).sqrt()
        gradient_mag = torch.mean((safe_norm - 1) ** 2)
        return gradient_mag

    def compute_grad_pen(
        self, expert_state, expert_action, policy_state, policy_action, lambda_=10
    ):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.discriminator(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()

        return grad_pen

    def compute_loss(self, update_dict):
        r_policy, self.phi_policy = self.forward(
            update_dict["policy_obs"], update_dict["policy_acs"]
        )
        r_expert, self.phi_expert = self.forward(
            update_dict["expert_obs"], update_dict["expert_acs"]
        )

        self.diff_loss = r_policy.mean() - r_expert.mean()
        self.grad_penalty = torch.norm(self.phi_expert - self.phi_policy)
        self.loss = self.diff_loss + self.irm_coeff * self.grad_penalty

        output_dict = {}
        output_dict["total_loss"] = self.loss
        output_dict["d_loss"] = self.diff_loss
        output_dict["grad_penalty"] = self.grad_penalty

        return output_dict

    def update(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()
