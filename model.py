import torch
import torch.nn as nn
import torch.nn.functional as F
import settings


class GaussianKernel(torch.nn.Module):
    def __init__(self, gamma=0.5):
        super(GaussianKernel, self).__init__()
        self.gamma = torch.nn.parameter.Parameter(torch.tensor(gamma), requires_grad=False)

    def forward(self, x, w, sigma):
        # Calculate L2-norm
        w = w.transpose(0, 3).reshape(1, 1, -1, w.size(0))
        l2 = x.unsqueeze(3) - w
        sigma = sigma.transpose(0, 3).reshape(1, 1, -1, sigma.size(0))

        out = torch.exp(-(l2 ** 2) / (sigma ** 2) / 2)
        out = out.sum(dim=2).clamp(min=1e-10).log() * 1e-2
        # print("out", out, out.shape)  # bs x kernel_size x out_channel
        out = torch.transpose(out, 1, 2)
        return out


class KernelConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_fn, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(KernelConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias, padding_mode)
        self.kernel_fn = kernel_fn
        kernel_num = out_channels
        mus = [1]
        bin_size = 2.0 / (kernel_num - 1)
        mus.append(1 - bin_size / 2)
        for i in range(1, kernel_num - 1):
            mus.append(mus[i] - bin_size)
        print("mus", mus)
        mus = torch.tensor(mus).view(kernel_num, 1, 1, 1).expand(-1, in_channels, self.kernel_size[0],
                                                                 self.kernel_size[1]).clone()
        self.weight.data = mus
        self.weight.requires_grad = False

        # sigmas = [0.001]
        sigmas = [0.1]
        sigmas += [0.1] * (kernel_num - 1)
        print("sigmas", sigmas)
        sigmas = torch.tensor(sigmas).view(kernel_num, 1, 1, 1). \
            expand(-1, in_channels, self.kernel_size[0], self.kernel_size[1]).cuda()
        self.sigmas = sigmas

        self.bia = None

    def __compute_shape(self, x):
        h = (x.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        w = (x.shape[3] - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
        return h, w

    def forward(self, x):
        x_unf = F.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride).transpose(1, 2)
        h, w = self.__compute_shape(x)
        out = self.kernel_fn(x_unf, self.weight, self.sigmas)
        return out.view(x.shape[0], self.out_channels, h, w)


class CroNDBase(nn.Module):
    def __init__(self, in_channels, matrix_size, channel1, kernel_size1, hidden_size, apply_dropout=False, conv_type="kc", esb=True, conv_tune=True):
        super(CroNDBase, self).__init__()
        self.mat_size_after_conv2 = (matrix_size - kernel_size1 + 1) ** 2 * channel1
        self.n_papers_per_author = in_channels
        in_channels = 1
        self.n_tokens = matrix_size
        self.apply_dropout = apply_dropout
        self.esb = esb
        channel1 = 21

        gaussian_kernel = GaussianKernel(gamma=5)

        if conv_type == "kc":
            self.conv1 = KernelConv2d(in_channels, channel1, kernel_size=(kernel_size1, kernel_size1), kernel_fn=gaussian_kernel, stride=(kernel_size1, kernel_size1))
            # self.conv1 = KernelConv2d(in_channels, channel1, kernel_size=(1, kernel_size1), kernel_fn=gaussian_kernel, stride=(1, kernel_size1))
            # self.conv1 = KernelConv2d(in_channels, channel1, kernel_size=(kernel_size1, 1), kernel_fn=gaussian_kernel, stride=(kernel_size1, 1))
        elif conv_type == "cnn":
            self.conv1 = nn.Conv2d(in_channels, channel1, kernel_size=(kernel_size1, kernel_size1), stride=(kernel_size1, kernel_size1))
            if not conv_tune:
                self.conv1.weight.requires_grad = False
                self.conv1.bias.requires_grad = False
        elif conv_type == "conna":
            self.conv1 = KernelConv2d(in_channels, channel1, kernel_size=(1, matrix_size), kernel_fn=gaussian_kernel,
                                      stride=(1, matrix_size))
        else:
            raise NotImplementedError

        self.mat_size_after_kc = (matrix_size - kernel_size1) // kernel_size1 + 1
        # self.mat_size_after_kc = matrix_size

        if conv_type == "kc" or conv_type == "cnn":
            self.left_size = self.mat_size_after_kc
        elif conv_type == "conna":
            self.left_size = matrix_size

        # self.attn_dim = channel1 * (self.mat_size_after_kc * self.n_papers_per_author)
        self.attn_dim = hidden_size

        self.attn_layer_coarse = nn.Linear(self.attn_dim, 1)

        self.dense_coarse_1 = nn.Linear(self.attn_dim, hidden_size)
        self.fc_add_1 = nn.Linear(17, hidden_size)
        self.fc_add_2 = nn.Linear(hidden_size, hidden_size)

        self.dense_coarse_2 = nn.Linear(hidden_size, 2)

        self.act = nn.Tanh()

        if conv_type == "kc" or conv_type == "cnn":
            self.linear_flatten = nn.Linear(self.mat_size_after_kc * self.n_papers_per_author * channel1, hidden_size)
        elif conv_type == "conna":
            self.linear_flatten = nn.Linear(self.n_papers_per_author * channel1,  hidden_size)
        # self.linear_flatten = nn.Linear(self.n_tokens * self.n_papers_per_author * channel1,  hidden_size)

        self.merge_fc = nn.Linear(4, 2)
        self.fc_add_3 = nn.Linear(hidden_size, 2)

        self.bn_layer_1 = nn.BatchNorm1d(self.attn_dim + 17 * 3)
        self.norm_hidden = nn.BatchNorm1d(self.attn_dim)
        self.norm_f_add = nn.BatchNorm1d(17)

        self.feature_attn = nn.Linear(hidden_size, 1)

        self.weight_feature = nn.Parameter(torch.zeros(size=(1,)), requires_grad=True)
        # self.weight_feature = nn.Parameter(torch.ones(size=(1, ))/2, requires_grad=True)

        if self.apply_dropout:
            self.drop_out_layer = nn.Dropout(p=0.5)

    def forward(self, sim_mat, f_add):
        bs = sim_mat.shape[0]

        mat_kc = []
        for i in range(self.n_papers_per_author):
            logit = self.conv1(sim_mat[:, i, :, :].unsqueeze(1))
            mat_kc.append(logit.unsqueeze(1))
        logits_coarse = torch.cat(mat_kc, dim=1)  # bs x n_papers_per_author x kernel_num x mat_size_kc x mat_size_kc
        logits_coarse = self.act(logits_coarse)

        logits_coarse = torch.transpose(logits_coarse, 1, 3)
        logits_coarse = logits_coarse.reshape(bs, self.left_size, -1)
        assert logits_coarse.shape[1] == self.left_size

        logits_coarse = self.act(self.linear_flatten(logits_coarse))

        attn_weights_coarse = self.attn_layer_coarse(logits_coarse)
        attn_weights_coarse = F.softmax(attn_weights_coarse, dim=1).expand(-1, -1, self.attn_dim)
        logits_coarse = (attn_weights_coarse * logits_coarse).sum(dim=1)

        logits_coarse = self.act(logits_coarse)
        f_add_proj = self.act(self.fc_add_1(f_add))

        if self.esb:
            if settings.data_source == "aminer":
                logits_coarse = (1 - torch.sigmoid(self.weight_feature)) * f_add_proj + torch.sigmoid(
                    self.weight_feature) * logits_coarse
            else:
                logits_coarse = torch.sigmoid(self.weight_feature) / 5 * f_add_proj + (
                            1 - torch.sigmoid(self.weight_feature) / 5) * logits_coarse
        else:
            logits_coarse = logits_coarse

        if self.apply_dropout:
            logits_coarse = self.drop_out_layer(logits_coarse)

        score_coarse = self.act(self.dense_coarse_1(logits_coarse))
        out_coarse = self.dense_coarse_2(score_coarse)

        out_fine = None

        f_add_hidden = self.act(self.fc_add_2(f_add_proj))
        f_add_out = self.fc_add_3(f_add_hidden)

        out = out_coarse

        return F.log_softmax(out, dim=1), out_coarse
