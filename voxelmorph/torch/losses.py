import torch
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()






import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MI(nn.Module):
    """
    Mutual information module.

    """
    def __init__(self, dimension, num_bins=64, sample_rate=1, kernel_sigma=1, eps=1e-8, **kwargs):
        super(MI, self).__init__()
        self.dimension = dimension
        self.num_bins = num_bins
        self.sample_rate = sample_rate
        self.kernel_sigma = kernel_sigma
        self._kernel_radius = math.ceil(2 * self.kernel_sigma)
        self.eps = eps
        self.kwargs = kwargs
        self.bk_threshold = self.kwargs.pop('bk_threshold', float('-inf'))
        self.normalized = self.kwargs.pop('normalized', False)
        if self.dimension == 2:
            self.scale_mode = 'bicubic'
        elif self.dimension == 3:
            self.scale_mode = 'trilinear'
        else:
            raise NotImplementedError

    def forward(self, source, target, mask=None, **kwargs):
        """
        Compute mutual information by Parzen window estimation.

        :param source: tensor of shape [B, 1, *vol_shape]
        :param target: tensor of shape [B, 1, *vol_shape]
        :param mask: tensor of shape [B, 1, *vol_shape]
        :return:
        """
        scale = kwargs.pop('scale', 0)
        num_bins = kwargs.pop('num_bins', self.num_bins)
        assert source.shape == target.shape
        if mask is None:
            mask = torch.ones_like(source)

        image_mask = mask.to(torch.bool) & (source > self.bk_threshold) & (target > self.bk_threshold)

        if scale > 0:
            source = F.interpolate(source, scale_factor=2 ** (- scale), mode=self.scale_mode)
            target = F.interpolate(target, scale_factor=2 ** (- scale), mode=self.scale_mode)
            image_mask = F.interpolate(image_mask.to(target.dtype), scale_factor=2 ** (- scale),
                                       mode='nearest').to(torch.bool)

        B = source.shape[0]

        masked_source = [torch.masked_select(source[i], mask=image_mask[i]) for i in range(B)]
        masked_target = [torch.masked_select(target[i], mask=image_mask[i]) for i in range(B)]

        sample_mask = torch.rand_like(masked_source[0]).le(self.sample_rate)
        sampled_source = [torch.masked_select(masked_source[i], mask=sample_mask) for i in range(B)]
        sampled_target = [torch.masked_select(masked_target[i], mask=sample_mask) for i in range(B)]

        source_max_v = torch.stack([s.amax().detach() for s in sampled_source])
        source_min_v = torch.stack([s.amin().detach() for s in sampled_source])
        target_max_v = torch.stack([t.amax().detach() for t in sampled_target])
        target_min_v = torch.stack([t.amin().detach() for t in sampled_target])
        source_bin_width = (source_max_v - source_min_v) / num_bins
        source_pad_min_v = source_min_v - source_bin_width * self._kernel_radius
        target_bin_width = (target_max_v - target_min_v) / num_bins
        target_pad_min_v = target_min_v - target_bin_width * self._kernel_radius
        bin_center = torch.arange(num_bins + 2 * self._kernel_radius, dtype=source.dtype, device=source.device)

        source_bin_pos = [(sampled_source[i] - source_pad_min_v[i]) / source_bin_width[i] for i in range(B)]
        target_bin_pos = [(sampled_target[i] - target_pad_min_v[i]) / target_bin_width[i] for i in range(B)]
        source_bin_idx = [p.floor().clamp(min=self._kernel_radius,
                                          max=self._kernel_radius + num_bins - 1).detach() for p in source_bin_pos]
        target_bin_idx = [p.floor().clamp(min=self._kernel_radius,
                                          max=self._kernel_radius + num_bins - 1).detach() for p in target_bin_pos]

        source_min_win_idx = [(i - self._kernel_radius + 1).to(torch.int64) for i in source_bin_idx]
        target_min_win_idx = [(i - self._kernel_radius + 1).to(torch.int64) for i in target_bin_idx]
        source_win_idx = [torch.stack([(smwi + r) for r in range(self._kernel_radius * 2)])
                          for smwi in source_min_win_idx]
        target_win_idx = [torch.stack([(tmwi + r) for r in range(self._kernel_radius * 2)])
                          for tmwi in target_min_win_idx]

        source_win_bin_center = [torch.gather(bin_center.unsqueeze(1).repeat(1, source_win_idx[i].size(1)),
                                              dim=0, index=source_win_idx[i])
                                 for i in range(B)]
        target_win_bin_center = [torch.gather(bin_center.unsqueeze(1).repeat(1, target_win_idx[i].size(1)),
                                              dim=0, index=target_win_idx[i])
                                 for i in range(B)]

        source_win_weight = [self._bspline_kernel(source_bin_pos[i].unsqueeze(0) - source_win_bin_center[i])
                             for i in range(B)]
        target_win_weight = [self._bspline_kernel(target_bin_pos[i].unsqueeze(0) - target_win_bin_center[i])
                             for i in range(B)]

        source_bin_weight = torch.stack([torch.stack([torch.sum(source_win_idx[i].eq(idx) * source_win_weight[i], dim=0)
                                                      for idx in range(num_bins + self._kernel_radius * 2)], dim=0)
                                         for i in range(B)])

        target_bin_weight = torch.stack([torch.stack([torch.sum(target_win_idx[i].eq(idx) * target_win_weight[i], dim=0)
                                                      for idx in range(num_bins + self._kernel_radius * 2)], dim=0)
                                         for i in range(B)]) 
        source_hist = source_bin_weight.sum(-1)
        target_hist = target_bin_weight.sum(-1)
        joint_hist = torch.bmm(source_bin_weight, target_bin_weight.transpose(1, 2))

        source_density = source_hist / source_hist.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        target_density = target_hist / target_hist.sum(dim=-1, keepdim=True).clamp(min=self.eps)

        joint_density = joint_hist / joint_hist.sum(dim=(1, 2), keepdim=True).clamp(min=self.eps)

        return source_density, target_density, joint_density

    def mi(self, target, source, mask=None, **kwargs):
        """
        (Normalized) mutual information

        :param source:
        :param target:
        :param mask:
        :return:
        """
        source_density, target_density, joint_density = self.forward(source, target, mask, **kwargs)
        source_entropy = - torch.sum(source_density * source_density.clamp(min=self.eps).log(), dim=-1)
        target_entropy = - torch.sum(target_density * target_density.clamp(min=self.eps).log(), dim=-1)
        joint_entropy = - torch.sum(joint_density * joint_density.clamp(min=self.eps).log(), dim=(1, 2))
        if self.normalized:
            return torch.mean(-(source_entropy + target_entropy) / joint_entropy)
        else:
            return torch.mean(-(source_entropy + target_entropy - joint_entropy))

    def je(self, source, target, mask=None, **kwargs):
        """
        Joint entropy H(S, T).

        :param source:
        :param target:
        :param mask:
        :return:
        """
        _, _, joint_density = self.forward(source, target, mask, **kwargs)
        joint_entropy = - torch.sum(joint_density * joint_density.clamp(min=self.eps).log(), dim=(1, 2)).mean()
        return joint_entropy

    def ce(self, source, target, mask=None, **kwargs):
        """
        Conditional entropy H(S | T) = H(S, T) - H(T).

        :param source:
        :param target:
        :param mask:
        :return:
        """
        _, target_density, joint_density = self.forward(source, target, mask, **kwargs)
        target_entropy = - torch.sum(target_density * target_density.clamp(min=self.eps).log(), dim=-1).mean()
        joint_entropy = - torch.sum(joint_density * joint_density.clamp(min=self.eps).log(), dim=(1, 2)).mean()
        return joint_entropy - target_entropy

    def _bspline_kernel(self, d):
        d /= self.kernel_sigma
        return torch.where(d.abs() < 1.,
                           (3. * d.abs() ** 3 - 6. * d.abs() ** 2 + 4.) / 6.,
                           torch.where(d.abs() < 2.,
                                       (2. - d.abs()) ** 3 / 6.,
                                       torch.zeros_like(d))
                           )




class MIND_loss(torch.nn.Module):
    """
        Local (over window) normalized cross correlation loss.
        """

    def __init__(self, win=None):
        super(MIND_loss, self).__init__()
        self.win = win

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

        # kernel size
        kernel_size = radius * 2 + 1

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        # compute patch-ssd
        ssd = F.avg_pool3d(rpad2(
            (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                           kernel_size, stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind /= mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

        return mind

    # def forward(self, y_pred, y_true):
    #     return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)
    def forward(self, y_true, y_pred):
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)