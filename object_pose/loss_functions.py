import torch
import torch_bingham
from torch import nn
from torch.nn import functional as F


class bingham_KL_divergence(torch.nn.Module):
    """
    Bingham KL divergence
    """
    def __init__(self):
        super(bingham_KL_divergence, self).__init__()
    
    def bingham_cross_entropy(self, V1, F1, dF1, V2, Z2, F2, B1_mode):
        batch_size = F1.size(0)
        d = V1.size(2)

        # compute the full, transposed V1
        V1_ft = torch.zeros((batch_size, d, d), device=F1.device)
        V1_ft[:, :, 0] = B1_mode
        V1_ft[:, :, 1:] = V1.permute(0, 2, 1)
        
        # compute the inverse of V1_ft
        V1_ft_inv = torch.inverse(V1_ft)
        
        # rotate B2 into B1's coordinate frame
        A = torch.zeros((batch_size, d-1, d), device=F1.device)
        for i in range(d-1):
            for j in range(d):
                A[:, i, j] = torch.sum(V1_ft_inv[:, j, :] * V2[:, i, :], dim=1).pow(2)
        
                
        # compute H(B1, B2)
        H = torch.log(F2)       
        for i in range(d-1):
            H_i = A[:, i, 0]
            for j in range(1, d):
                H_i = H_i + (A[:, i, j] - A[:, i, 0]) * (dF1[:, j-1] / F1)
            H_i = H_i * Z2[:, i]
            H = H - H_i
        
        return H

    def forward(self, q1, z1, q2, z2):
        
        [V1, Z1, F1, dF1, V2, Z2, F2, B1_mode] = torch_bingham.bingham_cross_entropy_paras(q1, z1, q2, z2)
        
        # Z1 Z2 B1_mode
        Z1 = torch.flip(z1, dims=[1])

        Z2 = torch.flip(z2, dims=[1])

        B1_mode = q1

        # V1
        q1_col1 = q1[:, 0].unsqueeze(1)
        q1_col2 = q1[:, 1].unsqueeze(1)
        q1_col3 = q1[:, 2].unsqueeze(1)
        q1_col4 = q1[:, 3].unsqueeze(1)

        V1_part1 = torch.cat([q1_col4, q1_col3, -q1_col2, -q1_col1], dim=1).unsqueeze(1)
        V1_part2 = torch.cat([-q1_col3, q1_col4, q1_col1, -q1_col2], dim=1).unsqueeze(1)
        V1_part3 = torch.cat([-q1_col2, q1_col1, -q1_col4, q1_col3], dim=1).unsqueeze(1)
    
        V1 = torch.cat([V1_part1, V1_part2, V1_part3], dim=1)
        
        # V2
        q2_col1 = q2[:, 0].unsqueeze(1)
        q2_col2 = q2[:, 1].unsqueeze(1)
        q2_col3 = q2[:, 2].unsqueeze(1)
        q2_col4 = q2[:, 3].unsqueeze(1)

        V2_part1 = torch.cat([q2_col4, q2_col3, -q2_col2, -q2_col1], dim=1).unsqueeze(1)
        V2_part2 = torch.cat([-q2_col3, q2_col4, q2_col1, -q2_col2], dim=1).unsqueeze(1)
        V2_part3 = torch.cat([-q2_col2, q2_col1, -q2_col4, q2_col3], dim=1).unsqueeze(1)
    
        V2 = torch.cat([V2_part1, V2_part2, V2_part3], dim=1)
        
        # cross_entropy
        cross_entropy_b1b2 = self.bingham_cross_entropy(V1, F1, dF1, V2, Z2, F2, B1_mode)
        
        entropy_b1 = torch.log(F1) - torch.sum(Z1*dF1, -1) / F1
        
        KL = cross_entropy_b1b2 - entropy_b1
        
        return KL
    
class RWTA_MB_KL_Loss(nn.Module):
    def __init__(self, nm, epsilon=0.95, use_l1=False):
        super(RWTA_MB_KL_Loss, self).__init__()
        self.nm = nm
        self.epsilon = epsilon
        self.use_l1 = use_l1
        
        self.KL_b = bingham_KL_divergence()

    def forward(self, pred_q, Zbatch, weights, gt_q):
        pred_q = pred_q.reshape(-1, 4)
        gt_q = gt_q.reshape(-1, 1, 4).repeat([1, self.nm, 1]).reshape(-1, 4)
        
        
        gt_Z = Zbatch.detach() 
        p = self.KL_b(pred_q, Zbatch, gt_q, gt_Z)
        p = p.reshape([-1, self.nm])

        if self.use_l1:
            l1 = torch.abs(pred_q - gt_q).sum(-1).reshape(-1, self.nm)
            best_indices = l1.argmin(1)
        else:
            best_indices = p.argmin(1)

        all_assignments = torch.mean(p)
        best_assignment = torch.mean(p[torch.arange(p.shape[0]), best_indices])

        rwta_loss = (self.epsilon - 1 / self.nm) * best_assignment + (
                1 - self.epsilon) * 1 / self.nm * all_assignments

        weights = F.softmax(weights, dim=-1)
        
        # DME
        mb_loss = - torch.mean(torch.logsumexp(torch.log(weights) - p, dim=-1))

        return rwta_loss, mb_loss

class RWTA_CE_KL_Loss(nn.Module):
    def __init__(self, nm, epsilon=0.95, use_l1=False):
        super(RWTA_CE_KL_Loss, self).__init__()
        self.nm = nm
        self.epsilon = epsilon
        self.use_l1 = use_l1
        
        self.KL_b = bingham_KL_divergence()

    def forward(self, pred_q, Zbatch, weights, gt_q):
        pred_q = pred_q.reshape(-1, 4)
        gt_q = gt_q.reshape(-1, 1, 4).repeat([1, self.nm, 1]).reshape(-1, 4)

        gt_Z = Zbatch.detach() 
        p = self.KL_b(pred_q, Zbatch, gt_q, gt_Z)
        p = p.reshape([-1, self.nm])

        if self.use_l1:
            l1 = torch.abs(pred_q - gt_q).sum(-1).reshape(-1, self.nm)
            best_indices = l1.argmin(1)
        else:
            best_indices = p.argmin(1)

        all_assignments = torch.mean(p)
        best_assignment = torch.mean(p[torch.arange(p.shape[0]), best_indices])

        rwta_loss = (self.epsilon - 1 / self.nm) * best_assignment + (
                1 - self.epsilon) * 1 / self.nm * all_assignments
        
        # ACE
        ce_wgt = torch.clone(weights.detach())
        ce_wgt[ce_wgt > 0.01] = (1 / ce_wgt[ce_wgt > 0.01])
        ce_wgt = ce_wgt.mean(0)
        ce_loss = F.cross_entropy(weights, best_indices, weight = ce_wgt)

        return rwta_loss, ce_loss
