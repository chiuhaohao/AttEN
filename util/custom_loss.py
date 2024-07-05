import torch.nn.functional as F
import torch.nn as nn
import torch

class SFC_module(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all', base_temperature=1):
        super(SFC_module, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
          
    def kl_mvn(self, m0, m1, S0, S1):
        d_S0 = torch.diag_embed(S0)
        d_S1 = torch.diag_embed(S1)
        N = m0.shape[1]
        iS1 = torch.linalg.inv(d_S1)
        diff = m1 - m0
        tr_term = torch.matmul(iS1, d_S0) 
        tr_term = torch.sum(tr_term, dim=(1, 2))
        det_term  = torch.linalg.det(d_S1)/(torch.linalg.det(d_S0)+1e-8)
        quad_term = torch.sum((diff*diff) * (1/S1), axis=1)
        return .5 * (tr_term + det_term + quad_term - N)

    def symmetric_dist(self, mu_1, mu_2, var_1, var_2):
        return (self.kl_mvn(mu_1, mu_2, var_1, var_2) + self.kl_mvn(mu_2, mu_1, var_2, var_1))/2

    def pairwise_kl(self, mu_1, mu_2, var_1, var_2, batch_size):
        mu_a = torch.reshape(torch.tile(mu_1, [1, mu_2.shape[0]]), (mu_1.shape[0] * mu_2.shape[0], mu_1.shape[1]))
        mu_b = torch.tile(mu_2, [mu_1.shape[0], 1])

        var_a = torch.reshape(torch.tile(var_1, [1, var_2.shape[0]]), (var_1.shape[0] * var_2.shape[0], var_1.shape[1]))
        var_b = torch.tile(var_2, [var_1.shape[0], 1])
        return torch.reshape(self.symmetric_dist(mu_a, mu_b, var_a, var_b), (batch_size, batch_size))

    def forward(self, mu, var, labels=None, device=None):
        batch_size = mu.shape[0]
        labels = labels.contiguous().view(-1, 1)
        p_mask = torch.eq(labels, labels.T).float().to(device)
        
        logits_mask = torch.scatter(
            torch.ones_like(p_mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        
        p_mask = p_mask * logits_mask
        logits = F.sigmoid(-self.alpha*self.pairwise_kl(mu, mu, var, var, batch_size)+self.beta) * logits_mask
        return logits, p_mask

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        
    
    def kl_mvn(self, m0, m1, S0, S1):
        d_S0 = torch.diag_embed(S0)
        d_S1 = torch.diag_embed(S1)
        N = m0.shape[1]
        iS1 = torch.linalg.inv(d_S1)
        diff = m1 - m0
        tr_term = torch.matmul(iS1, d_S0) 
        tr_term = torch.sum(tr_term, dim=(1, 2))
        det_term  = torch.linalg.det(d_S1)/(torch.linalg.det(d_S0)+1e-8)
        quad_term = torch.sum((diff*diff) * (1/S1), axis=1)
        
        return .5 * (tr_term + det_term + quad_term - N)
    
    def normal_kl(self, mean1, mean2, logvar1, logvar2):
        """
        source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
        Compute the KL divergence between two gaussians.
        Shapes are automatically broadcasted, so batches can be compared to
        scalars, among other use cases.
        """
        tensor = None
        for obj in (mean1, logvar1, mean2, logvar2):
            if isinstance(obj, torch.Tensor):
                tensor = obj
                break
        assert tensor is not None, "at least one argument must be a Tensor"

        # Force variances to be Tensors. Broadcasting helps convert scalars to
        # Tensors, but it does not work for torch.exp().
        logvar1, logvar2 = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
            for x in (logvar1, logvar2)
        ]

        return 0.5 * torch.sum(
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2),
            dim=1
        )

        

    def symmetric_dist(self, mu_1, mu_2, var_1, var_2):
        return (self.kl_mvn(mu_1, mu_2, var_1, var_2) + self.normal_kl(mu_2, mu_1, var_2, var_1))/2

    def pairwise_kl(self, mu_1, mu_2, var_1, var_2, batch_size):
        mu_a = torch.reshape(torch.tile(mu_1, [1, mu_2.shape[0]]), (mu_1.shape[0] * mu_2.shape[0], mu_1.shape[1]))
        mu_b = torch.tile(mu_2, [mu_1.shape[0], 1])

        var_a = torch.reshape(torch.tile(var_1, [1, var_2.shape[0]]), (var_1.shape[0] * var_2.shape[0], var_1.shape[1]))
        var_b = torch.tile(var_2, [var_1.shape[0], 1])

        return torch.reshape(self.symmetric_dist(mu_a, mu_b, var_a, var_b), (batch_size, batch_size))

    def forward(self, mu, var, labels=None, mask=None, alpha=0.1, device=None):
        batch_size = mu.shape[0]
       

        labels = labels.contiguous().view(-1, 1)
        p_mask = torch.eq(labels, labels.T).float().to(device)
        n_mask = torch.ne(labels, labels.T).float().to(device)
        logits = F.sigmoid(self.pairwise_kl(mu, mu, var, var, batch_size))
        logits_mask = torch.scatter(
            torch.ones_like(p_mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        p_mask = p_mask * logits_mask
        n_mask = n_mask * logits_mask


        
        return torch.mean(logits*p_mask), torch.mean(logits*n_mask)
    
    

class FSE(nn.Module):
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(FSE, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        
    
    def kl_mvn(self, m0, m1, S0, S1):
        d_S0 = torch.diag_embed(S0)
        d_S1 = torch.diag_embed(S1)
        N = m0.shape[1]
        iS1 = torch.linalg.inv(d_S1)
        diff = m1 - m0
        tr_term = torch.matmul(iS1, d_S0) 
        tr_term = torch.sum(tr_term, dim=(1, 2))
        det_term  = torch.linalg.det(d_S1)/(torch.linalg.det(d_S0)+1e-8)
        quad_term = torch.sum((diff*diff) * (1/S1), axis=1)
        return .5 * (tr_term + det_term + quad_term - N)
        

    def symmetric_kl(self, mu_1, mu_2, var_1, var_2):
        return (self.kl_mvn(mu_1, mu_2, var_1, var_2) + self.kl_mvn(mu_2, mu_1, var_2, var_1))/2

    def pairwise_kl(self, mu_1, mu_2, var_1, var_2, batch_size):
        mu_a = torch.reshape(torch.tile(mu_1, [1, mu_2.shape[0]]), (mu_1.shape[0] * mu_2.shape[0], mu_1.shape[1]))
        mu_b = torch.tile(mu_2, [mu_1.shape[0], 1])

        var_a = torch.reshape(torch.tile(var_1, [1, var_2.shape[0]]), (var_1.shape[0] * var_2.shape[0], var_1.shape[1]))
        var_b = torch.tile(var_2, [var_1.shape[0], 1])

        return torch.reshape(self.symmetric_kl(mu_a, mu_b, var_a, var_b), (batch_size, batch_size))

    def forward(self, mu, var, labels=None, mask=None, alpha=0.1, device=None):
        batch_size = mu.shape[0]
        labels = labels.contiguous().view(-1, 1)
        p_mask = torch.eq(labels, labels.T).float().to(device)
        n_mask = torch.ne(labels, labels.T).float().to(device)
        logits = F.sigmoid(self.pairwise_kl(mu, mu, var, var, batch_size))
        logits_mask = torch.scatter(
            torch.ones_like(p_mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        p_mask = p_mask * logits_mask
        n_mask = n_mask * logits_mask

        return torch.mean(logits*p_mask), torch.mean(logits*n_mask), torch.sum(logits*p_mask).detach(), torch.sum(logits*n_mask).detach()
    
class MMDLoss(nn.Module):
    def __init__(self, w_m, sigma, num_groups, num_classes, kernel):
        super(MMDLoss, self).__init__()
        self.w_m = w_m
        self.sigma = sigma
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.kernel = kernel

    def forward(self, f_s, f_t, groups, labels, jointfeature=False):
        if self.kernel == 'poly':
            student = F.normalize(f_s.view(f_s.shape[0], -1), dim=1)
            teacher = F.normalize(f_t.view(f_t.shape[0], -1), dim=1).detach()
        else:
            student = f_s.view(f_s.shape[0], -1)
            teacher = f_t.view(f_t.shape[0], -1).detach()

        mmd_loss = 0

        if jointfeature:
            K_TS, sigma_avg = self.pdist(teacher, student,
                              sigma_base=self.sigma, kernel=self.kernel)
            K_TT, _ = self.pdist(teacher, teacher, sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)
            K_SS, _ = self.pdist(student, student,
                              sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

            mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        else:
            with torch.no_grad():
                _, sigma_avg = self.pdist(teacher, student, sigma_base=self.sigma, kernel=self.kernel)

            for c in range(self.num_classes):
                if len(teacher[labels==c]) == 0:
                    continue
                for g in range(self.num_groups):
                    if len(student[(labels==c) * (groups == g)]) == 0:
                        continue
                    K_TS, _ = self.pdist(teacher[labels == c], student[(labels == c) * (groups == g)],
                                                 sigma_base=self.sigma, sigma_avg=sigma_avg,  kernel=self.kernel)
                    K_SS, _ = self.pdist(student[(labels == c) * (groups == g)], student[(labels == c) * (groups == g)],
                                         sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

                    K_TT, _ = self.pdist(teacher[labels == c], teacher[labels == c], sigma_base=self.sigma,
                                         sigma_avg=sigma_avg, kernel=self.kernel)

                    mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        loss = (1/2) * self.w_m * mmd_loss

        return loss

    @staticmethod
    def pdist(e1, e2, eps=1e-12, kernel='rbf', sigma_base=1.0, sigma_avg=None):
        if len(e1) == 0 or len(e2) == 0:
            res = torch.zeros(1)
        else:
            if kernel == 'rbf':
                e1_square = e1.pow(2).sum(dim=1)
                e2_square = e2.pow(2).sum(dim=1)
                prod = e1 @ e2.t()
                res = (e1_square.unsqueeze(1) + e2_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
                res = res.clone()
                sigma_avg = res.mean().detach() if sigma_avg is None else sigma_avg
                res = torch.exp(-res / (2*(sigma_base)*sigma_avg))
            elif kernel == 'poly':
                res = torch.matmul(e1, e2.t()).pow(2)

        return res, sigma_avg