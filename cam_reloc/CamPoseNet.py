import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from resnet import resnet34


class Discriminator(nn.Module):
    def __init__(self, z1size, z2size):
        super(Discriminator, self).__init__()
        
        self.dfc3_1 = nn.Linear(z1size, 2048)
        self.bn3_1 = nn.BatchNorm1d(2048)
        
        self.dfc3_2 = nn.Linear(z2size, 2048)
        self.bn3_2 = nn.BatchNorm1d(2048)
        
        self.dfc2 = nn.Linear(2048, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        
        self.dfc1 = nn.Linear(4096,256 * 6 * 6)
        self.bn1 = nn.BatchNorm1d(256*6*6)
        
        self.upsample2=nn.Upsample(scale_factor=2)
        
        self.dconv4 = nn.ConvTranspose2d(256, 256, 3, padding = 0)
        self.dconv3 = nn.ConvTranspose2d(256, 384, 3, padding = 1)
        self.dconv2 = nn.ConvTranspose2d(384, 192, 3, padding = 1)
        self.dconv1 = nn.ConvTranspose2d(192, 64, 5, padding = 2)
        
        self.dconv0 = nn.ConvTranspose2d(64, 3, 12, stride = 4, padding = 4)
        
    
    def forward(self, x1, x2): # [20, 200]

        x1 = self.dfc3_1(x1) # [20, 2048]
        x1 = F.relu(self.bn3_1(x1))
        
        x2 = self.dfc3_2(x2) # [20, 2048]
        x2 = F.relu(self.bn3_2(x2))
        
        x = x1 + x2
        
        x = self.dfc2(x) # [20, 4096]
        x = F.relu(self.bn2(x))
        
        x = self.dfc1(x) # [20, 256*6*6]
        x = F.relu(self.bn1(x))
        
        x = x.view(x.shape[0],256,6,6) # [20, 256, 6, 6]
        
        x = self.upsample2(x)
        
        x = F.relu(self.dconv4(x)) # [20, 256, 12, 12]
        
        x = F.relu(self.dconv3(x)) # [20, 384, 14, 14]
        
        x = F.relu(self.dconv2(x)) # [20, 192, 14, 14]
        
        x = self.upsample2(x) # [20, 192, 28, 28]

        x = F.relu(self.dconv1(x)) # [20, 64, 28, 28]
        
        x = self.upsample2(x) # [20, 64, 56, 56]
        
        x = self.dconv0(x) # [20, 3, 224, 224]

        x = F.sigmoid(x) # [20, 3, 224, 224]
        return x
    
class CamPoseNet(nn.Module):

    def __init__(self, num_coeff, stage, base='ResNet', loss='CE', pretrained=False):
        """Initialized the network architecture.

            Parameters
            ----------
            :param num_coeff : the number of components in the mixture model
            :param base: the network architecture to use, currently 'ResNet' or 'Inception'
            :param pretrained: True if a pretrained network from Pytorch should be used, False otherwise

            Raises
            ------
            NotImplementedError
                If a specific network architecture is not implemented.
        """

        super(CamPoseNet, self).__init__()
        self.num_coeff = num_coeff
        self.base = base
        self.stage = stage
        self.loss_type = loss
        
        self.num_rp = 50

        if self.base == 'ResNet':
            self.model = resnet34(pretrained=pretrained)
            fe_out_planes = self.model.fc.in_features
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)
            self.model.fc = nn.Linear(fe_out_planes, 2048)

        elif self.base == 'Inception':
            self.model = models.inception_v3(pretrained=pretrained, aux_logits=False)
            self.model.fc = nn.Linear(2048, 2048)
        else:
            raise NotImplementedError()
            

        self.fc_xyz = nn.Sequential(nn.Linear(2048, 3 * self.num_coeff))
        self.fc_pose = nn.Sequential(nn.Linear(2048, 4 * self.num_coeff))
        
        self.fc_Z = nn.Sequential(nn.Linear(2048, 3 * self.num_coeff))
        self.fc_coeff = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                            nn.Linear(512, self.num_coeff), nn.BatchNorm1d(self.num_coeff))
   
        self.fc_std = nn.Sequential(nn.Linear(2048, 3 * self.num_coeff))

        # initialize layers
        if pretrained:
            init_modules = [self.fc_pose, self.fc_xyz, 
                            self.fc_Z, self.fc_std, self.fc_coeff]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
                    
        self.discriminator = Discriminator(4*self.num_coeff, 3*self.num_coeff)
        
        self.generator = Discriminator(4*self.num_rp, 3*self.num_rp)

    def bingham_rj(self, E, K):
        lam = -K

        qa = lam.shape[1]
        mu = torch.zeros(qa, device=E.device)
        sigacginv = 1 + 2 * lam
        SigACG = torch.sqrt(1 / (1+2*lam))

        X = torch.zeros_like(K, device=E.device)
        rj = torch.zeros(lam.shape[0], dtype=torch.bool, device=E.device)

        while not rj.all():
            indx = torch.where(rj==0)
            yp = torch.normal(mu, SigACG[indx])
            y = yp / torch.sqrt(torch.sum(yp**2, 1, keepdim=True))
            X[indx] = y

            lratio = -torch.sum(y**2 * lam[indx], 1) - qa/2 * torch.log(torch.tensor([qa], device=E.device)) + 0.5*(qa-1) + qa/2 * torch.log(torch.sum(y**2 * sigacginv[indx], 1))
            rj[indx] = torch.log(torch.rand(len(lratio), device=E.device)) < lratio

        return torch.bmm(E.transpose(1, 2), X.unsqueeze(2)).squeeze()
        
    def reparameterize_rj(self, normalized_q_output, Zbatch):
        q0 = normalized_q_output[:, 0].unsqueeze(1)  # [800, 1]
        q1 = normalized_q_output[:, 1].unsqueeze(1)  # [800, 1]
        q2 = normalized_q_output[:, 2].unsqueeze(1)  # [800, 1]
        q3 = normalized_q_output[:, 3].unsqueeze(1)  # [800, 1]
            
        row1 = torch.cat([q0, -q1, -q2, q3], dim=1)
        row2 = torch.cat([q1, q0, q3, q2], dim=1)
        row3 = torch.cat([q2, -q3, q0, -q1], dim=1)
        row4 = torch.cat([q3, q2, -q1, -q0], dim=1)
            
        z_Q = torch.stack([row1, row2, row3, row4], dim=1)
        zero_tensor = torch.ones(Zbatch.size(0), 1, dtype=Zbatch.dtype, device=Zbatch.device)*(-0.000001)
        Z_expanded = torch.cat((zero_tensor, Zbatch), dim=1)
        
        assert torch.all(Z_expanded < 0)
        z = self.bingham_rj(z_Q, Z_expanded)
            
        return z
    
    def recon(self, pose, xyz):
        #pose [batchsize, 4]
        #xyz [batchsize, 3]
        
        xyz_rp = xyz.unsqueeze(1).expand(-1, self.num_rp, -1).reshape(-1, self.num_rp * 3)
        
        std_rp = torch.ones_like(xyz_rp) * 0.0001
        
        nwpqr_rp = pose.unsqueeze(1).expand(-1, self.num_rp, -1).reshape(-1, self.num_rp * 4)
        
        lambdas_rp = torch.ones_like(xyz_rp) * -500
        lambdas_rp = lambdas_rp.reshape(-1, self.num_rp, 3)
        lambdas_rp[:, :, 1] = lambdas_rp[:, :, 1] - 0.000001
        lambdas_rp[:, :, 2] = lambdas_rp[:, :, 2] - 0.000002
        lambdas_rp = lambdas_rp.reshape(-1, self.num_rp * 3)
        
        # Reparameterize 
        z2=torch.randn_like(xyz_rp)*torch.sqrt(std_rp+1e-8)+xyz_rp        
        
        z1 = self.reparameterize_rj(nwpqr_rp.reshape(-1, 4), lambdas_rp.reshape(-1, 3))
        z1_reshaped = z1.reshape(-1, self.num_rp*4) # [20, 4*nm]
        
        x_gen = self.generator(z1_reshaped, z2)
        
        return x_gen
    
    def forward(self, x_img, train=False): # [20, 3, 224, 224]
        x, indices, res_l1, res_l2, res_l3, res_4 = self.model(x_img) # [20, 2048]
        x = F.relu(x)
        
        wpqr = self.fc_pose(x) #[20, 4*nm]
        xyz = self.fc_xyz(x) #[20, 3*nm]
     
        # quaternion normalization
        nwpqr = wpqr.reshape(-1, self.num_coeff, 4)
        nwpqr = F.normalize(nwpqr, dim=2)     
        nwpqr = nwpqr.reshape(-1, self.num_coeff * 4) # [20, 4*nm]
  

        Z = self.fc_Z(x) # [20, 3*nm]
        Z = F.softplus(Z)
        dZ = Z.reshape(-1,3)

        # compute ordered lambdas from predicted offsets
        Z0 = dZ[:, 0:1]
        Z1 = Z0 + dZ[:, 1:2]
        Z2 = Z1 + dZ[:, 2:3]
        Zbatch = torch.cat([Z0, Z1, Z2], dim=1)
        lambdas = -1 * Zbatch.clamp(1e-12, 900)
        lambdas = lambdas.reshape(-1, self.num_coeff*3) # [20, 3*nm]

        std = self.fc_std(x) # [20, 3*nm]
        std = F.softplus(std)
        
        coeff_q = self.fc_coeff(x)
        if self.loss_type == 'CE':
            coeff_q = F.relu(coeff_q)
        
        
        # Reparameterize
        z2=torch.randn_like(xyz)*torch.sqrt(std+1e-8)+xyz        
        
        z1 = self.reparameterize_rj(nwpqr.reshape(-1, 4), lambdas.reshape(-1, 3))
        z1_reshaped = z1.reshape(-1, self.num_coeff*4) # [20, 4*nm]
        
        x_dis = self.discriminator(z1_reshaped, z2)
        
        dis_paras = [nwpqr, lambdas, coeff_q, xyz, std, z1_reshaped]
        
        
        #############################################################
        
        # Transfer xyz, nwpqr for reparameterize
        _, max_indices = coeff_q.max(dim=1)
        
        xyz_rp = xyz.reshape(-1, self.num_coeff, 3)
        xyz_rp = xyz_rp[torch.arange(xyz_rp.shape[0]), max_indices]
        xyz_rp = xyz_rp.unsqueeze(1).expand(-1, self.num_rp, -1).reshape(-1, self.num_rp * 3)
        
        std_rp = torch.ones_like(xyz_rp) * 0.0001
        
        nwpqr_rp = nwpqr.reshape(-1, self.num_coeff, 4)
        nwpqr_rp = nwpqr_rp[torch.arange(nwpqr_rp.shape[0]), max_indices]
        nwpqr_rp = nwpqr_rp.unsqueeze(1).expand(-1, self.num_rp, -1).reshape(-1, self.num_rp * 4)
        
        lambdas_rp = torch.ones_like(xyz_rp) * -500
        lambdas_rp = lambdas_rp.reshape(-1, self.num_rp, 3)
        lambdas_rp[:, :, 1] = lambdas_rp[:, :, 1] - 0.000001
        lambdas_rp[:, :, 2] = lambdas_rp[:, :, 2] - 0.000002
        lambdas_rp = lambdas_rp.reshape(-1, self.num_rp * 3)
        
        # Reparameterize 
        z2=torch.randn_like(xyz_rp)*torch.sqrt(std_rp+1e-8)+xyz_rp        
        
        z1 = self.reparameterize_rj(nwpqr_rp.reshape(-1, 4), lambdas_rp.reshape(-1, 3))
        z1_reshaped = z1.reshape(-1, self.num_rp*4) # [20, 4*nm]
        
        x_gen = self.generator(z1_reshaped, z2)
   
        gen_paras = [nwpqr, lambdas, coeff_q, xyz, std, z1_reshaped]
        

        return dis_paras, gen_paras, x_gen, x_dis


