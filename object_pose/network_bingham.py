import torch
from torch import nn
from torch.nn import functional as F

class conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bn=True,
                 activation_fn=nn.ReLU(inplace=True)):
        super(conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups,
                              bias=not bn)
        if bn:
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.bn = None

        if activation_fn:
            self.activation = activation_fn
        else:
            self.activation = None

    def forward(self, input):
        output = self.conv(input)
        if self.bn:
            output = self.bn(output)
        if self.activation:
            output = self.activation(output)
        return output
    
class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bn=True,
                 activation_fn=nn.ReLU(inplace=True)):
        super(conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups,
                              bias=not bn)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if activation_fn:
            self.activation = activation_fn
        else:
            self.activation = None

    def forward(self, input):
        output = self.conv(input)
        if self.bn:
            output = self.bn(output)
        if self.activation:
            output = self.activation(output)
        return output
    
class convTrans2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bn=True,
                 activation_fn=nn.ReLU(inplace=True)):
        super(convTrans2d, self).__init__()
        self.convTran = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups,
                              bias=not bn)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if activation_fn:
            self.activation = activation_fn
        else:
            self.activation = None

    def forward(self, input):
        output = self.convTran(input)
        if self.bn:
            output = self.bn(output)
        if self.activation:
            output = self.activation(output)
        return output

class convTrans1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bn=True,
                 activation_fn=nn.ReLU(inplace=True)):
        super(convTrans1d, self).__init__()
        self.convTran = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups,
                              bias=not bn)
        if bn:
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.bn = None

        if activation_fn:
            self.activation = activation_fn
        else:
            self.activation = None

    def forward(self, input):
        output = self.convTran(input)
        if self.bn:
            output = self.bn(output)
        if self.activation:
            output = self.activation(output)
        return output

class dense(nn.Module):
    def __init__(self, in_features, out_features, bn=True, bias=True, activation_fn=nn.ReLU(inplace=True)):
        super(dense, self).__init__()

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        if bn:
            self.bn = nn.BatchNorm1d(out_features)
        else:
            self.bn = None

        if activation_fn:
            self.activation = activation_fn
        else:
            self.activation = None

    def forward(self, input):
        output = self.fc(input)
        if self.bn:
            output = self.bn(output)
        if self.activation:
            output = self.activation(output)
        return output
    
class discriminator(nn.Module):
    def __init__(self, in_features, out_features):
        super(discriminator, self).__init__()
        
        self.out_features = out_features
        
        self.fc = dense(in_features, 512, bn=True)
        
        self.fc1 = dense(512, 512, bn=True)
        self.fc2 = dense(512, 512, bn=True)
        self.fc3 = dense(512, 1024*self.out_features, bn=None, activation_fn=None)
        
        self.convTran1 = convTrans2d(512, 512, kernel_size=(2, 2), stride=(1, 1), bn=True)
        self.convTran2 = convTrans2d(512, 256, kernel_size=(3, 3), stride=(1, 1), bn=True)
        self.convTran3 = convTrans2d(256, 256, kernel_size=(4, 4), stride=(2, 2), bn=True)
        self.convTran4 = convTrans2d(256, 128, kernel_size=(5, 5), stride=(3, 3), bn=True)
        self.convTran5 = convTrans2d(128, self.out_features, kernel_size=(1, 1), stride=(1, 1), bn=None, activation_fn=None)


    def forward(self, input):
        fc_output = self.fc(input)
        
        #=============FC DECODER===================
        fc_fc_viewed = fc_output.view(fc_output.shape[0], -1)
        
        fc1_output = self.fc1(fc_fc_viewed)
        fc2_output = self.fc2(fc1_output)
        fc3_output = self.fc3(fc2_output)
        
        fcs_output = fc3_output.view(fc3_output.shape[0], -1, self.out_features)
        
        #=============UPCONV DECODER===================
        fc_conv_viewed = fc_output.view(fc_output.shape[0], -1, 1, 1)

        convT1_output = self.convTran1(fc_conv_viewed)
        # print(convT1_output.shape)
        
        convT2_output = self.convTran2(convT1_output)
        # print(convT2_output.shape)
        
        convT3_output = self.convTran3(convT2_output)
        # print(convT3_output.shape)
        
        convT4_output = self.convTran4(convT3_output)
        # print(convT4_output.shape)
        
        convT5_output = self.convTran5(convT4_output)
        # print(convT5_output.shape)
        
        convT_output = convT5_output.view(convT5_output.shape[0], -1, self.out_features)
        
        # SET Union
        output = torch.concat((fcs_output, convT_output), dim=1)
        
        return output
    

class MBN(nn.Module):
    def __init__(self, num_points, input_dim, feat_dim, nm, loss_type, bn=True):
        super(MBN, self).__init__()
        self.num_points = num_points
        self.input_dim = input_dim
        self.nm = nm
        self.loss_type = loss_type
        
        self.num_rp = 50

        self.conv1 = conv2d(input_dim, 64, (1, 1), bn=bn)
        self.conv2 = conv2d(64, 128, (1, 1), bn=bn)
        self.conv3 = conv2d(128, 128, (1, 1), bn=bn)
        self.conv4 = conv2d(128, feat_dim, (1, 1), bn=bn)
        
        self.maxpool = nn.MaxPool2d((self.num_points, 1), return_indices=True)

        self.rot_layers = nn.Sequential(*[
            dense(feat_dim, 256, bn=bn),
            dense(256, 256, bn=bn),
        ])

        self.q_layer = dense(256, nm * 4, activation_fn=None, bn=None)
        self.l_layer = dense(256, nm * 3, activation_fn=None, bn=None)
        
        if self.loss_type == 'CE':
            self.weights = dense(256, nm, activation_fn=nn.Sigmoid(), bn=None)
        else:
            self.weights = dense(256, nm, activation_fn=None, bn=None)
        
        
        self.z_layer = dense(nm * 4, 256, activation_fn=None, bn=None)
        self.z_layer_gen = dense(self.num_rp * 4, 256, activation_fn=None, bn=None)
            
        self.discrinimator = discriminator(256, 3)
        self.generator = discriminator(256, 3)
        
        
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
    
    def recon(self, pose):
        #pose [batchsize, 4]
        #xyz [batchsize, 3]
        
        nwpqr_rp = pose.unsqueeze(1).expand(-1, self.num_rp, -1).reshape(-1, self.num_rp * 4)
        
        lambdas_rp = torch.ones(nwpqr_rp.shape[0], self.num_rp, 3, device=nwpqr_rp.device) * -500
        lambdas_rp = lambdas_rp.reshape(-1, self.num_rp, 3)
        lambdas_rp[:, :, 1] = lambdas_rp[:, :, 1] - 0.000001
        lambdas_rp[:, :, 2] = lambdas_rp[:, :, 2] - 0.000002
        lambdas_rp = lambdas_rp.reshape(-1, self.num_rp * 3)
        
        nwpqr_rp = nwpqr_rp.reshape(-1, 4)
        lambdas_rp = lambdas_rp.reshape(-1, 3) 
        
        # Reparameterize 
        z = self.reparameterize_rj(nwpqr_rp, lambdas_rp)
        
        z_reshaped = z.reshape(-1, self.num_rp*4) # [84, 50*4]
        z_layer_output = self.z_layer_gen(z_reshaped) # [84, 256]       
 
        
        x_gen = self.generator(z_layer_output) # [84, 2048, 3]
            
        return x_gen

    def forward(self, input): # [84, 2048, 3]
        input_reshaped = input.reshape(-1, 1, self.num_points, self.input_dim) # [84, 1, 2048, 3]
        input_reordered = input_reshaped.permute(0, 3, 2, 1) # [84, 3, 2048, 1]

        conv1_output = self.conv1(input_reordered) # [84, 64, 2048, 1]
        conv2_output = self.conv2(conv1_output) # [84, 128, 2048, 1]
        conv3_output = self.conv3(conv2_output) # [84, 128, 2048, 1]
        conv4_output = self.conv4(conv3_output) # [84, 128, 2048, 1]
        
        max_pool_outptut, indices = self.maxpool(conv4_output) # [84, 128, 1, 1]
        max_pool_reshaped_output = max_pool_outptut.reshape(max_pool_outptut.shape[0], -1) # [84, 128]        
        rot_layers_output = self.rot_layers(max_pool_reshaped_output)
        
        q_layer_output = self.q_layer(rot_layers_output)
        l_layer_output = F.softplus(self.l_layer(rot_layers_output))      

        # convert from original output of network to lambdas
        dZ = l_layer_output.reshape(-1, 3)
        Z0 = dZ[:, 0:1]
        Z1 = Z0 + dZ[:, 1:2]
        Z2 = Z1 + dZ[:, 2:3]
        Zbatch = torch.cat([Z0, Z1, Z2], dim=1)
        Zbatch = -1 * Zbatch.clamp(1e-12, 900)

        # normalize q
        q_layer_output = q_layer_output.reshape(-1, 4)
        norm_q_output = torch.norm(q_layer_output, dim=-1, keepdim=True)
        normalized_q_output = q_layer_output / (norm_q_output + 1e-12)
        normalized_q_output = ((normalized_q_output[:, 0:1] > 0).float() - 0.5) * 2 * normalized_q_output

        weights = self.weights(rot_layers_output)
        
        #=========================Discrinimator Mixture============================
        # Transfer xyz, nwpqr for reparameterize
        z = self.reparameterize_rj(normalized_q_output, Zbatch)
        
        z_reshaped = z.reshape(-1, self.nm*4) # [84, 50*4]
        z_layer_output = self.z_layer(z_reshaped) # [84, 256]
        
        x_dis = self.discrinimator(z_layer_output) # [84, 2048, 3]
 
        #========================Generator single=============================
        # Transfer xyz, nwpqr for SINGLE reparameterize
        
        nwpqr_rp = normalized_q_output
        lambdas_rp = Zbatch
        
        _, max_indices = weights.max(dim=1)
        
        nwpqr_rp = nwpqr_rp.reshape(-1, self.nm, 4)
        nwpqr_rp = nwpqr_rp[torch.arange(nwpqr_rp.shape[0]), max_indices]
        nwpqr_rp = nwpqr_rp.unsqueeze(1).expand(-1, self.num_rp, -1).reshape(-1, self.num_rp * 4)
        
        lambdas_rp = torch.ones(nwpqr_rp.shape[0], self.num_rp, 3, device=nwpqr_rp.device) * -500
        lambdas_rp[:, :, 1] = lambdas_rp[:, :, 1] - 0.000001
        lambdas_rp[:, :, 2] = lambdas_rp[:, :, 2] - 0.000002
        lambdas_rp = lambdas_rp.reshape(-1, self.num_rp * 3)
        
        nwpqr_rp = nwpqr_rp.reshape(-1, 4)
        lambdas_rp = lambdas_rp.reshape(-1, 3)       
        #=====================================================
        
        z = self.reparameterize_rj(nwpqr_rp, lambdas_rp)
        
        z_reshaped = z.reshape(-1, self.num_rp*4) # [84, 50*4]
        z_layer_output = self.z_layer_gen(z_reshaped) # [84, 256]       
 
        
        x_gen = self.generator(z_layer_output) # [84, 2048, 3]        
        
        return normalized_q_output, Zbatch, weights, x_dis, x_gen
