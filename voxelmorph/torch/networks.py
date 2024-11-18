import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .. import default_unet_features
from . import layers
from .modelio import LoadableModel, store_config_args

from . import networks_cyclegan
from . import networks_route


netG_A2B_image = networks_cyclegan.Generator_mini(1, 1)
netG_A2B_image.cuda()

netG_A2B_level0 = networks_cyclegan.Generator_mini(16, 16)
netG_A2B_level0.cuda()

netG_A2B_level1 = networks_cyclegan.Generator_mini(32, 32)
netG_A2B_level1.cuda()


netG_A2B_level2 = networks_cyclegan.Generator_mini(32, 32)
netG_A2B_level2.cuda()

netG_A2B_level3 = networks_cyclegan.Generator_mini(32, 32)
netG_A2B_level3.cuda()

netG_A2B = [netG_A2B_level0, netG_A2B_level1, netG_A2B_level2, netG_A2B_level3]


class Unet_fea_trans(nn.Module):
  

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):


        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        
        # cache some parameters
        self.half_res = half_res
        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
        # print('type(nb_features)',type(nb_features))
        # build feature list automatically
        # print('nb_levels',nb_levels) default:none
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        

        # extract any surplus (full resolution) decoder convolutions
        # [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]
        # print('nb_features',nb_features)
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        # nb_dec_convs == 4
        final_convs = dec_nf[nb_dec_convs:]
        # final_convs == [32,16,6]
        dec_nf = dec_nf[:nb_dec_convs]
        # dec_nf == [32,32,32,32]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1
        # print('self.nb_levels',self.nb_levels)
        # self.nb_levels = 5

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels
        # print('max_pool',max_pool)
        # max_pool:[2,2,2,2,2]

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        # print('MaxPooling',MaxPooling)
        # MaxPooling(s) == MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]
        # Upsample(scale_factor=2.0, mode=nearest)
        # print('self.upsampling',self.upsampling)

        # configure encoder (down-sampling path)
        # infeats:2
        # prev_nf = infeats
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        self.concating = nn.ModuleList()
        
        # self.nb_levels = 5
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            # convs_concating = nn.ModuleList()
            # nb_conv_per_level == 1
            for conv in range(nb_conv_per_level):
                # level * nb_conv_per_level + conv: 0~3 
                # enc_nf:[16, 32, 32, 32]
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                convs_concating = ConvBlock(ndims, nf*2, nf) 
                prev_nf = nf
            self.encoder.append(convs)
            self.concating.append(convs_concating)
            
            encoder_nfs.append(prev_nf)
        # encoder_nfs [2, 16, 32, 32, 32]
        
        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        # encoder_nfs [32, 32, 32, 16, 2]
        self.decoder = nn.ModuleList()
        # self.nb_levels = 5
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                # level * nb_conv_per_level + conv: 0~3 
                # dec_nf:[32, 32, 32, 32]
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]
                # print('prev_nf',prev_nf)


        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        # print('final_convs',final_convs)
        # final_convs:[32,16,16]
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf
        # print('self.remaining',self.remaining)

        # cache final number of features
        self.final_nf = prev_nf
        
        # self.SE_module_0 = SE(32,4)
        # self.SE_module_123 = SE(64,8)
        
        # self.conv_level3 = ConvBlock(ndims, 64, 32)
        
          

    def forward(self, source, target):
      
        # encoder forward pass
        # print('x.shape',x.shape)
        x = torch.cat([source, target], dim=1)
        x_history = [x]
        
        # print('self.encoder',self.encoder)
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                # print('conv',conv)
                # print('source.shape',source.shape)
                source = conv(source)   
                target = conv(target)
                x = torch.cat([source, target], dim=1)
                # print('x.shape',x.shape)
                # print('self.concating[level]',self.concating[level])
                x = self.concating[level](x)
            if level == 2:
                visual_source = source
                visual_target = target
  
            x_history.append(x)
            # x = self.pooling[level](x)
            source = self.pooling[level](source)
            target = self.pooling[level](target)
            
        # x = torch.cat([source, target], dim=1)
        
        # x = self.conv_level3(x)
            
            
        x = self.pooling[-1](x)
            # print('level',level)
            # print('x.shape',x.shape)
        # for item in x_history:
        #     print('item.shape',item.shape)
        # print('x.shape',x.shape)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            

    
            for conv in convs:
                x = conv(x)
            # print('level',level)
            # print('x.shape',x.shape)
            if not self.half_res or level < (self.nb_levels - 2):
                
                x = self.upsampling[level](x)
                # print('x.shape',x.shape)
                # mycode
                x_temp = x_history.pop()
                # if x.shape[4] == x_temp.shape[4] - 1:
                #     x = F.pad(x, pad=(1,0,0,0,0,0), mode="constant",value=0) 
                
                x = torch.cat([x, x_temp], dim=1)
                # print('x.shape',x.shape)
            

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)
            
        # print('last_x.shape',x.shape)

        return x, visual_source, visual_target
    
    

class Unet_joint_level0(nn.Module):
  

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):


        super().__init__()
        
        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        
        # cache some parameters
        self.half_res = half_res
        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
        # print('type(nb_features)',type(nb_features))
        # build feature list automatically
        # print('nb_levels',nb_levels) default:none
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        

        # extract any surplus (full resolution) decoder convolutions
        # [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]
        # print('nb_features',nb_features)
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        # nb_dec_convs == 4
        final_convs = dec_nf[nb_dec_convs:]
        # final_convs == [32,16,6]
        dec_nf = dec_nf[:nb_dec_convs]
        # dec_nf == [32,32,32,32]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1
        # print('self.nb_levels',self.nb_levels)
        # self.nb_levels = 5

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels
        # print('max_pool',max_pool)
        # max_pool:[2,2,2,2,2]

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        # print('MaxPooling',MaxPooling)
        # MaxPooling(s) == MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]
        # Upsample(scale_factor=2.0, mode=nearest)
        # print('self.upsampling',self.upsampling)

        # configure encoder (down-sampling path)
        # infeats:2
        # prev_nf = infeats
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        self.concating = nn.ModuleList()
        
        # self.nb_levels = 5
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            # convs_concating = nn.ModuleList()
            # nb_conv_per_level == 1
            for conv in range(nb_conv_per_level):
                # level * nb_conv_per_level + conv: 0~3 
                # enc_nf:[16, 32, 32, 32]
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                convs_concating = ConvBlock(ndims, nf*2, nf) 
                prev_nf = nf
            self.encoder.append(convs)
            self.concating.append(convs_concating)
            
            encoder_nfs.append(prev_nf)
        # encoder_nfs [2, 16, 32, 32, 32]
        
        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        # encoder_nfs [32, 32, 32, 16, 2]
        self.decoder = nn.ModuleList()
        # self.nb_levels = 5
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                # level * nb_conv_per_level + conv: 0~3 
                # dec_nf:[32, 32, 32, 32]
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]
                # print('prev_nf',prev_nf)
        # print('self.encoder',self.encoder)
        # print('self.concating',self.concating)
        # print('self.decoder',self.decoder)

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        # print('final_convs',final_convs)
        # final_convs:[32,16,16]
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf
        # print('self.remaining',self.remaining)

        # cache final number of features
        self.final_nf = prev_nf
        

        
        # self.conv_level3 = ConvBlock(ndims, 64, 32)
        
     
        
        
          

    def forward(self, source, target):
      
        # encoder forward pass
        # print('x.shape',x.shape)
        x = torch.cat([source, target], dim=1)
        x_history = [x]

        
        # print('self.encoder',self.encoder)
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                # print('conv',conv)
                # print('source.shape',source.shape)
                source = conv(source)   
                target = conv(target)
                
                if level == 0:
                    source_for_cyclegan = source
                    target_for_cyclegan = target
            
                    netG_A2B_level0.load_state_dict(torch.load('/scripts/torch/model_joint_R_G_level0/best_netG_A2B.pth'))
                    source = netG_A2B_level0(source)
                
                x = torch.cat([source, target], dim=1)
                # print('x.shape',x.shape)
                # print('self.concating[level]',self.concating[level])
                x = self.concating[level](x)
                
            
                
                      
            x_history.append(x)
            
            # x = self.pooling[level](x)
            source = self.pooling[level](source)
            target = self.pooling[level](target)
            
        # x = torch.cat([source, target], dim=1)
        
        # x = self.conv_level3(x)
            
        x = self.pooling[-1](x)
            # print('level',level)
            # print('x.shape',x.shape)
        # for item in x_history:
        #     print('item.shape',item.shape)
        # print('x.shape',x.shape)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            # print('level',level)
            # print('x.shape',x.shape)
            if not self.half_res or level < (self.nb_levels - 2):
                
                x = self.upsampling[level](x)
                # print('x.shape',x.shape)
                # mycode
                x_temp = x_history.pop()
                # if x.shape[4] == x_temp.shape[4] - 1:
                #     x = F.pad(x, pad=(1,0,0,0,0,0), mode="constant",value=0) 
                
                x = torch.cat([x, x_temp], dim=1)
                # print('x.shape',x.shape)
            

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)
            
        # print('last_x.shape',x.shape)

        return x, source_for_cyclegan,target_for_cyclegan  

    
class Unet_joint_level1(nn.Module):
  

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):


        super().__init__()
        
        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        
        # cache some parameters
        self.half_res = half_res
        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
        # print('type(nb_features)',type(nb_features))
        # build feature list automatically
        # print('nb_levels',nb_levels) default:none
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        

        # extract any surplus (full resolution) decoder convolutions
        # [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]
        # print('nb_features',nb_features)
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        # nb_dec_convs == 4
        final_convs = dec_nf[nb_dec_convs:]
        # final_convs == [32,16,6]
        dec_nf = dec_nf[:nb_dec_convs]
        # dec_nf == [32,32,32,32]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1
        # print('self.nb_levels',self.nb_levels)
        # self.nb_levels = 5

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels
        # print('max_pool',max_pool)
        # max_pool:[2,2,2,2,2]

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        # print('MaxPooling',MaxPooling)
        # MaxPooling(s) == MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]
        # Upsample(scale_factor=2.0, mode=nearest)
        # print('self.upsampling',self.upsampling)

        # configure encoder (down-sampling path)
        # infeats:2
        # prev_nf = infeats
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        self.concating = nn.ModuleList()
        
        # self.nb_levels = 5
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            # convs_concating = nn.ModuleList()
            # nb_conv_per_level == 1
            for conv in range(nb_conv_per_level):
                # level * nb_conv_per_level + conv: 0~3 
                # enc_nf:[16, 32, 32, 32]
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                convs_concating = ConvBlock(ndims, nf*2, nf) 
                prev_nf = nf
            self.encoder.append(convs)
            self.concating.append(convs_concating)
            
            encoder_nfs.append(prev_nf)
        # encoder_nfs [2, 16, 32, 32, 32]
        
        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        # encoder_nfs [32, 32, 32, 16, 2]
        self.decoder = nn.ModuleList()
        # self.nb_levels = 5
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                # level * nb_conv_per_level + conv: 0~3 
                # dec_nf:[32, 32, 32, 32]
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]
                # print('prev_nf',prev_nf)
        # print('self.encoder',self.encoder)
        # print('self.concating',self.concating)
        # print('self.decoder',self.decoder)

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        # print('final_convs',final_convs)
        # final_convs:[32,16,16]
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf
        # print('self.remaining',self.remaining)

        # cache final number of features
        self.final_nf = prev_nf
        

        
        # self.conv_level3 = ConvBlock(ndims, 64, 32)
        
     
        
        
          

    def forward(self, source, target):
      
        # encoder forward pass
        # print('x.shape',x.shape)
        x = torch.cat([source, target], dim=1)
        x_history = [x]

        
        # print('self.encoder',self.encoder)
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                # print('conv',conv)
                # print('source.shape',source.shape)
                source = conv(source)   
                target = conv(target)
                
                if level == 1:
                    source_for_cyclegan = source
                    target_for_cyclegan = target
            
                    netG_A2B_level1.load_state_dict(torch.load('/scripts/torch/model_joint_R_G_level1/best_netG_A2B.pth'))
                    source = netG_A2B_level1(source)
                
                x = torch.cat([source, target], dim=1)
                # print('x.shape',x.shape)
                # print('self.concating[level]',self.concating[level])
                x = self.concating[level](x)
                
            
                
                      
            x_history.append(x)
            
            # x = self.pooling[level](x)
            source = self.pooling[level](source)
            target = self.pooling[level](target)
            
        # x = torch.cat([source, target], dim=1)
        
        # x = self.conv_level3(x)
            
        x = self.pooling[-1](x)
            # print('level',level)
            # print('x.shape',x.shape)
        # for item in x_history:
        #     print('item.shape',item.shape)
        # print('x.shape',x.shape)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            # print('level',level)
            # print('x.shape',x.shape)
            if not self.half_res or level < (self.nb_levels - 2):
                
                x = self.upsampling[level](x)
                # print('x.shape',x.shape)
                # mycode
                x_temp = x_history.pop()
                # if x.shape[4] == x_temp.shape[4] - 1:
                #     x = F.pad(x, pad=(1,0,0,0,0,0), mode="constant",value=0) 
                
                x = torch.cat([x, x_temp], dim=1)
                # print('x.shape',x.shape)
            

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)
            
        # print('last_x.shape',x.shape)

        return x, source_for_cyclegan,target_for_cyclegan  
    
class Unet_joint_level2(nn.Module):
  

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):


        super().__init__()
        
        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        
        # cache some parameters
        self.half_res = half_res
        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
        # print('type(nb_features)',type(nb_features))
        # build feature list automatically
        # print('nb_levels',nb_levels) default:none
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        

        # extract any surplus (full resolution) decoder convolutions
        # [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]
        # print('nb_features',nb_features)
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        # nb_dec_convs == 4
        final_convs = dec_nf[nb_dec_convs:]
        # final_convs == [32,16,6]
        dec_nf = dec_nf[:nb_dec_convs]
        # dec_nf == [32,32,32,32]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1
        # print('self.nb_levels',self.nb_levels)
        # self.nb_levels = 5

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels
        # print('max_pool',max_pool)
        # max_pool:[2,2,2,2,2]

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        # print('MaxPooling',MaxPooling)
        # MaxPooling(s) == MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]
        # Upsample(scale_factor=2.0, mode=nearest)
        # print('self.upsampling',self.upsampling)

        # configure encoder (down-sampling path)
        # infeats:2
        # prev_nf = infeats
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        self.concating = nn.ModuleList()
        
        # self.nb_levels = 5
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            # convs_concating = nn.ModuleList()
            # nb_conv_per_level == 1
            for conv in range(nb_conv_per_level):
                # level * nb_conv_per_level + conv: 0~3 
                # enc_nf:[16, 32, 32, 32]
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                convs_concating = ConvBlock(ndims, nf*2, nf) 
                prev_nf = nf
            self.encoder.append(convs)
            self.concating.append(convs_concating)
            
            encoder_nfs.append(prev_nf)
        # encoder_nfs [2, 16, 32, 32, 32]
        
        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        # encoder_nfs [32, 32, 32, 16, 2]
        self.decoder = nn.ModuleList()
        # self.nb_levels = 5
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                # level * nb_conv_per_level + conv: 0~3 
                # dec_nf:[32, 32, 32, 32]
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]
                # print('prev_nf',prev_nf)
        # print('self.encoder',self.encoder)
        # print('self.concating',self.concating)
        # print('self.decoder',self.decoder)

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        # print('final_convs',final_convs)
        # final_convs:[32,16,16]
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf
        # print('self.remaining',self.remaining)

        # cache final number of features
        self.final_nf = prev_nf
        

        
        # self.conv_level3 = ConvBlock(ndims, 64, 32)
        
     
        
        
          

    def forward(self, source, target):
      
        # encoder forward pass
        # print('x.shape',x.shape)
        x = torch.cat([source, target], dim=1)
        x_history = [x]

        
        # print('self.encoder',self.encoder)
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                # print('conv',conv)
                # print('source.shape',source.shape)
                source = conv(source)   
                target = conv(target)
                
                if level == 2:
                    source_for_cyclegan = source
                    target_for_cyclegan = target
                    # print('aaaaaaaa')
            
                    netG_A2B_level2.load_state_dict(torch.load('/scripts/torch/model_joint_R_G_level2/best_netG_A2B.pth'))
                    # start = time.time()
                    source = netG_A2B_level2(source)
                    visual_source = source
                    visual_target = target
                    # trans_time = time.time() - start
                    # print('trans_time',trans_time)
                
                x = torch.cat([source, target], dim=1)
                # print('x.shape',x.shape)
                # print('self.concating[level]',self.concating[level])
                x = self.concating[level](x)
                
            
                
                      
            x_history.append(x)
            
            # x = self.pooling[level](x)
            source = self.pooling[level](source)
            target = self.pooling[level](target)
            
        # x = torch.cat([source, target], dim=1)
        
        # x = self.conv_level3(x)
            
        x = self.pooling[-1](x)
            # print('level',level)
            # print('x.shape',x.shape)
        # for item in x_history:
        #     print('item.shape',item.shape)
        # print('x.shape',x.shape)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            # print('level',level)
            # print('x.shape',x.shape)
            if not self.half_res or level < (self.nb_levels - 2):
                
                x = self.upsampling[level](x)
                # print('x.shape',x.shape)
                # mycode
                x_temp = x_history.pop()
                # if x.shape[4] == x_temp.shape[4] - 1:
                #     x = F.pad(x, pad=(1,0,0,0,0,0), mode="constant",value=0) 
                
                x = torch.cat([x, x_temp], dim=1)
                # print('x.shape',x.shape)
            

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)
            
        # print('last_x.shape',x.shape)

        return x, source_for_cyclegan,target_for_cyclegan,visual_source,visual_target
    
class Unet_joint_level3(nn.Module):
  

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):


        super().__init__()
        
        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        
        # cache some parameters
        self.half_res = half_res
        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
        # print('type(nb_features)',type(nb_features))
        # build feature list automatically
        # print('nb_levels',nb_levels) default:none
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        

        # extract any surplus (full resolution) decoder convolutions
        # [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]
        # print('nb_features',nb_features)
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        # nb_dec_convs == 4
        final_convs = dec_nf[nb_dec_convs:]
        # final_convs == [32,16,6]
        dec_nf = dec_nf[:nb_dec_convs]
        # dec_nf == [32,32,32,32]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1
        # print('self.nb_levels',self.nb_levels)
        # self.nb_levels = 5

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels
        # print('max_pool',max_pool)
        # max_pool:[2,2,2,2,2]

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        # print('MaxPooling',MaxPooling)
        # MaxPooling(s) == MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]
        # Upsample(scale_factor=2.0, mode=nearest)
        # print('self.upsampling',self.upsampling)

        # configure encoder (down-sampling path)
        # infeats:2
        # prev_nf = infeats
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        self.concating = nn.ModuleList()
        
        # self.nb_levels = 5
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            # convs_concating = nn.ModuleList()
            # nb_conv_per_level == 1
            for conv in range(nb_conv_per_level):
                # level * nb_conv_per_level + conv: 0~3 
                # enc_nf:[16, 32, 32, 32]
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                convs_concating = ConvBlock(ndims, nf*2, nf) 
                prev_nf = nf
            self.encoder.append(convs)
            self.concating.append(convs_concating)
            
            encoder_nfs.append(prev_nf)
        # encoder_nfs [2, 16, 32, 32, 32]
        
        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        # encoder_nfs [32, 32, 32, 16, 2]
        self.decoder = nn.ModuleList()
        # self.nb_levels = 5
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                # level * nb_conv_per_level + conv: 0~3 
                # dec_nf:[32, 32, 32, 32]
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]
                # print('prev_nf',prev_nf)
        # print('self.encoder',self.encoder)
        # print('self.concating',self.concating)
        # print('self.decoder',self.decoder)

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        # print('final_convs',final_convs)
        # final_convs:[32,16,16]
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf
        # print('self.remaining',self.remaining)

        # cache final number of features
        self.final_nf = prev_nf
        

        
        # self.conv_level3 = ConvBlock(ndims, 64, 32)
        
     
        
        
          

    def forward(self, source, target):
      
        # encoder forward pass
        # print('x.shape',x.shape)
        x = torch.cat([source, target], dim=1)
        x_history = [x]

        
        # print('self.encoder',self.encoder)
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                # print('conv',conv)
                # print('source.shape',source.shape)
                source = conv(source)   
                target = conv(target)
                
                if level == 3:
                    source_for_cyclegan = source
                    target_for_cyclegan = target
            
                    netG_A2B_level3.load_state_dict(torch.load('/scripts/torch/model_joint_R_G_level3/best_netG_A2B.pth'))
                    source = netG_A2B_level3(source)
                
                x = torch.cat([source, target], dim=1)
                # print('x.shape',x.shape)
                # print('self.concating[level]',self.concating[level])
                x = self.concating[level](x)
                      
            x_history.append(x)
            
            # x = self.pooling[level](x)
            source = self.pooling[level](source)
            target = self.pooling[level](target)
            
        # x = torch.cat([source, target], dim=1)
        
        # x = self.conv_level3(x)
            
        x = self.pooling[-1](x)
            # print('level',level)
            # print('x.shape',x.shape)
        # for item in x_history:
        #     print('item.shape',item.shape)
        # print('x.shape',x.shape)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            # print('level',level)
            # print('x.shape',x.shape)
            if not self.half_res or level < (self.nb_levels - 2):
                
                x = self.upsampling[level](x)
                # print('x.shape',x.shape)
                # mycode
                x_temp = x_history.pop()
                # if x.shape[4] == x_temp.shape[4] - 1:
                #     x = F.pad(x, pad=(1,0,0,0,0,0), mode="constant",value=0) 
                
                x = torch.cat([x, x_temp], dim=1)
                # print('x.shape',x.shape)
            

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)
            
        # print('last_x.shape',x.shape)

        return x, source_for_cyclegan,target_for_cyclegan    
    
    
class Unet_route(nn.Module):
  

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):


        super().__init__()
        
        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        
        # cache some parameters
        self.half_res = half_res
        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
        # print('type(nb_features)',type(nb_features))
        # build feature list automatically
        # print('nb_levels',nb_levels) default:none
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        

        # extract any surplus (full resolution) decoder convolutions
        # [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]
        # print('nb_features',nb_features)
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        # nb_dec_convs == 4
        final_convs = dec_nf[nb_dec_convs:]
        # final_convs == [32,16,6]
        dec_nf = dec_nf[:nb_dec_convs]
        # dec_nf == [32,32,32,32]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1
        # print('self.nb_levels',self.nb_levels)
        # self.nb_levels = 5

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels
        # print('max_pool',max_pool)
        # max_pool:[2,2,2,2,2]

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        # print('MaxPooling',MaxPooling)
        # MaxPooling(s) == MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]
        # Upsample(scale_factor=2.0, mode=nearest)
        # print('self.upsampling',self.upsampling)

        # configure encoder (down-sampling path)
        # infeats:2
        # prev_nf = infeats
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        self.concating = nn.ModuleList()
        
        # self.nb_levels = 5
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            # convs_concating = nn.ModuleList()
            # nb_conv_per_level == 1
            for conv in range(nb_conv_per_level):
                # level * nb_conv_per_level + conv: 0~3 
                # enc_nf:[16, 32, 32, 32]
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                convs_concating = ConvBlock(ndims, nf*2, nf) 
                prev_nf = nf
            self.encoder.append(convs)
            self.concating.append(convs_concating)
            
            encoder_nfs.append(prev_nf)
        # encoder_nfs [2, 16, 32, 32, 32]
        
        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        # encoder_nfs [32, 32, 32, 16, 2]
        self.decoder = nn.ModuleList()
        # self.nb_levels = 5
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                # level * nb_conv_per_level + conv: 0~3 
                # dec_nf:[32, 32, 32, 32]
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]
                # print('prev_nf',prev_nf)
        # print('self.encoder',self.encoder)
        # print('self.concating',self.concating)
        # print('self.decoder',self.decoder)

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        # print('final_convs',final_convs)
        # final_convs:[32,16,16]
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf
        # print('self.remaining',self.remaining)

        # cache final number of features
        self.final_nf = prev_nf
        
        # self.conv_level3 = ConvBlock(ndims, 64, 32)
        
     
        
        
          

    def forward(self, source, target, i):
      
        # encoder forward pass
        # print('x.shape',x.shape)
        x = torch.cat([source, target], dim=1)
        x_history = [x]

        
        # print('self.encoder',self.encoder)
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                # print('conv',conv)
                # print('source.shape',source.shape)
                source = conv(source)   
                target = conv(target)
                
                if level == i:
                    source_for_cyclegan = source
                    target_for_cyclegan = target
                    path = '/scripts/torch/model_joint_R_G_level'+str(i)+'/best_netG_A2B.pth'
    
                    netG_A2B[i].load_state_dict(torch.load(path))
                    source = netG_A2B[i](source)
                
                x = torch.cat([source, target], dim=1)
                # print('x.shape',x.shape)
                # print('self.concating[level]',self.concating[level])
                x = self.concating[level](x)
                
            
                
                      
            x_history.append(x)
            
            # x = self.pooling[level](x)
            source = self.pooling[level](source)
            target = self.pooling[level](target)
            
        # x = torch.cat([source, target], dim=1)
        
        # x = self.conv_level3(x)
            
        x = self.pooling[-1](x)
            # print('level',level)
            # print('x.shape',x.shape)
        # for item in x_history:
        #     print('item.shape',item.shape)
        # print('x.shape',x.shape)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            # print('level',level)
            # print('x.shape',x.shape)
            if not self.half_res or level < (self.nb_levels - 2):
                
                x = self.upsampling[level](x)
                # print('x.shape',x.shape)
                # mycode
                x_temp = x_history.pop()
                # if x.shape[4] == x_temp.shape[4] - 1:
                #     x = F.pad(x, pad=(1,0,0,0,0,0), mode="constant",value=0) 
                
                x = torch.cat([x, x_temp], dim=1)
                # print('x.shape',x.shape)
            

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)
            
        # print('last_x.shape',x.shape)

        return x, source_for_cyclegan,target_for_cyclegan
    
 
class VxmDense_fea_trans(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet_fea_trans(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.transformer_for_label = layers.SpatialTransformer(inshape, mode='nearest')
        
    def get_visual(self, source, target):
        all_put = self.unet_model(source,target)
        # visualization
        visual_source = all_put[1]
        visual_target = all_put[2]
        return visual_source, visual_target
        
       
    def get_pos_flow(self, source, target, registration=False):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        # x = torch.cat([source, target], dim=1)
        x = self.unet_model(source,target)[0]
        # print('x.shape',x.shape)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
        return pos_flow
    
    def predict_label(self, source, pos_flow):
         # warp image with flow field
        y_source = self.transformer_for_label(source, pos_flow)

        return y_source
      
    
    # def forward(self, source, target, label_source,registration=False):
    def forward(self, source, target,registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''
     
        # concatenate inputs and propagate unet
        # x = torch.cat([source, target], dim=1)
        x = self.unet_model(source, target)[0]
        # print('x.shape',x.shape)
       
        # transform into flow field
        flow_field = self.flow(x)
        # print('flow_field.shape',flow_field.shape)
      
        # resize flow for integration
        pos_flow = flow_field
        # print('pos_flow.shape',pos_flow.shape)
        if self.resize:
            pos_flow = self.resize(pos_flow)
            
        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None
        
        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
        
        # print('source.shape',source.shape)
        # print('pos_flow.shape',pos_flow.shape)
     
        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None
        
        # label_warpped = self.transformer(label_source,pos_flow)
        # print('preint_flow.shape',preint_flow.shape)
        # print('pos_flow.shape',pos_flow.shape)
        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow
        # if not registration:
        #     return (y_source, y_target, preint_flow, label_warpped) if self.bidir else (y_source, preint_flow, label_warpped)
        # else:
        #     return y_source, pos_flow, label_warpped        



class VxmDense_route(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet_route(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.transformer_for_label = layers.SpatialTransformer(inshape, mode='nearest')
        
    def get_pos_flow(self, source, target,i, registration=False):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        # x = torch.cat([source, target], dim=1)
        x = self.unet_model(source,target,i)[0]
        # print('x.shape',x.shape)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
        return pos_flow
    
    def predict_label(self, source, pos_flow):
         # warp image with flow field
        y_source = self.transformer_for_label(source, pos_flow)

        return y_source
    
    def get_fea_for_cyclegan(self, source, target):
        source_for_cyclegan = self.unet_model(source, target)[1]
        target_for_cyclegan = self.unet_model(source, target)[2]

        return source_for_cyclegan, target_for_cyclegan
      
    


    # def forward(self, source, target, label_source,registration=False):
    def forward(self, source, target,i,registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        # x = torch.cat([source, target], dim=1)
        x = self.unet_model(source, target,i)[0]
        # print('x.shape',x.shape)

        # transform into flow field
        flow_field = self.flow(x)
        # print('flow_field.shape',flow_field.shape)

        # resize flow for integration
        pos_flow = flow_field
        # print('pos_flow.shape',pos_flow.shape)
        if self.resize:
            pos_flow = self.resize(pos_flow)
            
        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
                
        # print('source.shape',source.shape)
        # print('pos_flow.shape',pos_flow.shape)

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None
        
        # label_warpped = self.transformer(label_source,pos_flow)

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow
        # if not registration:
        #     return (y_source, y_target, preint_flow, label_warpped) if self.bidir else (y_source, preint_flow, label_warpped)
        # else:
        #     return y_source, pos_flow, label_warpped



class VxmDense_joint_level0(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet_joint_level0(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.transformer_for_label = layers.SpatialTransformer(inshape, mode='nearest')
        
    def get_pos_flow(self, source, target, registration=False):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        # x = torch.cat([source, target], dim=1)
        x = self.unet_model(source,target)[0]
        # print('x.shape',x.shape)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
        return pos_flow
    
    def predict_label(self, source, pos_flow):
         # warp image with flow field
        y_source = self.transformer_for_label(source, pos_flow)

        return y_source
    
    # def get_fea_for_cyclegan(self, source, target):
    #     source_for_cyclegan = self.unet_model(source, target)[1]
    #     target_for_cyclegan = self.unet_model(source, target)[2]

    #     return source_for_cyclegan, target_for_cyclegan
      
    


    # def forward(self, source, target, label_source,registration=False):
    def forward(self, source, target,registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        # x = self.unet_model(source, target)[0]
        unet_output = self.unet_model(source, target)
        x = unet_output[0]
        source_for_cyclegan = unet_output[1]
        target_for_cyclegan = unet_output[2]
        
        # print('x.shape',x.shape)
      

        # transform into flow field
        flow_field = self.flow(x)
        # print('flow_field.shape',flow_field.shape)

        # resize flow for integration
        pos_flow = flow_field
        # print('pos_flow.shape',pos_flow.shape)
        if self.resize:
            pos_flow = self.resize(pos_flow)
            
        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
                
        # print('source.shape',source.shape)
        # print('pos_flow.shape',pos_flow.shape)

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None
        
        # label_warpped = self.transformer(label_source,pos_flow)

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow, source_for_cyclegan, target_for_cyclegan)
        else:
            return y_source, pos_flow, source_for_cyclegan, target_for_cyclegan
        # if not registration:
        #     return (y_source, y_target, preint_flow, label_warpped) if self.bidir else (y_source, preint_flow, label_warpped)
        # else:
        #     return y_source, pos_flow, label_warpped

        
class VxmDense_joint_level1(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet_joint_level1(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.transformer_for_label = layers.SpatialTransformer(inshape, mode='nearest')
        
    def get_pos_flow(self, source, target, registration=False):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        # x = torch.cat([source, target], dim=1)
        x = self.unet_model(source,target)[0]
        # print('x.shape',x.shape)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
        return pos_flow
    
    def predict_label(self, source, pos_flow):
         # warp image with flow field
        y_source = self.transformer_for_label(source, pos_flow)

        return y_source
    
    


    # def forward(self, source, target, label_source,registration=False):
    def forward(self, source, target,registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        # x = self.unet_model(source, target)[0]
        unet_output = self.unet_model(source, target)
        x = unet_output[0]
        source_for_cyclegan = unet_output[1]
        target_for_cyclegan = unet_output[2]
        
        # print('x.shape',x.shape)
      

        # transform into flow field
        flow_field = self.flow(x)
        # print('flow_field.shape',flow_field.shape)

        # resize flow for integration
        pos_flow = flow_field
        # print('pos_flow.shape',pos_flow.shape)
        if self.resize:
            pos_flow = self.resize(pos_flow)
            
        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
                
        # print('source.shape',source.shape)
        # print('pos_flow.shape',pos_flow.shape)

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None
        
        # label_warpped = self.transformer(label_source,pos_flow)

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow, source_for_cyclegan, target_for_cyclegan)
        else:
            return y_source, pos_flow, source_for_cyclegan, target_for_cyclegan
        # if not registration:
        #     return (y_source, y_target, preint_flow, label_warpped) if self.bidir else (y_source, preint_flow, label_warpped)
        # else:
        #     return y_source, pos_flow, label_warpped

        
class VxmDense_joint_level2(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet_joint_level2(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.transformer_for_label = layers.SpatialTransformer(inshape, mode='nearest')
    
    def get_visual(self, source, target):
        all_put = self.unet_model(source,target)
        # visualization
        visual_source = all_put[3]
        visual_target = all_put[4]
        return visual_source, visual_target
    
    def get_pos_flow(self, source, target, registration=False):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        # x = torch.cat([source, target], dim=1)
        x = self.unet_model(source,target)[0]
        # print('x.shape',x.shape)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
        return pos_flow
    
    def predict_label(self, source, pos_flow):
         # warp image with flow field
        y_source = self.transformer_for_label(source, pos_flow)

        return y_source
    
    # def get_fea_for_cyclegan(self, source, target):
    #     source_for_cyclegan = self.unet_model(source, target)[1]
    #     target_for_cyclegan = self.unet_model(source, target)[2]

    #     return source_for_cyclegan, target_for_cyclegan
      
    


    # def forward(self, source, target, label_source,registration=False):
    def forward(self, source, target,registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        # x = self.unet_model(source, target)[0]
        unet_output = self.unet_model(source, target)
        x = unet_output[0]
        source_for_cyclegan = unet_output[1]
        target_for_cyclegan = unet_output[2]
        
        # print('x.shape',x.shape)
      

        # transform into flow field
        flow_field = self.flow(x)
        # print('flow_field.shape',flow_field.shape)

        # resize flow for integration
        pos_flow = flow_field
        # print('pos_flow.shape',pos_flow.shape)
        if self.resize:
            pos_flow = self.resize(pos_flow)
            
        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
                
        # print('source.shape',source.shape)
        # print('pos_flow.shape',pos_flow.shape)

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None
        
        # label_warpped = self.transformer(label_source,pos_flow)

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow, source_for_cyclegan, target_for_cyclegan)
        else:
            return y_source, pos_flow, source_for_cyclegan, target_for_cyclegan
        # if not registration:
        #     return (y_source, y_target, preint_flow, label_warpped) if self.bidir else (y_source, preint_flow, label_warpped)
        # else:
        #     return y_source, pos_flow, label_warpped
        
class VxmDense_joint_level3(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet_joint_level3(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.transformer_for_label = layers.SpatialTransformer(inshape, mode='nearest')
        
    def get_pos_flow(self, source, target, registration=False):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        # x = torch.cat([source, target], dim=1)
        x = self.unet_model(source,target)[0]
        # print('x.shape',x.shape)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
        return pos_flow
    
    def predict_label(self, source, pos_flow):
         # warp image with flow field
        y_source = self.transformer_for_label(source, pos_flow)

        return y_source

    #     return source_for_cyclegan, target_for_cyclegan
      
    


    # def forward(self, source, target, label_source,registration=False):
    def forward(self, source, target,registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        # x = self.unet_model(source, target)[0]
        unet_output = self.unet_model(source, target)
        x = unet_output[0]
        source_for_cyclegan = unet_output[1]
        target_for_cyclegan = unet_output[2]
      

        # transform into flow field
        flow_field = self.flow(x)
        # print('flow_field.shape',flow_field.shape)

        # resize flow for integration
        pos_flow = flow_field
        # print('pos_flow.shape',pos_flow.shape)
        if self.resize:
            pos_flow = self.resize(pos_flow)
            
        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
                
       

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None
        
      
      
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow, source_for_cyclegan, target_for_cyclegan)
        else:
            return y_source, pos_flow 

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

