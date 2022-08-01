
# limitations under the License.

from typing import Tuple, Union
import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
# from monai.networks.blocks.unetr_block import UnetstrBasicBlock, UnetrPrUpBlock, UnetResBlock


from scripts.networks.unest_block import UNesTConvBlock, UNestUpBlock, UNesTBlock

from monai.networks.blocks import Convolution

# from scripts.networks.swin_transformer_3d import SwinTransformer3D
from scripts.networks.nest_transformer_3D import NestTransformer3D

import pdb

class UNesT(nn.Module):
    """
    UNesT model implementation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int] = [96, 96, 96],
        feature_size: int = 16,
        patch_size: int = 2,
        depths: Tuple[int, int, int, int] = [2, 2, 2, 2],
        num_heads: Tuple[int, int, int, int] = [3, 6, 12, 24],
        window_size: Tuple[int, int, int] = [7, 7, 7],
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        # featResBlock: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")
        # self.featResBlock = featResBlock
        # if featResBlock:
        #     self.feat_res_block = UnetResBlock(
        #         spatial_dims=3,
        #         in_channels=in_channels,
        #         out_channels=1,
        #         kernel_size=3,
        #         stride=1,
        #         norm_name=norm_name,
        #     )
        # self.swinViT = SwinTransformer3D(
        #     pretrained=None,
        #     pretrained2d=False,
        #     patch_size=(patch_size, patch_size, patch_size),
        #     in_chans=in_channels,
        #     embed_dim=feature_size,
        #     depths = depths,
        #     num_heads=num_heads,
        #     window_size=window_size,
        #     mlp_ratio=4.,
        #     qkv_bias=True,
        #     qk_scale=None,
        #     drop_rate=0.,
        #     attn_drop_rate=0.,
        #     drop_path_rate=0.0,
        #     norm_layer=nn.LayerNorm,
        # )
        self.embed_dim = [128, 256, 512]

        self.nestViT = NestTransformer3D(
            img_size=96, 
            in_chans=1, 
            patch_size=4, 
            num_levels=3, 
            embed_dims=(128, 256, 512),                 
            num_heads=(4, 8, 16), 
            depths=(2, 2, 8), 
            num_classes=1000, 
            mlp_ratio=4., 
            qkv_bias=True,                
            drop_rate=0., 
            attn_drop_rate=0., 
            drop_path_rate=0.5, 
            norm_layer=None, 
            act_layer=None,
            pad_type='', 
            weight_init='', 
            global_pool='avg',
        )

        self.encoder1 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=1,
            out_channels=feature_size * 2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UNestUpBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=False,
            res_block=False,
        )
        # self.encoder2 = UnetrBasicBlock(
        #     spatial_dims=3,
        #     in_channels=self.embed_dim[0],
        #     out_channels=4 * feature_size,
        #     kernel_size=3,
        #     stride=1,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.encoder2 = UnetrBasicBlock(
        #     spatial_dims=3,
        #     in_channels=self.embed_dim[0],
        #     out_channels=8 * feature_size,
        #     kernel_size=3,
        #     stride=1,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.encoder3 = UnetrPrUpBlock(
        #     spatial_dims=3,
        #     in_channels=self.embed_dim[0],
        #     out_channels=feature_size * 4,
        #     num_layer=1,
        #     kernel_size=3,
        #     stride=1,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     conv_block=conv_block,
        #     res_block=res_block,
        # )

        self.encoder3 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder4 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[1],
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder5 = UNesTBlock(
            spatial_dims=3,
            in_channels=2*self.embed_dim[2],
            out_channels=feature_size * 32,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UNesTBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[2],
            out_channels=feature_size * 16,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UNesTBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UNesTBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder1 = UNesTBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        # self.encoder10 = UnetrBasicBlock(
        #     spatial_dims=3,
        #     in_channels=32*feature_size,
        #     out_channels=64*feature_size,
        #     kernel_size=3,
        #     stride=1,
        #     norm_name=norm_name,
        #     res_block=res_block)

        # self.encoder10 = UnetrPrUpBlock(
        #     spatial_dims=3,
        #     in_channels=32*feature_size,
        #     out_channels=64*feature_size,
        #     num_layer=0,
        #     kernel_size=3,
        #     stride=1,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     conv_block=conv_block,
        #     res_block=res_block,
        # )


        self.encoder10 = Convolution(
            dimensions=3,
            in_channels=32*feature_size,
            out_channels=64*feature_size,
            strides=2,
            adn_ordering="ADN",
            dropout=0.0,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights['state_dict']:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(weights['state_dict']['module.transformer.patch_embedding.position_embeddings_3d'])
            self.vit.patch_embedding.cls_token.copy_(weights['state_dict']['module.transformer.patch_embedding.cls_token'])
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(weights['state_dict']['module.transformer.patch_embedding.patch_embeddings_3d.1.weight'])
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(weights['state_dict']['module.transformer.patch_embedding.patch_embeddings_3d.1.bias'])

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights['state_dict']['module.transformer.norm.weight'])
            self.vit.norm.bias.copy_(weights['state_dict']['module.transformer.norm.bias'])
            
    def forward(self, x_in):
        # if self.featResBlock:
        #     x_in = self.feat_res_block(x_in)
        x, hidden_states_out = self.nestViT(x_in)

        enc0 = self.encoder1(x_in) # 2, 32, 96, 96, 96
        # print('enc0 shape: {}'.format(enc0.shape))

        x1 = hidden_states_out[0] # 2, 128, 24, 24, 24
        # print('x1 shape: {}'.format(x1.shape))

        enc1 = self.encoder2(x1) # 2, 64, 48, 48, 48
        # print('enc1 shape: {}'.format(enc1.shape))

        x2 = hidden_states_out[1] # 2, 128, 24, 24, 24
        # print('x2 shape: {}'.format(x2.shape))

        enc2 = self.encoder3(x2) # 2, 128, 24, 24, 24
        # print('enc2 shape: {}'.format(enc2.shape))

        x3 = hidden_states_out[2] # 2, 256, 12, 12, 12
        # print('x3 shape: {}'.format(x3.shape))

        enc3 = self.encoder4(x3) # 2, 256, 12, 12, 12
        # print('enc3 shape: {}'.format(enc3.shape))

        x4 = hidden_states_out[3]

        enc4 = x4 # 2, 512, 6, 6, 6
        # print('enc4 shape: {}'.format(enc4.shape))

        dec4 = x # 2, 512, 6, 6, 6
        # print('dec4 shape: {}'.format(dec4.shape))

        dec4 = self.encoder10(dec4) # 2, 1024, 3, 3, 3
        # print('new dec4 shape: {}'.format(dec4.shape))

        dec3 = self.decoder5(dec4, enc4) # 2, 512, 6, 6, 6
        # print('dec3 shape: {}'.format(dec3.shape))

        dec2 = self.decoder4(dec3, enc3) # 2, 256, 12, 12, 12
        # print('dec2 shape: {}'.format(dec2.shape))

        dec1 = self.decoder3(dec2, enc2) # 2, 128, 24, 24, 24
        # print('dec1 shape: {}'.format(dec1.shape))

        dec0 = self.decoder2(dec1, enc1) # 2, 64, 48, 48, 48
        # print('dec0 shape: {}'.format(dec0.shape))

        out = self.decoder1(dec0, enc0) # 2, 32, 96, 96, 96
        # print('out shape: {}'.format(out.shape))

        logits = self.out(out)
        return logits
