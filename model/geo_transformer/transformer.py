import copy
import torch
import torch.nn as nn

from utils.common_utils import sample_descriptors
from model.geo_transformer.geo_attention import LinearAttention, FullAttention


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead, linear=False):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        if linear:
            self.attention = LinearAttention()
        else:
            self.attention = FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.Tanh(),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """


        bs = x.size(0)

        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, x_mask, source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)


        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message

class GeoTransformer(nn.Module):

    def __init__(self, config, layer_names, d_model, linear=True):
        super(GeoTransformer, self).__init__()

        self.config = config
        self.d_model = d_model
        self.layer_names = layer_names
        self.nhead = config['nhead']
        encoder_layer = LoFTREncoderLayer(self.d_model, self.nhead, linear)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()
        self.norm = nn.LayerNorm(self.d_model)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, kp0_cross, kp1_cross, h0, w0, h1, w1, scale,
                mask_self0=None, mask_self1=None, mask_cross0=None, mask_cross1=None, **kwargs):
        """
        inplace operation for feat0 and feat1
        :param feat0:
        :param feat1:
        :param kp0_cross: the corresponding areas in feat0 of each keypoint in feat1
        :param kp1_cross: the corresponding areas in feat1 of each keypoint in feat0
        :param h0: size of feat0 (2D)
        :param w0:
        :param h1: size of feat1 (2D)
        :param w1:
        :param scale: feat(2D) to raw size
        :param mask_self0: used for self-attention on feat0
        :param mask_self1:
        :param mask_cross0: illegal area of kp0_cross
        :param mask_cross1:
        :return:
        """
        assert self.d_model == feat0.size(2) , "the feature number of src and transformer must be equal"

        # 初始化用于存储更新后的特征的张量
        updated_feat0 = torch.zeros_like(feat0)
        updated_feat1 = torch.zeros_like(feat1)

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                for step in range(len(feat0)):
                    feat0_at = feat0[step]
                    feat1_at = feat1[step]
                    mask0_at = mask_self0[step]
                    mask1_at = mask_self1[step]
                    if mask0_at.sum() > 0:
                        feat0_at = layer(feat0_at.unsqueeze(0), feat0_at[mask0_at].unsqueeze(0))[0]

                    if mask1_at.sum() > 0:
                        feat1_at = layer(feat1_at.unsqueeze(0), feat1_at[mask1_at].unsqueeze(0))[0]

                    updated_feat0[step] = feat0_at
                    updated_feat1[step] = feat1_at
            elif name == 'cross':
                feat0_map = feat0.view(len(feat0), h0, w0, feat0.shape[-1]).permute(0, 3, 1, 2)
                feat1_map = feat1.view(len(feat1), h1, w1, feat1.shape[-1]).permute(0, 3, 1, 2)
                feat0_cross = sample_descriptors(kp0_cross, feat0_map, scale)
                feat1_cross = sample_descriptors(kp1_cross, feat1_map, scale)
                for step in range(len(feat0_cross)):
                    feat0_at = feat0[step].unsqueeze(1)
                    feat1_at = feat1[step].unsqueeze(1)
                    if feat1_cross[step] is not None:
                        feat0_at_cross = feat0_cross[step]
                        feat1_at_cross = feat1_cross[step]
                        feat0_at = layer(feat0_at, feat1_at_cross, None, mask_cross1[step])
                        feat1_at = layer(feat1_at, feat0_at_cross, None, mask_cross0[step])
                    updated_feat0[step] = feat0_at.squeeze(1)
                    updated_feat1[step] = feat1_at.squeeze(1)

            else:
                raise KeyError
        return updated_feat0, updated_feat1


class GeoTransformer_Semantic(GeoTransformer):
    def __init__(self, config, layer_names, d_model, linear):
        print('old layer_names:', layer_names)
        layer_names = ['self', 'cross'] * 2 + ['self_semantic', 'cross_semantic'] * 1
        # layer_names = ['self', 'cross'] * 1 + ['self', 'self_semantic', 'cross', 'cross_semantic'] * 1
        layer_names_old = [x for x in layer_names if 'semantic' not in x]
        print('new layer_names:', layer_names)
        super(GeoTransformer_Semantic, self).__init__(config, layer_names_old, d_model, linear)

        block_dims = [self.d_model] * 2
        # add semantic
        up_factors = [1, 1]
        img_dims = block_dims
        semantic_outdims = [x // 2 for x in block_dims]
        from model.sed_module_attention import AddSemantic

        addsemantic = nn.ModuleList(
            AddSemantic(semantic_indim=384, img_dim=dim, semantic_outdim=sdim, up_factor=factor, bypass=False)
            for i, (dim, sdim, factor) in enumerate(zip(img_dims, semantic_outdims, up_factors))
        )

        old_layers = self.layers
        layers = nn.ModuleList()
        for name in layer_names:
            if 'semantic' in name:
                layers.append(addsemantic.pop(0))
            else:
                layers.append(old_layers.pop(0))
        self.layers = layers
        self.layer_names = layer_names

    def forward(self, feat0, feat1, kp0_cross, kp1_cross, h0, w0, h1, w1, scale,
                mask_self0=None, mask_self1=None, mask_cross0=None, mask_cross1=None, data=None):
        """
        inplace operation for feat0 and feat1
        :param feat0:
        :param feat1:
        :param kp0_cross: the corresponding areas in feat0 of each keypoint in feat1
        :param kp1_cross: the corresponding areas in feat1 of each keypoint in feat0
        :param h0: size of feat0 (2D)
        :param w0:
        :param h1: size of feat1 (2D)
        :param w1:
        :param scale: feat(2D) to raw size
        :param mask_self0: used for self-attention on feat0
        :param mask_self1:
        :param mask_cross0: illegal area of kp0_cross
        :param mask_cross1:
        :return:
        """
        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"
        assert feat0.size(0) == 1, "the batchsize must equal to 1"

        ## add semantic
        semantic0, semantic1 = data.get('dino_fea0ori', None), data.get('dino_fea1ori', None)
        b, c, h0, w0, h1, w1 = feat0.size(0), feat0.size(-1), *data['hw0_c'], *data['hw1_c'],

        def to_blc(x, h, w):
            return x.reshape(b, c, h * w).permute(0, 2, 1)

        def to_bchw(x, h, w):
            return x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        FUNCS = {'self': to_blc, 'cross': to_blc, 'self_semantic': to_bchw, 'cross_semantic': to_bchw}
        last_reshape_func = to_blc
        for name, layer in zip(self.layer_names, self.layers):
            reshape_func = FUNCS.get(name, None)
            if reshape_func != last_reshape_func:
                feat0, feat1 = reshape_func(feat0, h0, w0), reshape_func(feat1, h1, w1)
            last_reshape_func = reshape_func

            if name == 'self':
                for step in range(len(feat0)):
                    feat0_at = feat0[step]
                    feat1_at = feat1[step]
                    mask0_at = mask_self0[step]
                    mask1_at = mask_self1[step]
                    if mask0_at.sum() > 0:
                        feat0_at = layer(feat0_at.unsqueeze(0), feat0_at[mask0_at].unsqueeze(0))[0]

                    if mask1_at.sum() > 0:
                        feat1_at = layer(feat1_at.unsqueeze(0), feat1_at[mask1_at].unsqueeze(0))[0]

                    feat0 = feat0_at.unsqueeze(0)
                    feat1 = feat1_at.unsqueeze(0)
            elif name == 'cross':
                feat0_map = feat0.view(len(feat0), h0, w0, feat0.shape[-1]).permute(0, 3, 1, 2)
                feat1_map = feat1.view(len(feat1), h1, w1, feat1.shape[-1]).permute(0, 3, 1, 2)
                feat0_cross = sample_descriptors(kp0_cross, feat0_map, scale)
                feat1_cross = sample_descriptors(kp1_cross, feat1_map, scale)
                for step in range(len(feat0_cross)):
                    feat0_at = feat0[step].unsqueeze(1)
                    feat1_at = feat1[step].unsqueeze(1)
                    if feat1_cross[step] is not None:
                        feat0_at_cross = feat0_cross[step]
                        feat1_at_cross = feat1_cross[step]
                        feat0_at = layer(feat0_at, feat1_at_cross, None, mask_cross1[step])
                        feat1_at = layer(feat1_at, feat0_at_cross, None, mask_cross0[step])
                    feat0 = feat0_at.squeeze(1).unsqueeze(0)
                    feat1 = feat1_at.squeeze(1).unsqueeze(0)
            elif name == 'self_semantic':
                feat0 = layer(semantic0, feat0)
                feat1 = layer(semantic1, feat1)
            elif name == 'cross_semantic':
                feat0 = layer(semantic0, feat1, concat_source=feat0)
                feat1 = layer(semantic1, feat0, concat_source=feat1)
            else:
                raise KeyError('name: {} is not valid.'.format(name))
        else:
            if last_reshape_func != to_blc:
                feat0, feat1 = to_blc(feat0, h0, w0), to_blc(feat1, h1, w1)

        return feat0, feat1

