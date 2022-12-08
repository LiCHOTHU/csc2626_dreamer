import torch
from torch import nn
from torch.nn import functional as F

from nerv.training.model import BaseModel
# from nerv.models import deconv_out_shape, conv_norm_act, deconv_norm_act

from .resnet import resnet18, resnet34
from .utils import assert_shape, SoftPositionEmbed, torch_cat
from .predictor import ResidualMLPPredictor, TransformerPredictor, \
    RNNPredictorWrapper

def deconv_out_shape(
    in_size,
    stride,
    padding,
    kernel_size,
    out_padding,
    dilation=1,
):
    """Calculate the output shape of a ConvTranspose layer."""
    if isinstance(in_size, int):
        return (in_size - 1) * stride - 2 * padding + dilation * (
            kernel_size - 1) + out_padding + 1
    elif isinstance(in_size, (tuple, list)):
        return type(in_size)((deconv_out_shape(s) for s in in_size))
    else:
        raise TypeError(f'Got invalid type {type(in_size)} for `in_size`')

def get_normalizer(norm, channels, groups=16, dim='2d'):
    """Get normalization layer."""
    if norm == '':
        return nn.Identity()
    elif norm == 'bn':
        return eval(f'nn.BatchNorm{dim}')(channels)
    elif norm == 'gn':
        # 16 is taken from Table 3 of the GN paper
        return nn.GroupNorm(groups, channels)
    elif norm == 'in':
        return eval(f'nn.InstanceNorm{dim}')(channels)
    elif norm == 'ln':
        return nn.LayerNorm(channels)
    else:
        raise ValueError(f'Normalizer {norm} not supported!')

def get_act_func(act):
    """Get activation function."""
    if act == '':
        return nn.Identity()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'leakyrelu':
        return nn.LeakyReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'swish':
        return nn.SiLU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'softplus':
        return nn.Softplus()
    elif act == 'mish':
        return nn.Mish()
    elif act == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Activation function {act} not supported!')

def conv_norm_act(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    norm='bn',
    act='relu',
    dim='2d',
):
    """Conv - Norm - Act."""
    conv = get_conv(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=norm not in ['bn', 'in'],
        dim=dim)
    normalizer = get_normalizer(norm, out_channels, dim=dim)
    act_func = get_act_func(act)
    return nn.Sequential(conv, normalizer, act_func)

def deconv_norm_act(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    norm='bn',
    act='relu',
    dim='2d',
):
    """ConvTranspose - Norm - Act."""
    deconv = get_deconv(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=norm not in ['bn', 'in'],
        dim=dim)
    normalizer = get_normalizer(norm, out_channels, dim=dim)
    act_func = get_act_func(act)
    return nn.Sequential(deconv, normalizer, act_func)

class SlotAttention(nn.Module):
    """Slot attention module that iteratively performs cross-attention."""

    def __init__(
        self,
        in_features,
        num_iterations,
        num_slots,
        slot_size,
        mlp_hidden_size,
        eps=1e-6,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.eps = eps
        self.attn_scale = self.slot_size**-0.5

        self.norm_inputs = nn.LayerNorm(self.in_features)

        # Linear maps for the attention module.
        self.project_q = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.slot_size, bias=False),
        )
        self.project_k = nn.Linear(in_features, self.slot_size, bias=False)
        self.project_v = nn.Linear(in_features, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

    def forward(self, inputs, slots):
        """Forward function.

        Args:
            inputs (torch.Tensor): [B, N, C], flattened per-pixel features.
            slots (torch.Tensor): [B, num_slots, C] slot inits.

        Returns:
            updated slots, same shape as `slots`.
        """
        # `inputs` has shape [B, num_inputs, inputs_size].
        # `num_inputs` is actually the spatial dim of feature map (H*W)
        bs, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        # Shape: [B, num_inputs, slot_size].
        k = self.project_k(inputs)
        # Shape: [B, num_inputs, slot_size].
        v = self.project_v(inputs)

        # Initialize the slots. Shape: [B, num_slots, slot_size].
        assert len(slots.shape) == 3

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots

            # Attention. Shape: [B, num_slots, slot_size].
            q = self.project_q(slots)

            attn_logits = self.attn_scale * torch.einsum('bnc,bmc->bnm', k, q)
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [B, num_inputs, num_slots].

            # Normalize along spatial dim and do weighted mean.
            attn = attn + self.eps
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.einsum('bnm,bnc->bmc', attn, v)
            # `updates` has shape: [B, num_slots, slot_size].

            # Slot update.
            # GRU is expecting inputs of size (N, L)
            # so flatten batch and slots dimension
            slots = self.gru(
                updates.view(bs * self.num_slots, self.slot_size),
                slots_prev.view(bs * self.num_slots, self.slot_size),
            )
            slots = slots.view(bs, self.num_slots, self.slot_size)
            slots = slots + self.mlp(slots)

        return slots

    @property
    def dtype(self):
        return self.project_k.weight.dtype

    @property
    def device(self):
        return self.project_k.weight.device


class SAVi(BaseModel):
    """SAVi model that operates on videos."""

    def __init__(
        self,
        resolution,
        clip_len,
        slot_dict=dict(
            num_slots=7,
            slot_size=128,
            slot_mlp_size=256,
            num_iterations=2,
        ),
        enc_dict=dict(
            enc_channels=(3, 64, 64, 64, 64),
            enc_ks=5,
            enc_out_channels=128,
            enc_norm='',
        ),
        dec_dict=dict(
            dec_channels=(128, 64, 64, 64, 64),
            dec_resolution=(8, 8),
            dec_ks=5,
            dec_norm='',
        ),
        pred_dict=dict(
            pred_type='transformer',
            pred_rnn=True,
            pred_norm_first=True,
            pred_num_layers=2,
            pred_num_heads=4,
            pred_ffn_dim=512,
            pred_sg_every=None,
        ),
        eps=1e-6,
    ):
        super().__init__()

        self.resolution = resolution
        self.clip_len = clip_len
        self.eps = eps

        self.slot_dict = slot_dict
        self.enc_dict = enc_dict
        self.dec_dict = dec_dict
        self.pred_dict = pred_dict

        self._build_slot_attention()
        self._build_encoder()
        self._build_decoder()
        self._build_predictor()

        # a hack for only extracting slots
        self.testing = True

    def _build_slot_attention(self):
        # Build SlotAttention module
        # kernels x img_feats --> posterior_slots
        self.enc_out_channels = self.enc_dict['enc_out_channels']
        self.num_slots = self.slot_dict['num_slots']
        self.slot_size = self.slot_dict['slot_size']
        self.slot_mlp_size = self.slot_dict['slot_mlp_size']
        self.num_iterations = self.slot_dict['num_iterations']

        # directly use learnable embeddings for each slot
        self.init_latents = nn.Parameter(
            nn.init.normal_(torch.empty(1, self.num_slots, self.slot_size)))

        self.slot_attention = SlotAttention(
            in_features=self.enc_out_channels,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
            eps=self.eps,
        )

    def _build_encoder(self):
        # Build Encoder
        # ResNet with less downsampling
        if self.enc_dict.get('resnet', False):
            use_layer4 = self.enc_dict['use_layer4']
            self.encoder = eval(self.enc_dict['resnet'])(
                small_inputs=True, use_layer4=use_layer4)
            if use_layer4:  # downsample by 8
                self.visual_resolution = tuple(i // 8 for i in self.resolution)
                self.visual_channels = 512
            else:  # downsample by 4
                self.visual_resolution = tuple(i // 4 for i in self.resolution)
                self.visual_channels = 256
        # Conv CNN --> PosEnc --> MLP
        else:
            enc_channels = list(self.enc_dict['enc_channels'])  # CNN channels
            self.visual_resolution = (64, 64)  # CNN out visual resolution
            self.visual_channels = enc_channels[-1]  # CNN out visual channels

            enc_layers = len(enc_channels) - 1
            self.encoder = nn.Sequential(*[
                conv_norm_act(
                    enc_channels[i],
                    enc_channels[i + 1],
                    kernel_size=self.enc_dict['enc_ks'],
                    # 2x downsampling for 128x128 image
                    stride=2 if (i == 0 and self.resolution[0] == 128) else 1,
                    norm=self.enc_dict['enc_norm'],
                    act='relu' if i != (enc_layers - 1) else '',
                ) for i in range(enc_layers)
            ])  # relu except for the last layer

        # Build Encoder related modules
        self.encoder_pos_embedding = SoftPositionEmbed(self.visual_channels,
                                                       self.visual_resolution)
        self.encoder_out_layer = nn.Sequential(
            nn.LayerNorm(self.visual_channels),
            nn.Linear(self.visual_channels, self.enc_out_channels),
            nn.ReLU(),
            nn.Linear(self.enc_out_channels, self.enc_out_channels),
        )

    def _build_decoder(self):
        # Build Decoder
        # Spatial broadcast --> PosEnc --> DeConv CNN
        dec_channels = self.dec_dict['dec_channels']  # CNN channels
        self.dec_resolution = self.dec_dict['dec_resolution']  # broadcast size
        dec_ks = self.dec_dict['dec_ks']  # kernel size
        assert dec_channels[0] == self.slot_size, \
            'wrong in_channels for Decoder'
        modules = []
        in_size = self.dec_resolution[0]
        out_size = in_size
        stride = 2
        for i in range(len(dec_channels) - 1):
            if out_size == self.resolution[0]:
                stride = 1
            modules.append(
                deconv_norm_act(
                    dec_channels[i],
                    dec_channels[i + 1],
                    kernel_size=dec_ks,
                    stride=stride,
                    norm=self.dec_dict['dec_norm'],
                    act='relu',
                ))
            out_size = deconv_out_shape(out_size, stride, dec_ks // 2, dec_ks,
                                        stride - 1)

        assert_shape(
            self.resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. "
            "Try changing `decoder_resolution`.",
        )

        # out Conv for RGB and seg mask
        modules.append(
            nn.Conv2d(dec_channels[-1], 4, kernel_size=1, stride=1, padding=0))

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.slot_size,
                                                       self.dec_resolution)

    def _build_predictor(self):
        """Predictor as in SAVi to transition slot from time t to t+1."""
        # Build Predictor
        pred_type = self.pred_dict.get('pred_type', 'transformer')
        # Transformer (object interaction) --> LSTM (scene dynamic)
        if pred_type == 'mlp':
            self.predictor = ResidualMLPPredictor(
                [self.slot_size, self.slot_size * 2, self.slot_size],
                norm_first=self.pred_dict['pred_norm_first'],
            )
        else:
            self.predictor = TransformerPredictor(
                self.slot_size,
                self.pred_dict['pred_num_layers'],
                self.pred_dict['pred_num_heads'],
                self.pred_dict['pred_ffn_dim'],
                norm_first=self.pred_dict['pred_norm_first'],
            )
        # wrap LSTM
        if self.pred_dict['pred_rnn']:
            self.predictor = RNNPredictorWrapper(
                self.predictor,
                self.slot_size,
                self.slot_mlp_size,
                num_layers=1,
                rnn_cell='LSTM',
                sg_every=self.pred_dict['pred_sg_every'],
            )

    def _get_encoder_out(self, img):
        """Encode image, potentially add pos enc, apply MLP."""
        encoder_out = self.encoder(img).type(self.dtype)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [B, C, H, W]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [B, C, H*W]
        encoder_out = encoder_out.permute(0, 2, 1).contiguous()
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [B, H*W, enc_out_channels]
        return encoder_out

    def encode(self, img, prev_slots=None):
        """Encode from img to slots."""
        B, T, C, H, W = img.shape
        img = img.flatten(0, 1)

        encoder_out = self._get_encoder_out(img)
        encoder_out = encoder_out.unflatten(0, (B, T))
        # `encoder_out` has shape: [B, T, H*W, out_features]

        # init slots
        init_latents = self.init_latents.repeat(B, 1, 1)  # [B, N, C]

        # apply SlotAttn on video frames via reusing slots
        all_slots = []
        for idx in range(T):
            # init
            if prev_slots is None:
                latents = init_latents  # [B, N, C]
            else:
                latents = self.predictor(prev_slots)  # [B, N, C]

            # perform SA to get `slots`
            slots = self.slot_attention(encoder_out[:, idx], latents)
            all_slots.append(slots)

            # next timestep
            prev_slots = slots

        # (B, T, self.num_slots, self.slot_size)
        slots = torch.stack(all_slots, dim=1)

        return slots

    def _reset_rnn(self):
        self.predictor.reset()

    def forward(self, data_dict):
        """A wrapper for model forward.
        If the input video is too long in testing, we manually cut it.
        """
        img = data_dict['img']
        T = img.shape[1]
        if T <= self.clip_len or self.training:
            return self._forward(img, None)

        # try to find the max len to input each time
        clip_len = T
        while True:
            try:
                _ = self._forward(img[:, :clip_len], None)
                del _
                torch.cuda.empty_cache()
                break
            except RuntimeError:  # CUDA out of memory
                clip_len = clip_len // 2 + 1
        # update `clip_len`
        self.clip_len = min(max(self.clip_len, clip_len), 100)
        # no need to split
        if clip_len == T:
            return self._forward(img, None)

        # split along temporal dim
        cat_dict = None
        prev_slots = None
        for clip_idx in range(0, T, clip_len):
            out_dict = self._forward(img[:, clip_idx:clip_idx + clip_len],
                                     prev_slots)
            # because this should be in test mode, we detach the outputs
            if cat_dict is None:
                cat_dict = {k: [v.detach()] for k, v in out_dict.items()}
            else:
                for k, v in out_dict.items():
                    cat_dict[k].append(v.detach())
            prev_slots = cat_dict['slots'][-1][:, -1].detach().clone()
            del out_dict
            torch.cuda.empty_cache()
        cat_dict = {k: torch_cat(v, dim=1) for k, v in cat_dict.items()}
        return cat_dict

    def _forward(self, img, prev_slots=None):
        """Forward function.

        Args:
            img: [B, T, C, H, W]
            prev_slots: [B, num_slots, slot_size] or None,
                the `slots` from last timestep.
        """
        # reset RNN states if this is the first frame
        if prev_slots is None:
            self._reset_rnn()

        B, T = img.shape[:2]
        slots = self.encode(img, prev_slots=prev_slots)
        # `slots` has shape: [B, T, self.num_slots, self.slot_size]

        if self.testing:
            return {'slots': slots}

        # decode
        slots = slots.flatten(0, 1)
        recon_img, recons, masks, _ = self.decode(slots)

        out_dict = {
            'recon_img': recon_img,  # [B, T, 3, H, W]
            'recons': recons,  # [B, T, N, 3, H, W]
            'masks': masks,  # [B, T, N, 1, H, W]
            'slots': slots,  # [B, T, N, C]
        }
        out_dict = {k: v.unflatten(0, (B, T)) for k, v in out_dict.items()}
        return out_dict

    def decode(self, slots):
        """Decode from slots to reconstructed images and masks."""
        # `slots` has shape: [B, self.num_slots, self.slot_size].
        bs, num_slots, slot_size = slots.shape
        height, width = self.resolution
        num_channels = 3

        # spatial broadcast
        decoder_in = slots.view(bs * num_slots, slot_size, 1, 1)
        decoder_in = decoder_in.repeat(1, 1, self.dec_resolution[0],
                                       self.dec_resolution[1])

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [B*num_slots, 4, H, W].

        out = out.view(bs, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]  # [B, num_slots, 3, H, W]
        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)  # [B, num_slots, 1, H, W]
        recon_combined = torch.sum(recons * masks, dim=1)  # [B, 3, H, W]
        return recon_combined, recons, masks, slots

    def calc_train_loss(self, data_dict, out_dict):
        """Compute loss that are general for SlotAttn models."""
        loss_dict = {}
        recon_img = out_dict['recon_img']
        img = data_dict['img']
        loss_dict['img_recon_loss'] = F.mse_loss(recon_img, img)
        return loss_dict

    @property
    def dtype(self):
        return self.slot_attention.dtype

    @property
    def device(self):
        return self.slot_attention.device
