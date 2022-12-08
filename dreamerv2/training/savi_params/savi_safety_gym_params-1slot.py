from nerv.training import BaseParams


class SlotAttentionParams(BaseParams):
    project = 'Imitation-Learning-Project'

    # model configs
    savi_path = './savi/weights/safetygym-1slot.pth'
    model = 'SAVi'
    resolution = (64, 64)  # SAVi paper uses 128x128

    # Slot Attention
    slot_size = 128 * 7
    num_slots = 1
    slot_dict = dict(
        num_slots=num_slots,
        slot_size=slot_size,
        slot_mlp_size=slot_size * 2,
        num_iterations=2,
    )

    # CNN Encoder
    enc_dict = dict(
        enc_channels=(3, 64, 64, 64, 64),
        enc_ks=5,
        enc_out_channels=slot_size,
        enc_norm='',
    )

    # CNN Decoder
    dec_dict = dict(
        dec_channels=(slot_size, 64, 64, 64, 64),
        dec_resolution=(8, 8),
        dec_ks=5,
        dec_norm='',
    )

    # Predictor
    pred_dict = dict(
        pred_type='transformer',
        pred_rnn=False,
        pred_norm_first=True,
        pred_num_layers=2,
        pred_num_heads=4,
        pred_ffn_dim=slot_size * 4,
        pred_sg_every=None,
    )
