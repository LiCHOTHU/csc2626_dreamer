from .savi import SAVi


def build_savi(params):
    """Build model."""
    model = SAVi(
        resolution=params.resolution,
        clip_len=50,
        slot_dict=params.slot_dict,
        enc_dict=params.enc_dict,
        dec_dict=params.dec_dict,
        pred_dict=params.pred_dict,
    )
    return model
