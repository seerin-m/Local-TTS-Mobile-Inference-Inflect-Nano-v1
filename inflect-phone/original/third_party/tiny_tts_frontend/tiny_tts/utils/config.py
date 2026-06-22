# Audio
SAMPLING_RATE = 44100
FILTER_LENGTH = 2048
HOP_LENGTH = 512
SEGMENT_FRAMES = 32
ADD_BLANK = True
SPEC_CHANNELS = FILTER_LENGTH // 2 + 1  # 1025
N_MEL_CHANNELS = 128                      # updated in new checkpoint

# Speakers
N_SPEAKERS = 1
SPK2ID = {"MALE": 0}

# Model — matches config.json for G_150000.pth (lighter version)
MODEL_PARAMS = dict(
    use_spk_conditioned_encoder=True,
    use_noise_scaled_mas=True,
    inter_channels=32,
    hidden_channels=32,
    filter_channels=128,
    n_heads=2,
    n_layers=3,
    n_layers_trans_flow=3,
    kernel_size=3,
    p_dropout=0.1,
    resblock="1",
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[8, 8, 2, 2, 2],
    upsample_initial_channel=64,
    upsample_kernel_sizes=[16, 16, 8, 2, 2],
    n_layers_q=3,
    use_spectral_norm=False,
    gin_channels=128,
    use_sdp=True,
    mas_noise_scale_initial=0.01,
    noise_scale_delta=2e-06,
)

# Language / Tone
NUM_LANGUAGES = 1
NUM_TONES = 6
