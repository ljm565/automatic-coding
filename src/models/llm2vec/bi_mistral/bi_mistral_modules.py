from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralFlashAttention2,
    MistralSdpaAttention,
)



class ModifiedMistralAttention(MistralAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedMistralFlashAttention2(MistralFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedMistralSdpaAttention(MistralSdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


MISTRAL_ATTENTION_CLASSES = {
    "eager": ModifiedMistralAttention,
    "flash_attention_2": ModifiedMistralFlashAttention2,
    "sdpa": ModifiedMistralSdpaAttention,
}