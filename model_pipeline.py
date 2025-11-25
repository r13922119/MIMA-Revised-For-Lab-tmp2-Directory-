from typing import Callable, Optional
import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from accelerate.logging import get_logger

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.models.attention import Attention
from diffusers.utils.import_utils import is_xformers_available
from diffusers.configuration_utils import FrozenDict

import pdb


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

logger = get_logger(__name__)

def set_use_memory_efficient_attention_xformers(
    self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
):
    if use_memory_efficient_attention_xformers:
        if self.added_kv_proj_dim is not None:
            # TODO(Anton, Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
            # which uses this type of cross attention ONLY because the attention mask of format
            # [0, ..., -10.000, ..., 0, ...,] is not supported
            raise NotImplementedError(
                "Memory efficient attention with `xformers` is currently not supported when"
                " `self.added_kv_proj_dim` is defined."
            )
        elif not is_xformers_available():
            raise ModuleNotFoundError(
                (
                    "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                    " xformers"
                ),
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                " only available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e

        processor = MIMAXFormersAttnProcessor(attention_op=attention_op)
    else:
        processor = MIMAAttnProcessor()

    self.set_processor(processor)


class MIMAAttnProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):  
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        crossattn = False
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :]*0.
            key = detach*key + (1-detach)*key.detach()
            value = detach*value + (1-detach)*value.detach()

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class MIMAXFormersAttnProcessor:
    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        crossattn = False
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.cross_attention_norm:
                encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :]*0.
            key = detach*key + (1-detach)*key.detach()
            value = detach*value + (1-detach)*value.detach()

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class MIMAPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for IMMA Pro model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
        modifier_token: list of new modifier tokens added or to be added to text_encoder
        modifier_token_id: list of id of new modifier tokens added or to be added to text_encoder
    """
    # FIX: Removed "modifier_token" (it's config, not a component)
    # FIX: Support "image_encoder" (it is a component)
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
        modifier_token: list = [],
        modifier_token_id: list = [],
        image_encoder = None, # FIX: Support image_encoder
    ):
        super().__init__(vae,
                         text_encoder,
                         tokenizer,
                         unet,
                         scheduler,
                         safety_checker,
                         feature_extractor,
                         requires_safety_checker=requires_safety_checker)

        # FIX: Register image_encoder as a module (so .to(cuda) works)
        self.register_modules(image_encoder=image_encoder)

        # FIX: Register modifier tokens as CONFIG, not modules
        self.register_to_config(modifier_token=modifier_token, modifier_token_id=modifier_token_id)

        # change attn class
        self.modifier_token = modifier_token
        self.modifier_token_id = modifier_token_id

    def add_token(self, initializer_token):
        initializer_token_id = []
        for modifier_token_, initializer_token_ in zip(self.modifier_token, initializer_token):
            # Add the placeholder token in tokenizer
            num_added_tokens = self.tokenizer.add_tokens(modifier_token_)
            
            # FIX: Do not crash if num_added_tokens == 0, just pass
            if num_added_tokens == 0:
                pass 
                # Original File raised ValueError here.

            # Convert the initializer_token, placeholder_token to ids
            token_ids = self.tokenizer.encode([initializer_token_], add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            self.modifier_token_id.append(self.tokenizer.convert_tokens_to_ids(modifier_token_))
            initializer_token_id.append(token_ids[0])
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        for (x, y) in zip(self.modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]

    def save_pretrained(self, save_path, query_weight="attn2"):
            # Create a dictionary to hold all necessary data
            delta_dict = {}
            
            # 1. Save the Delta Weights (Attention layers)
            unet_weights = {}
            for name, params in self.unet.named_parameters():
                if 'attn2' in name:
                    unet_weights[name] = params.cpu().clone()
            delta_dict['unet'] = unet_weights

            # 2. Save Modifier Tokens (if they exist)
            # This allows the model to remember the specific tokens you trained
            if hasattr(self, 'modifier_token') and self.modifier_token:
                token_dict = {}
                # We need the embeddings for these tokens
                token_embeds = self.text_encoder.get_input_embeddings().weight.data
                
                for token in self.modifier_token:
                    # Get the ID
                    ids = self.tokenizer.convert_tokens_to_ids(token)
                    # Get the vector
                    token_dict[token] = token_embeds[ids].cpu().clone()
                
                delta_dict['modifier_token'] = token_dict

            # 3. Save as a SINGLE file (matches train.py expectation)
            torch.save(delta_dict, save_path)

    def load_model(self, save_path, compress=False):
        st = torch.load(save_path)
        
        # FIX: Detect structure (Flat vs Nested)
        if 'unet' in st:
            unet_state = st['unet']
        else:
            unet_state = st

        if 'text_encoder' in st:
            self.text_encoder.load_state_dict(st['text_encoder'])
        if 'modifier_token' in st:
            # FIX: Handle config dict vs direct list
            if isinstance(st['modifier_token'], dict):
                modifier_tokens = list(st['modifier_token'].keys())
                token_values = st['modifier_token']
            else:
                # FIX: Fallback if structure is different, though previous code implied dict
                modifier_tokens = st['modifier_token']
                token_values = None

            modifier_token_id = []
            for modifier_token in modifier_tokens:
                num_added_tokens = self.tokenizer.add_tokens(modifier_token)
                if num_added_tokens == 0:
                    pass # FIX: Do not crash if num_added_tokens == 0, just pass.  Original File raised ValueError here.
                modifier_token_id.append(self.tokenizer.convert_tokens_to_ids(modifier_token))
            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            
            if token_values:
                token_embeds = self.text_encoder.get_input_embeddings().weight.data
                for i, id_ in enumerate(modifier_token_id):
                    token_embeds[id_] = token_values[modifier_tokens[i]]

        for name, params in self.unet.named_parameters():
            if 'attn2' in name:
                # FIX: [HYBRID LOGIC] 
                # 1. Use the original logic for Matrix Decomposition (compress=True)
                # 2. Use the 'unet_state' variable to handle structure robustly
                if compress and ('to_k' in name or 'to_v' in name):
                    if name in unet_state:
                        params.data += unet_state[name]['u']@unet_state[name]['v']
                # FIX: Standard robust loading
                elif name in unet_state:
                    params.data.copy_(unet_state[f'{name}'])
