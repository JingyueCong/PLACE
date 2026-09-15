import os
import copy
import torch

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from ..utils import NameTimer
from .peft_util import find_all_linear_names


def _flash_attn_available():
    try:
        import flash_attn  # noqa: F401
        import flash_attn_2_cuda  # noqa: F401
        return True
    except Exception:
        return False


def _attn_kwargs():
    """Select an explicit backend, or preserve the legacy FA2 auto-detection."""
    requested = os.environ.get("ULD_ATTN_IMPLEMENTATION")
    if requested:
        allowed = {"eager", "sdpa", "flash_attention_2"}
        if requested not in allowed:
            raise ValueError(
                "ULD_ATTN_IMPLEMENTATION must be one of "
                f"{sorted(allowed)}, got {requested!r}"
            )
        return {"attn_implementation": requested}
    return {"use_flash_attention_2": True} if _flash_attn_available() else {}


def get_dtype(data_type):
    if data_type == 'bfloat16':
        return torch.bfloat16
    elif data_type == 'float16':
        return torch.float16

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def copy_weights(base_llm, model, layer_indices=None):
    config = model.config
    name = model.config._name_or_path.lower()
    if ('llama' in name) or ('zephyr' in name) or ('mistral' in name):
        if layer_indices is None:
            layer_indices = list(range(config.num_hidden_layers))
        print(f"Copying {name}: small layers ← base layers {layer_indices}")
        model.model.embed_tokens.load_state_dict(
            base_llm.model.embed_tokens.state_dict()
        )
        model.model.norm.load_state_dict(
            base_llm.model.norm.state_dict()
        )
        for small_idx, base_idx in enumerate(layer_indices):
            model.model.layers[small_idx].load_state_dict(
                base_llm.model.layers[base_idx].state_dict()
            )
        model.lm_head.load_state_dict(
            base_llm.lm_head.state_dict()
        )
        return model
    else:
        raise ValueError(f"Unsupported model: {name}")

def init_small_llm(origin_config, num_layer, device, hparams=None, base_llm=None, saved_path=None, layer_indices=None):
    config = copy.deepcopy(origin_config)
    if layer_indices is not None:
        layer_indices = list(layer_indices)
        config.num_hidden_layers = len(layer_indices)
    else:
        config.num_hidden_layers = num_layer
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        **_attn_kwargs(),
    ).to('cpu')

    if base_llm is not None:
        copy_weights(base_llm, model, layer_indices=layer_indices)

    if saved_path is not None:
        model.load_state_dict(
            torch.load(saved_path)
        )

    return model

def create_full_model(model_path, num_layer=0 ,data_type='bfloat16', layer_indices=None, **kwargs):
    with NameTimer("Init full model"):
        basellm = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=get_dtype(data_type),
            trust_remote_code=True,
            **_attn_kwargs(),
        )
        if num_layer != 0 or layer_indices is not None: #! Construct the small model
            basellm = init_small_llm(
                basellm.model.config,
                num_layer=num_layer,
                base_llm=basellm,
                layer_indices=layer_indices,
                device='cpu',
            )
        return basellm

def create_peft_model(model_path, Lora, baseoutdir, num_layer=0, data_type='bfloat16', layer_indices=None, **kwargs):
    with NameTimer("Init peft model"):
        basellm = create_full_model(model_path, num_layer, data_type, layer_indices=layer_indices)
        if num_layer != 0 or layer_indices is not None:
            #! We save the extracted small LLM to disk to speed up test time model loading
            basellm.save_pretrained(os.path.join(baseoutdir, 'fullmodel'))
        peftconfig = LoraConfig(
            r=Lora.r,
            lora_alpha=Lora.alpha,
            target_modules=find_all_linear_names(basellm),
            lora_dropout=Lora.dropout,
            bias=Lora.bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(basellm, peftconfig)
        return model
