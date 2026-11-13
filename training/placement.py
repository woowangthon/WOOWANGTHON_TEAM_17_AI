"""Split a tied Qwen text model across CPU embedding/head and CUDA decoder."""
from __future__ import annotations

from typing import Any


def _base_model(model: Any) -> Any:
    getter = getattr(model, "get_base_model", None)
    return getter() if callable(getter) else model


def trainable_parameter_names(model: Any) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def place_cpu_shared_embedding_head(model: Any, torch: Any, device: Any) -> dict[str, Any]:
    """Keep tied token embedding and output head on CPU; place decoder/adapters on device.

    The embedding/head weights stay the same shared Parameter and are converted to
    float32 in-place. Decoder operations and LoRA modules use the requested device.
    """
    base = _base_model(model)
    config = getattr(base, "config", None)
    if not bool(getattr(config, "tie_word_embeddings", False)):
        raise ValueError("CPU shared-head placement requires tie_word_embeddings=true")
    embedding = base.get_input_embeddings()
    head = base.get_output_embeddings()
    if embedding is None or head is None or getattr(embedding, "weight", None) is not getattr(head, "weight", None):
        raise ValueError("Input embedding and output head are not the same shared Parameter")
    decoder = getattr(base, "model", None)
    if decoder is None:
        raise ValueError("Causal LM does not expose its text decoder as .model")

    embedding.to(device="cpu")
    # data assignment keeps Parameter identity and the tied reference intact.
    embedding.weight.data = embedding.weight.data.to(device="cpu", dtype=torch.float32)
    for child in decoder.children():
        if child is embedding:
            continue
        # Move only: preserve pretrained bf16, LoRA fp32 and rotary buffer dtypes.
        child.to(device=device)
    head.to(device="cpu")
    if head.weight is not embedding.weight or head.weight.device.type != "cpu" or head.weight.dtype != torch.float32:
        raise RuntimeError("Tied CPU embedding/head placement was not preserved")
    if embedding.weight.requires_grad:
        raise RuntimeError("Shared embedding/head must remain frozen")

    trainable = trainable_parameter_names(model)
    if not trainable:
        raise RuntimeError("No trainable adapter parameters were found")
    bad_names = [name for name in trainable if "embed_tokens" in name or "lm_head" in name]
    if bad_names:
        raise RuntimeError(f"LoRA must not target tied embedding/head parameters: {bad_names[:3]}")
    target = torch.device(device)
    misplaced_decoder = [(name, str(parameter.device)) for name, parameter in decoder.named_parameters()
                         if parameter is not embedding.weight and parameter.device != target]
    misplaced_buffers = [(name, str(buffer.device)) for name, buffer in decoder.named_buffers()
                         if buffer.device != target]
    if misplaced_decoder or misplaced_buffers:
        raise RuntimeError(f"Decoder tensors did not all move to {target}: "
                           f"parameters={misplaced_decoder[:2]}, buffers={misplaced_buffers[:2]}")
    bad_devices = [(name, str(parameter.device)) for name, parameter in model.named_parameters()
                   if parameter.requires_grad and parameter.device != target]
    if bad_devices:
        raise RuntimeError(f"Trainable parameters are not all on {target}: {bad_devices[:3]}")
    trainable_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters() if parameter.requires_grad})
    decoder_dtypes = sorted({str(parameter.dtype) for name, parameter in model.named_parameters()
                             if parameter.device == torch.device(device)})
    return {"mode": "cpu_shared_embedding_head", "embeddingDevice": str(embedding.weight.device),
            "embeddingDtype": str(embedding.weight.dtype), "headDevice": str(head.weight.device),
            "headDtype": str(head.weight.dtype), "decoderDevice": str(target),
            "trainableParameterCount": len(trainable), "sharedWeight": True,
            "sharedWeightFrozen": True, "trainableDtypes": trainable_dtypes,
            "decoderDtypes": decoder_dtypes, "headPrecision": "float32_cpu"}


def completion_loss_cpu_shared(model: Any, input_ids: list[int], completion_start: int,
                               torch: Any, device: Any) -> Any:
    """Compute assistant-only loss while retaining gradient across CPU/GPU copies."""
    base = _base_model(model)
    embedding = base.get_input_embeddings()
    head = base.get_output_embeddings()
    ids_cpu = torch.tensor([input_ids], dtype=torch.long, device="cpu")
    start = completion_start
    if start < 1 or start >= len(input_ids):
        raise ValueError("Completion start is outside the token sequence")
    # Frozen CPU embedding, then differentiable copy to decoder device.
    with torch.no_grad():
        inputs_embeds = embedding(ids_cpu)
    inputs_embeds = inputs_embeds.to(device=device, dtype=torch.bfloat16 if str(device).startswith("cuda") else inputs_embeds.dtype)
    attention_mask = torch.ones(ids_cpu.shape, dtype=torch.long, device=device)
    result = base.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                        use_cache=False, return_dict=True)
    hidden = result.last_hidden_state
    positions = torch.arange(start - 1, len(input_ids) - 1, dtype=torch.long, device=device)
    selected_hidden = hidden.index_select(1, positions).to(device="cpu", dtype=torch.float32)
    # This cross-device copy remains in autograd; gradients return to CUDA LoRA weights.
    logits = torch.nn.functional.linear(selected_hidden, head.weight, getattr(head, "bias", None))
    targets = torch.tensor([input_ids[start:]], dtype=torch.long, device="cpu")
    if logits.shape[1] != targets.shape[1]:
        raise RuntimeError("CPU shared-head logits do not align with completion targets")
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
