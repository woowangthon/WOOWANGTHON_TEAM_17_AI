import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from training.placement import (completion_loss_cpu_shared, place_cpu_shared_embedding_head,
                               trainable_parameter_names)


if torch is not None:
    class ToyLayer(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.base = torch.nn.Linear(width, width, bias=False)
            self.lora_a = torch.nn.Parameter(torch.randn(width, 2) * 0.1)
            self.lora_b = torch.nn.Parameter(torch.randn(2, width) * 0.1)
            self.base.weight.requires_grad_(False)

        def forward(self, x):
            return torch.tanh(self.base(x) + (x @ self.lora_a @ self.lora_b) * 0.1)


    class ToyDecoder(torch.nn.Module):
        def __init__(self, vocab, width):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(vocab, width)
            self.layers = torch.nn.ModuleList([ToyLayer(width)])
            self.norm = torch.nn.LayerNorm(width)
            self.norm.weight.requires_grad_(False)
            self.norm.bias.requires_grad_(False)

        def forward(self, inputs_embeds, **kwargs):
            x = inputs_embeds
            for layer in self.layers:
                x = layer(x)
            return SimpleNamespace(last_hidden_state=self.norm(x))


    class ToyCausalLM(torch.nn.Module):
        def __init__(self, vocab=11, width=8):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=True)
            self.model = ToyDecoder(vocab, width)
            self.lm_head = torch.nn.Linear(width, vocab, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight
            self.model.embed_tokens.weight.requires_grad_(False)

        def get_base_model(self):
            return self

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids):
            hidden = self.model(inputs_embeds=self.model.embed_tokens(input_ids)).last_hidden_state
            return SimpleNamespace(logits=torch.nn.functional.linear(hidden, self.lm_head.weight))


@unittest.skipIf(torch is None, "torch is an optional training dependency")
class TrainingPlacementTests(unittest.TestCase):
    def test_tied_cpu_head_loss_and_lora_gradient_match_reference(self):
        torch.manual_seed(19)
        model = ToyCausalLM()
        input_ids = [1, 2, 3, 4, 5, 6]
        start = 3
        ids = torch.tensor([input_ids], dtype=torch.long)
        full_logits = model(ids).logits
        expected = torch.nn.functional.cross_entropy(
            full_logits[:, start - 1:len(input_ids) - 1, :].reshape(-1, 11),
            torch.tensor(input_ids[start:], dtype=torch.long))
        expected.backward()
        expected_grads = {name: parameter.grad.detach().clone()
                          for name, parameter in model.named_parameters() if parameter.requires_grad}
        reference_update = {name: parameter.detach().clone() - 0.01 * parameter.grad.detach()
                            for name, parameter in model.named_parameters() if parameter.requires_grad}

        # Reload identical initial weights, then run the CPU-shared implementation.
        torch.manual_seed(19)
        placed = ToyCausalLM()
        initial_base = {name: parameter.detach().clone() for name, parameter in placed.named_parameters()
                        if not parameter.requires_grad}
        info = place_cpu_shared_embedding_head(placed, torch, torch.device("cpu"))
        self.assertTrue(info["sharedWeight"])
        self.assertTrue(info["sharedWeightFrozen"])
        self.assertEqual(info["trainableDtypes"], ["torch.float32"])
        self.assertIs(placed.get_input_embeddings().weight, placed.get_output_embeddings().weight)
        self.assertEqual(placed.get_output_embeddings().weight.dtype, torch.float32)
        actual = completion_loss_cpu_shared(placed, input_ids, start, torch, torch.device("cpu"))
        actual.backward()
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)
        for name, parameter in placed.named_parameters():
            if parameter.requires_grad:
                self.assertTrue(torch.allclose(parameter.grad, expected_grads[name], atol=1e-6, rtol=1e-6), name)
                updated = parameter.detach() - 0.01 * parameter.grad.detach()
                self.assertTrue(torch.allclose(updated, reference_update[name], atol=1e-6, rtol=1e-6), name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertTrue(torch.equal(parameter.detach(), initial_base[name]), name)
        trainable_names = trainable_parameter_names(placed)
        self.assertTrue(trainable_names)
        self.assertTrue(all("lora_" in name for name in trainable_names))
        self.assertFalse(any("embed_tokens" in name or "lm_head" in name for name in trainable_names))


if __name__ == "__main__":
    unittest.main()
