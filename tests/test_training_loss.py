import unittest
from types import SimpleNamespace

from training.train import Encoded, _completion_loss

try:
    import torch
except ImportError:
    torch = None


class FakeTailModel:
    def __init__(self):
        self.last_positions = None

    def __call__(self, input_ids, logits_to_keep=None):
        length = input_ids.shape[1]
        vocab = 8
        # Every position has distinguishable logits, so an off-by-one shift changes loss.
        full = torch.arange(length * vocab, dtype=torch.float32).reshape(1, length, vocab) / 7
        if logits_to_keep is not None:
            self.last_positions = logits_to_keep.detach().cpu().tolist()
            full = full[:, logits_to_keep, :]
        return SimpleNamespace(logits=full)


@unittest.skipIf(torch is None, "torch is an optional training dependency")
class TrainingLossTests(unittest.TestCase):
    def test_tail_logits_match_full_sequence_completion_only_loss(self):
        tokens = [1, 2, 3, 4, 5, 6]
        start = 3
        item = Encoded(input_ids=tokens, labels=[-100] * start + tokens[start:], completion_start=start)
        model = FakeTailModel()

        input_ids = torch.tensor([tokens], dtype=torch.long)
        full = model(input_ids=input_ids).logits
        expected_logits = full[:, start - 1:len(tokens) - 1, :]
        targets = torch.tensor([tokens[start:]], dtype=torch.long)
        expected = torch.nn.functional.cross_entropy(expected_logits.reshape(-1, 8), targets.reshape(-1))

        actual = _completion_loss(model, item, torch, torch.device("cpu"))

        self.assertEqual(model.last_positions, list(range(start - 1, len(tokens) - 1)))
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)


if __name__ == "__main__":
    unittest.main()
