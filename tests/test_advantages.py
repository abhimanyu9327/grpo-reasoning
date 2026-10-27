import torch

from grpo.grpo import group_advantages


def test_group_advantages():
    G = 4
    rewards = torch.tensor([
        1.0, 0.1, 0.1, 0.0,   # mixed group
        0.0, 1.0, 1.0, 0.1,   # another mixed group
        0.1, 0.1, 0.1, 0.1,   # all equal: nothing to learn
        1.0, 1.0, 1.0, 1.0,   # all equal: nothing to learn
    ])
    adv = group_advantages(rewards, G).view(-1, G)

    assert not torch.isnan(adv).any()
    for group in adv[:2]:
        assert abs(group.mean().item()) < 1e-5
        assert abs(group.std().item() - 1.0) < 1e-4
    assert torch.all(adv[2:] == 0)
    # Best answer in a group gets the highest advantage.
    assert adv[0].argmax().item() == 0
