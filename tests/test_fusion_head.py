import torch
from production.brain.empathy_fusion_head import EmpathyFusionHead
from production.brain.personality_vector import PersonalityEMA


def test_fusion_head_softmax_and_ema():
    model = EmpathyFusionHead(8, 16, 4)
    feats = {
        'emotion': torch.randn(1, 8),
        'context': torch.randn(1, 16),
        'memory': torch.randn(1, 4),
    }
    p = model(feats)
    assert p.shape[-1] == 5
    s = float(p.sum())
    assert abs(s - 1.0) < 1e-5
    ema = PersonalityEMA()
    v1 = ema.update(p[0])
    v2 = ema.update(p[0])
    assert v1.shape[0] == 5 and v2.shape[0] == 5
    assert abs(float(v2.sum()) - 1.0) < 1e-5




