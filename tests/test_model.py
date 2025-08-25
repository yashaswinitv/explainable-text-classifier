from xtc.model import build_model

def test_model():
    m = build_model(4)
    assert m.config.num_labels == 4
