# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics.nn.extra_modules.transformer import LRPB_Attention, LRPB_AIFI  # noqa: E402

S5_SHAPES = {
    '416x416': (13, 13),
    '640x640': (20, 20),
    '416x640': (13, 20),
    '640x416': (20, 13),
}
C = 1024
NUM_HEADS = 8
BATCH = 2


def _feat(h, w, c=C):
    return torch.randn(BATCH, c, h, w)


def test_lrpb_attention_accepts_every_feature_map_size():
    attn = LRPB_Attention(C, None, num_heads=NUM_HEADS)
    for tag, (h, w) in S5_SHAPES.items():
        x = _feat(h, w).flatten(2).permute(0, 2, 1)
        y = attn(x, hw=(h, w))
        assert y.shape == x.shape, '%s: %s != %s' % (tag, tuple(y.shape), tuple(x.shape))
        assert torch.isfinite(y).all(), '%s produced non-finite output' % tag


def test_lrpb_encoder_layer_runs_at_paper_resolutions():
    layer = LRPB_AIFI(C)
    layer.eval()
    for tag, (h, w) in S5_SHAPES.items():
        with torch.no_grad():
            y = layer(_feat(h, w))
        assert y.shape == (BATCH, C, h, w), '%s: %s' % (tag, tuple(y.shape))


def test_tables_are_cached_per_resolution_and_are_not_shared():
    attn = LRPB_Attention(C, None, num_heads=NUM_HEADS)
    x13 = _feat(13, 13).flatten(2).permute(0, 2, 1)
    x20 = _feat(20, 20).flatten(2).permute(0, 2, 1)

    attn(x13, hw=(13, 13))
    n_after_first = len(attn._bias_table_cache)
    attn(x13, hw=(13, 13))
    assert len(attn._bias_table_cache) == n_after_first, 'tables must be cached per resolution'

    attn(x20, hw=(20, 20))
    assert len(attn._bias_table_cache) > n_after_first, 'a new resolution must add a table'

    b13, i13 = attn._get_bias_tables(13, 13, x13.device, x13.dtype)
    b20, i20 = attn._get_bias_tables(20, 20, x20.device, x20.dtype)
    assert b13.shape == (2 * 13 - 1, 2 * 13 - 1, 2), tuple(b13.shape)
    assert b20.shape == (2 * 20 - 1, 2 * 20 - 1, 2), tuple(b20.shape)
    assert i13.shape == (169, 169) and i20.shape == (400, 400)
    assert i13.max() < (2 * 13 - 1) * (2 * 13 - 1)


def test_lrpb_bias_depends_only_on_relative_offset():
    layer = LRPB_AIFI(C)
    layer.eval()
    torch.manual_seed(0)
    a = torch.randn(1, C, 13, 13)
    b = torch.randn(1, C, 13, 13)
    with torch.no_grad():
        ba = layer.lrpb_attention.pos(layer.lrpb_attention._get_bias_tables(13, 13, a.device, a.dtype)[0])
        bb = layer.lrpb_attention.pos(layer.lrpb_attention._get_bias_tables(13, 13, b.device, b.dtype)[0])
    assert torch.allclose(ba, bb), 'position bias must not depend on the input content'


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print('PASS  %s' % fn.__name__)
        except Exception as e:
            failed += 1
            print('FAIL  %s -> %s' % (fn.__name__, e))
    print('\n%d/%d passed' % (len(fns) - failed, len(fns)))
    sys.exit(1 if failed else 0)
