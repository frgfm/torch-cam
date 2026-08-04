import copy
from contextlib import nullcontext
from functools import partial

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.vision_transformer import VisionTransformer

from torchcam.methods import LayerCAM, LeGrad, ScoreCAM
from torchcam.metrics import ClassificationMetric


def _build_vit() -> VisionTransformer:
    torch.manual_seed(0)
    model = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        mlp_dim=64,
        num_classes=10,
    )
    torch.nn.init.normal_(model.heads.head.weight)
    return model.eval()


def _reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[:, 1:].reshape(tensor.shape[0], 4, 4, tensor.shape[-1]).permute(0, 3, 1, 2)


def test_reshape_transform_validation():
    model = _build_vit()
    target_layer = "encoder.layers.encoder_layer_1.ln_1"

    with pytest.raises(TypeError, match="reshape_transform"):
        LayerCAM(model, target_layer, reshape_transform=1)
    with pytest.raises(ValueError, match="target_layer"):
        LayerCAM(model, reshape_transform=_reshape_transform)


def test_layercam_with_reshape_transform():
    model = _build_vit()
    input_tensor = torch.randn(1, 3, 32, 32)

    with LayerCAM(
        model,
        "encoder.layers.encoder_layer_1.ln_1",
        reshape_transform=_reshape_transform,
    ) as extractor:
        scores = model(input_tensor)
        cam = extractor(scores[0].argmax().item(), scores)[0]

        assert extractor.hook_a[0].shape == (1, 32, 4, 4)
        assert extractor.hook_g[0].shape == (1, 32, 4, 4)
        assert cam.shape == (1, 4, 4)
        assert torch.isfinite(cam).all()
        assert cam.std() > 0


def test_scorecam_with_reshape_transform():
    model = _build_vit()
    input_tensor = torch.randn(1, 3, 32, 32)

    with ScoreCAM(
        model,
        "encoder.layers.encoder_layer_1.ln_1",
        batch_size=16,
        reshape_transform=_reshape_transform,
    ) as extractor:
        scores = model(input_tensor)
        cam = extractor(scores[0].argmax().item())[0]

        assert extractor.hook_a[0].shape == (1, 32, 4, 4)
        assert cam.shape == (1, 4, 4)
        assert torch.isfinite(cam).all()
        assert cam.std() > 0


class _TinyBlock(nn.Module):
    def __init__(
        self,
        hidden_dim=8,
        num_heads=2,
        *,
        batch_first=True,
        attention_dropout=0.0,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=batch_first,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, input_tensor):
        x = self.ln_1(input_tensor)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x += input_tensor
        return x + self.mlp(self.ln_2(x))


class _CrossAttentionBlock(_TinyBlock):
    def forward(self, input_tensor):
        x = self.ln_1(input_tensor)
        memory = x.roll(1, dims=1)
        x, _ = self.self_attention(x, memory, memory, need_weights=False)
        x += input_tensor
        return x + self.mlp(self.ln_2(x))


class _PositionalAttentionBlock(_TinyBlock):
    def forward(self, input_tensor):
        x = self.ln_1(input_tensor)
        x, _ = self.self_attention(x, x, x, None, False, None, True)
        x += input_tensor
        return x + self.mlp(self.ln_2(x))


class _InvalidOutputBlock(_TinyBlock):
    def forward(self, input_tensor):
        return super().forward(input_tensor).mean(dim=1)


class _DropTokenBlock(_TinyBlock):
    def forward(self, input_tensor):
        return super().forward(input_tensor)[:, :-1]


class _NoWeightsAttention(nn.MultiheadAttention):
    def forward(self, *args, **kwargs):
        output, _ = super().forward(*args, **kwargs)
        return output, None


class _AveragedWeightsAttention(nn.MultiheadAttention):
    def forward(self, *args, **kwargs):
        output, weights = super().forward(*args, **kwargs)
        assert isinstance(weights, torch.Tensor)
        return output, weights.mean(dim=1)


class _ShortWeightsAttention(nn.MultiheadAttention):
    def forward(self, *args, **kwargs):
        output, weights = super().forward(*args, **kwargs)
        assert isinstance(weights, torch.Tensor)
        return output, weights[..., :-1]


def _separate_projection_block():
    block = _TinyBlock()
    block.self_attention = nn.MultiheadAttention(8, 2, kdim=4, vdim=4, batch_first=True)
    return block


class _TinyViT(nn.Module):
    def __init__(self, grid_shape=(2, 2), prefix_tokens=1, num_layers=2, *, attention_dropout=0.0):
        super().__init__()
        self.grid_shape = grid_shape
        self.patch_embed = nn.Conv2d(3, 8, 1)
        self.prefix = nn.Parameter(torch.randn(1, prefix_tokens, 8))
        self.blocks = nn.ModuleList([_TinyBlock(attention_dropout=attention_dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 3)

    def project(self, tokens):
        return self.head(self.norm(tokens.mean(dim=1)))

    def forward(self, input_tensor):
        tokens = self.patch_embed(input_tensor).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.prefix.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.project(tokens)


def _manual_forward(model, input_tensor):
    tokens = model.patch_embed(input_tensor).flatten(2).transpose(1, 2)
    tokens = torch.cat((model.prefix.expand(tokens.shape[0], -1, -1), tokens), dim=1)
    attentions, layer_scores = [], []

    for block in model.blocks:
        normalized = block.ln_1(tokens)
        attention = block.self_attention
        q, k, v = F.linear(normalized, attention.in_proj_weight, attention.in_proj_bias).chunk(3, dim=-1)
        batch_size, token_count, embed_dim = q.shape
        head_dim = embed_dim // attention.num_heads
        q = q.reshape(batch_size, token_count, attention.num_heads, head_dim).transpose(1, 2)
        k = k.reshape(batch_size, token_count, attention.num_heads, head_dim).transpose(1, 2)
        v = v.reshape(batch_size, token_count, attention.num_heads, head_dim).transpose(1, 2)
        probabilities = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / head_dim**0.5, dim=-1)
        output = torch.matmul(probabilities, v).transpose(1, 2).reshape(batch_size, token_count, embed_dim)
        output = attention.out_proj(output)
        x = tokens + output
        tokens = x + block.mlp(block.ln_2(x))
        attentions.append(probabilities)
        layer_scores.append(model.project(tokens))

    return model.project(tokens), attentions, layer_scores


def _manual_legrad(model, input_tensor, class_idx, target_indices, prefix_tokens, grid_shape):
    output, attentions, layer_scores = _manual_forward(model, input_tensor)
    maps = []
    for idx, layer_idx in enumerate(target_indices):
        scores = layer_scores[layer_idx]
        selected = (
            scores[:, class_idx]
            if isinstance(class_idx, int)
            else scores.gather(1, torch.tensor(class_idx).view(-1, 1)).squeeze(1)
        )
        grad = torch.autograd.grad(selected.sum(), attentions[layer_idx], retain_graph=idx < len(target_indices) - 1)[0]
        relevance = grad.relu().mean(dim=(1, 2))[:, prefix_tokens:]
        maps.append(relevance.reshape(relevance.shape[0], *grid_shape))

    cam = torch.stack(maps).mean(dim=0)
    cam -= cam.flatten(1).min(dim=-1).values[:, None, None]
    cam /= cam.flatten(1).max(dim=-1).values[:, None, None] + 1e-8
    return output, [attentions[idx] for idx in target_indices], cam


@pytest.mark.parametrize(
    ("grid_shape", "prefix_tokens", "target_indices", "batch_size", "class_idx", "infer_grid"),
    [
        ((2, 2), 1, [0], 1, 1, True),
        ((2, 3), 2, [0, 1], 2, [0, 2], False),
    ],
)
def test_legrad_matches_manual_oracle(
    grid_shape,
    prefix_tokens,
    target_indices,
    batch_size,
    class_idx,
    infer_grid,
):
    torch.manual_seed(0)
    model = _TinyViT(grid_shape, prefix_tokens)
    oracle_model = copy.deepcopy(model)
    input_tensor = torch.randn(batch_size, 3, *grid_shape)
    expected_scores, expected_attention, expected_cam = _manual_legrad(
        oracle_model,
        input_tensor,
        class_idx,
        target_indices,
        prefix_tokens,
        grid_shape,
    )

    targets = [model.blocks[idx] for idx in target_indices]
    with LeGrad(
        model,
        targets,
        score_projection=model.project,
        prefix_tokens=prefix_tokens,
        grid_shape=None if infer_grid else grid_shape,
    ) as extractor:
        scores = model(input_tensor)
        cam = extractor(class_idx, torch.full_like(scores, 1e6), retain_graph=True)[0]

        assert torch.allclose(scores, expected_scores, atol=1e-6, rtol=1e-5)
        assert torch.allclose(cam, expected_cam, atol=1e-6, rtol=1e-5)
        for idx, (attention, expected) in enumerate(zip(extractor.hook_attn, expected_attention, strict=True)):
            assert isinstance(attention, torch.Tensor)
            assert torch.allclose(attention, expected, atol=1e-6, rtol=1e-5)
            assert torch.allclose(attention.sum(dim=-1), torch.ones_like(attention[..., 0]))
            assert torch.autograd.grad(extractor.hook_a[idx].sum(), attention, retain_graph=True)[0] is not None


def test_legrad_torchvision_vit_repeated_calls_and_frozen_parameters():
    model = _build_vit().double()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    flags = [parameter.requires_grad for parameter in model.parameters()]
    state = {name: value.clone() for name, value in model.state_dict().items()}
    input_tensor = torch.randn(2, 3, 32, 32, dtype=torch.float64)

    with LeGrad(model, list(model.encoder.layers)) as extractor:
        for _ in range(2):
            scores = model(input_tensor)
            cam = extractor([0, 1], scores)[0]
            assert cam.shape == (2, 4, 4)
            assert cam.dtype == input_tensor.dtype
            assert cam.device == input_tensor.device
            assert torch.isfinite(cam).all()

    assert flags == [parameter.requires_grad for parameter in model.parameters()]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())


def test_legrad_classification_metric_compatibility():
    model = _build_vit()
    input_tensor = torch.randn(1, 3, 32, 32)

    with LeGrad(model, model.encoder.layers[-1]) as extractor:
        metric = ClassificationMetric(extractor, partial(torch.softmax, dim=-1))
        metric.update(input_tensor)
        assert set(metric.summary()) == {"avg_drop", "conf_increase"}


@pytest.mark.parametrize("raise_inside", [False, True])
def test_legrad_restores_model_and_hooks(raise_inside):
    model = _TinyViT().train()
    for idx, parameter in enumerate(model.parameters()):
        parameter.requires_grad_(idx % 2 == 0)
    attention = model.blocks[0].self_attention
    forward = attention.forward.__func__
    training = [module.training for module in model.modules()]
    flags = [parameter.requires_grad for parameter in model.parameters()]
    pre_hooks = len(attention._forward_pre_hooks)
    forward_hooks = len(attention._forward_hooks)

    expectation = pytest.raises(RuntimeError, match="boom") if raise_inside else nullcontext()
    with expectation, LeGrad(model, model.blocks[0], score_projection=model.project) as extractor:
        assert attention.forward.__func__ is forward
        if raise_inside:
            raise RuntimeError("boom")
        scores = model(torch.randn(1, 3, 2, 2))
        assert torch.isfinite(extractor(0, scores)[0]).all()

    assert attention.forward.__func__ is forward
    assert len(attention._forward_pre_hooks) == pre_hooks
    assert len(attention._forward_hooks) == forward_hooks
    assert training == [module.training for module in model.modules()]
    assert flags == [parameter.requires_grad for parameter in model.parameters()]


@pytest.mark.parametrize(
    ("target_factory", "exception", "error"),
    [
        (nn.Identity, TypeError, "self_attention"),
        (partial(_TinyBlock, batch_first=False), ValueError, "batch-first"),
        (_separate_projection_block, ValueError, "shared-dimension"),
        (partial(_TinyBlock, add_bias_kv=True), ValueError, "added attention tokens"),
        (partial(_TinyBlock, add_zero_attn=True), ValueError, "added attention tokens"),
    ],
)
def test_legrad_rejects_unsupported_target_blocks(target_factory, exception, error):
    model = _TinyViT()
    model.blocks[0] = target_factory()
    with pytest.raises(exception, match=error):
        LeGrad(model, model.blocks[0], score_projection=model.project)


def test_legrad_rejects_invalid_configuration():
    model = _TinyViT(grid_shape=(2, 3))

    with pytest.raises(ValueError, match="score_projection"):
        LeGrad(model, model.blocks[0])
    with pytest.raises(ValueError, match="explicit target"):
        LeGrad(model, [], score_projection=model.project)
    with pytest.raises(TypeError, match="integer"):
        LeGrad(model, model.blocks[0], score_projection=model.project, prefix_tokens=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        LeGrad(model, model.blocks[0], score_projection=model.project, prefix_tokens=-1)
    with pytest.raises(TypeError, match="tuple of two integers"):
        LeGrad(model, model.blocks[0], score_projection=model.project, grid_shape=[2, 3])
    with pytest.raises(ValueError, match="positive"):
        LeGrad(model, model.blocks[0], score_projection=model.project, grid_shape=(2, 0))
    with pytest.raises(TypeError, match="callable"):
        LeGrad(model, model.blocks[0], score_projection=object())

    with LeGrad(model, model.blocks[0], score_projection=model.project) as extractor:
        scores = model(torch.randn(1, 3, 2, 3))
        with pytest.raises(ValueError, match="square patch grid"):
            extractor(0, scores)

    with LeGrad(model, model.blocks[0], score_projection=model.project, grid_shape=(2, 2)) as extractor:
        scores = model(torch.randn(1, 3, 2, 3))
        with pytest.raises(ValueError, match="does not match"):
            extractor(0, scores)

    with LeGrad(model, model.blocks[0], score_projection=model.project, prefix_tokens=7) as extractor:
        scores = model(torch.randn(1, 3, 2, 3))
        with pytest.raises(ValueError, match="at least one patch"):
            extractor(0, scores)


def test_legrad_supports_positional_attention_options():
    model = _TinyViT()
    model.blocks[0] = _PositionalAttentionBlock()
    input_tensor = torch.randn(1, 3, 2, 2)

    with LeGrad(model, model.blocks[0], score_projection=model.project) as extractor:
        scores = model(input_tensor)
        assert torch.isfinite(extractor(0, scores)[0]).all()


def test_legrad_rejects_invalid_attention_inputs():
    model = _TinyViT()
    attention = model.blocks[0].self_attention
    query = torch.randn(1, 5, 8)

    with LeGrad(model, model.blocks[0], score_projection=model.project):
        with pytest.raises(ValueError, match="tensor query"):
            attention(query, None, query)
        unbatched = query.squeeze(0)
        with pytest.raises(ValueError, match="batch, tokens, embedding"):
            attention(unbatched, unbatched, unbatched)


def test_legrad_rejects_disconnected_scores_and_invalid_forward_modes():
    model = _TinyViT(attention_dropout=0.1).train()
    input_tensor = torch.randn(1, 3, 2, 2)
    with LeGrad(model, model.blocks[0], score_projection=model.project), pytest.raises(ValueError, match="dropout"):
        model(input_tensor)

    model = _TinyViT()
    with (
        LeGrad(model, model.blocks[0], score_projection=model.project),
        pytest.raises(RuntimeError, match="gradient tracking"),
        torch.inference_mode(),
    ):
        model(input_tensor)

    def disconnected(tokens):
        return model.project(tokens.detach())

    with LeGrad(model, model.blocks[0], score_projection=disconnected) as extractor:
        scores = model(input_tensor)
        with pytest.raises(RuntimeError, match="not connected"):
            extractor(0, scores)

    with (
        LeGrad(model, model.blocks[0], score_projection=lambda tokens: tokens.mean()),
        pytest.raises(ValueError, match="class logits"),
    ):
        model(input_tensor)


def test_legrad_rejects_cross_attention():
    input_tensor = torch.randn(1, 3, 2, 2)
    model = _TinyViT()
    model.blocks[0] = _CrossAttentionBlock()
    with (
        LeGrad(model, model.blocks[0], score_projection=model.project),
        pytest.raises(ValueError, match="self-attention"),
    ):
        model(input_tensor)


@pytest.mark.parametrize(
    ("attention_type", "error"),
    [
        (_NoWeightsAttention, "per-head attention probabilities"),
        (_AveragedWeightsAttention, "probabilities shaped"),
        (_ShortWeightsAttention, "incompatible attention probability shape"),
    ],
)
def test_legrad_rejects_incompatible_attention_outputs(attention_type, error):
    model = _TinyViT()
    model.blocks[0].self_attention = attention_type(8, 2, batch_first=True)
    with (
        LeGrad(model, model.blocks[0], score_projection=model.project),
        pytest.raises(ValueError, match=error),
    ):
        model(torch.randn(1, 3, 2, 2))


@pytest.mark.parametrize(
    ("block_type", "error", "during_forward"),
    [
        (_InvalidOutputBlock, "target blocks must return tokens", True),
        (_DropTokenBlock, "attention keys and target-block tokens", False),
    ],
)
def test_legrad_rejects_incompatible_block_outputs(block_type, error, during_forward):
    model = _TinyViT()
    model.blocks[0] = block_type()
    input_tensor = torch.randn(1, 3, 2, 2)

    with LeGrad(model, model.blocks[0], score_projection=model.project) as extractor:
        if during_forward:
            with pytest.raises(ValueError, match=error):
                model(input_tensor)
        else:
            scores = model(input_tensor)
            with pytest.raises(ValueError, match=error):
                extractor(0, scores)


def test_legrad_rejects_missing_captured_state():
    model = _TinyViT()
    input_tensor = torch.randn(1, 3, 2, 2)

    with LeGrad(model, model.blocks[0], score_projection=model.project) as extractor:
        scores = model(input_tensor)
        extractor.hook_attn[0] = None
        with pytest.raises(AssertionError, match="every LeGrad target block"):
            extractor(0, scores)

    with LeGrad(model, model.blocks[0], score_projection=model.project) as extractor:
        scores = model(input_tensor)
        extractor.hook_a[0] = None
        with pytest.raises(AssertionError, match="every LeGrad target block"):
            extractor.compute_cams(0, scores)


def test_legrad_rejects_missing_attention_return_options():
    model = _TinyViT()
    block = model.blocks[0]
    query = torch.randn(1, 5, 8)

    with LeGrad(model, block, score_projection=model.project) as extractor:
        model(torch.randn(1, 3, 2, 2))
        attention = extractor.hook_attn[0]
        assert isinstance(attention, torch.Tensor)
        extractor._weight_options[0] = None
        with pytest.raises(RuntimeError, match="missing original attention return options"):
            extractor._capture_attention(
                block.self_attention,
                (query, query, query),
                {},
                (query, attention),
            )


def test_legrad_remove_hooks_restores_attention():
    model = _TinyViT()
    attention = model.blocks[0].self_attention
    pre_hooks = len(attention._forward_pre_hooks)
    forward_hooks = len(attention._forward_hooks)
    extractor = LeGrad(model, model.blocks[0], score_projection=model.project)

    extractor.remove_hooks()

    assert len(attention._forward_pre_hooks) == pre_hooks
    assert len(attention._forward_hooks) == forward_hooks
    assert extractor.hook_handles == []
