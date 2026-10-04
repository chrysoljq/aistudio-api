from pathlib import Path

from aistudio_api.infrastructure.gateway.wire_codec import AistudioWireCodec
from aistudio_api.infrastructure.gateway.wire_types import (
    AistudioGenerationConfig,
    AistudioImageOutputMode,
    AistudioThinkingConfig,
    MediaResolution,
    ThinkingLevel,
    budget_sized_thinking_level,
)


ROOT = Path(__file__).resolve().parents[1]


def test_decode_image_request_maps_generation_config_fields_from_proto_indexes():
    codec = AistudioWireCodec()
    raw = (ROOT / "test-image-input.json").read_text()

    request = codec.decode(raw)

    assert request.model == "models/gemini-3.1-flash-image-preview"
    assert request.generation_config.stop_sequences == ["6"]
    assert request.generation_config.max_tokens == 65536
    assert request.generation_config.temperature == 1
    assert request.generation_config.top_p == 0.95
    assert request.generation_config.top_k == 64
    assert request.generation_config.image_output_mode == [2, 1]
    assert request.generation_config.thinking_config == [1, None, None, 3]
    assert request.request_flag == 1
    assert request.cached_content.startswith("v1_")


def test_encode_preserves_newly_mapped_proto_fields():
    codec = AistudioWireCodec()
    raw = (ROOT / "test-image-input.json").read_text()

    request = codec.decode(raw)
    encoded = codec.encode(request)
    reparsed = codec.decode(encoded)

    assert reparsed.generation_config.stop_sequences == ["6"]
    assert reparsed.generation_config.image_output_mode == [2, 1]
    assert reparsed.generation_config.thinking_config == [1, None, None, 3]
    assert reparsed.request_flag == 1
    assert reparsed.cached_content == request.cached_content


def test_thinking_config_encodes_high_level_wire_shape():
    assert AistudioThinkingConfig(ThinkingLevel.HIGH).to_wire() == [1, None, None, 3]


def test_generation_config_enables_default_thinking():
    config = AistudioGenerationConfig([])

    config.enable_default_thinking()

    assert config.thinking_config == [1, None, None, 3]
    assert config.media_resolution is None


def test_generation_config_disables_thinking_when_output_budget_is_small():
    # HIGH thinking consumed 93/96 tokens at max_tokens=100, leaving 2 tokens of
    # visible content ("长城是"). Below the threshold the level drops to MINIMAL,
    # measured 0 reasoning tokens. Omitting the config is not an option: these
    # models reason even with no thinking config (95-96 reasoning tokens).
    config = AistudioGenerationConfig([])

    config.enable_default_thinking(100, model="models/gemini-3.5-flash")

    assert config.thinking_config == [1, None, None, 4]
    assert config.media_resolution is None


def test_generation_config_uses_low_when_model_rejects_minimal():
    # gemini-flash-latest answers HTTP 400 for MINIMAL; LOW is its cheapest
    # accepted level and also measured 0 reasoning tokens.
    config = AistudioGenerationConfig([])

    config.enable_default_thinking(100, model="models/gemini-flash-latest")

    assert config.thinking_config == [1, None, None, 1]


def test_budget_sized_thinking_level_defaults_to_minimal():
    assert budget_sized_thinking_level("models/gemini-3.5-flash") == ThinkingLevel.MINIMAL
    assert budget_sized_thinking_level(None) == ThinkingLevel.MINIMAL


def test_budget_sized_thinking_level_falls_back_for_unsupported_model():
    assert budget_sized_thinking_level("models/gemini-flash-latest") == ThinkingLevel.LOW


def test_generation_config_keeps_high_thinking_without_token_cap():
    config = AistudioGenerationConfig([])

    config.enable_default_thinking(None)

    assert config.thinking_config == [1, None, None, 3]


def test_generation_config_keeps_high_thinking_when_budget_is_generous():
    config = AistudioGenerationConfig([])

    config.enable_default_thinking(2000)

    assert config.thinking_config == [1, None, None, 3]


def test_generation_config_never_overrides_caller_thinking_config():
    # An explicit caller/model-default choice always wins, capped or not.
    config = AistudioGenerationConfig([])
    config.thinking_config = AistudioThinkingConfig(ThinkingLevel.MEDIUM).to_wire()

    config.enable_default_thinking(100)

    assert config.thinking_config == [1, None, None, 2]


def test_generation_config_keeps_image_model_thinking_level_under_small_cap():
    # Image models get MINIMAL from their profile; force=True must not clear it.
    config = AistudioGenerationConfig([])
    config.thinking_config = AistudioThinkingConfig(ThinkingLevel.MINIMAL).to_wire()

    config.enable_default_thinking(64, force=True)

    assert config.thinking_config == [1, None, None, 4]


def test_generation_config_force_resizes_inherited_high_thinking():
    # The captured browser body carries HIGH; with a small cap it must be resized.
    config = AistudioGenerationConfig([])
    config.thinking_config = AistudioThinkingConfig(ThinkingLevel.HIGH).to_wire()

    config.enable_default_thinking(100, force=True, model="models/gemini-3.5-flash")

    assert config.thinking_config == [1, None, None, 4]


def test_generation_config_accepts_readable_image_output_mode_wrapper():
    config = AistudioGenerationConfig([])

    config.image_output_mode = AistudioImageOutputMode.text_and_image()

    assert config.image_output_mode == [2, 1]


def test_generation_config_accepts_media_resolution_enum():
    config = AistudioGenerationConfig([])

    config.media_resolution = MediaResolution.MEDIUM

    assert config.media_resolution == 2
