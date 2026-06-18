from aistudio_api.infrastructure.gateway.model_defaults import resolve_model_defaults
from aistudio_api.infrastructure.gateway.wire_types import AistudioImageOutputMode


def test_resolve_model_defaults_uses_repo_config_for_gemma():
    defaults = resolve_model_defaults("models/gemma-4-31b-it")

    assert defaults.default_tools == ("google_search",)
    assert defaults.is_image_model is False
    assert defaults.safety_settings == (
        (None, None, 7, 5),
        (None, None, 8, 5),
        (None, None, 9, 5),
        (None, None, 10, 5),
    )


def test_resolve_model_defaults_merges_yaml_profiles(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_defaults:
  profiles:
    - name: gemini_models
      match:
        prefixes: [gemini-]
      generation_config_defaults:
        response_mime_type: application/json
    - name: image_models
      match:
        contains: [image]
      is_image_model: true
      generation_config_defaults:
        image_output_mode: text_and_image
        thinking_config:
          level: HIGH
          mode: 9
        media_resolution: HIGH
      clear_generation_config_indexes: [7, 17]
      disable_safety_settings: true
  models:
    gemini-3.1-flash-image-preview:
      generation_config_defaults:
        response_mime_type: null
"""
    )

    defaults = resolve_model_defaults("models/gemini-3.1-flash-image-preview", config_path=config_path)

    assert defaults.is_image_model is True
    assert defaults.disable_safety_settings is True
    assert defaults.clear_generation_config_indexes == (7, 17)
    assert defaults.generation_config_defaults["response_mime_type"] is None
    assert defaults.generation_config_defaults["image_output_mode"] == AistudioImageOutputMode.text_and_image()
    assert defaults.generation_config_defaults["thinking_config"] == [9, None, None, 3]
    assert defaults.generation_config_defaults["media_resolution"] == 3


def test_resolve_model_defaults_coerces_safety_settings(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_defaults:
  profiles:
    - name: gemini_models
      match:
        prefixes: [gemini-]
      safety_settings:
        Harassment: 1
        Hate: 2
        Sexually Explicit: 3
        Dangerous Content: 5
"""
    )

    defaults = resolve_model_defaults("models/gemini-2.5-flash", config_path=config_path)

    assert defaults.safety_settings == (
        (None, None, 7, 1),
        (None, None, 8, 2),
        (None, None, 9, 3),
        (None, None, 10, 5),
    )


def test_resolve_model_defaults_supports_image_default_tool_names(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_defaults:
  profiles:
    - name: image_models
      match:
        contains: [image]
      is_image_model: true
      default_tools:
        - google_search_and_image_search
"""
    )

    defaults = resolve_model_defaults("models/gemini-3.1-flash-image-preview", config_path=config_path)

    assert defaults.is_image_model is True
    assert defaults.default_tools == ("google_search_and_image_search",)


def test_resolve_model_defaults_keeps_gemini_four_tool_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_defaults:
  profiles:
    - name: gemini_models
      match:
        prefixes: [gemini-]
      default_tools:
        - google_search
        - code_execution
        - google_maps
        - url_context
"""
    )

    defaults = resolve_model_defaults("models/gemini-3.5-flash", config_path=config_path)

    assert defaults.default_tools == ("google_search", "code_execution", "google_maps", "url_context")


def test_resolve_model_defaults_model_override_replaces_profile_tools_and_flags(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_defaults:
  profiles:
    - name: gemini_models
      match:
        prefixes: [gemini-]
      default_tools:
        - google_search
      disable_safety_settings: true
  models:
    gemini-3.5-flash:
      default_tools:
        - code_execution
      disable_safety_settings: false
"""
    )

    defaults = resolve_model_defaults("models/gemini-3.5-flash", config_path=config_path)

    assert defaults.default_tools == ("code_execution",)
    assert defaults.disable_safety_settings is False
