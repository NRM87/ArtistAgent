from artist_agent.backends import (
    MockLLMBackend,
    _is_prompt_aligned_with_vision,
    _is_usable_image_prompt,
    _meaningful_feedback,
    _meaningful_next_action,
    _normalize_action_command,
    _normalize_action_vision,
)
from artist_agent.revision import (
    DEFAULT_CRITIQUE_DIRECTIVE,
    DEFAULT_REVISION_DIRECTIVE,
    DEFAULT_VISION_DIRECTIVE,
    normalize_run_intent,
)


def test_normalize_run_intent_defaults_when_missing():
    out = normalize_run_intent({})
    assert out["vision_directive"] == DEFAULT_VISION_DIRECTIVE
    assert out["critique_directive"] == DEFAULT_CRITIQUE_DIRECTIVE
    assert out["revision_directive"] == DEFAULT_REVISION_DIRECTIVE


def test_normalize_run_intent_converts_first_person_to_commands():
    out = normalize_run_intent(
        {
            "vision_directive": "I will create a moonlit bridge with hard shadows",
            "critique_directive": "My goal is to score fidelity first, then aesthetics",
            "revision_directive": "I intend to record one concrete lesson from this run",
        }
    )
    assert out["vision_directive"].lower().startswith("create a moonlit bridge")
    assert out["critique_directive"].lower().startswith("score fidelity")
    assert out["revision_directive"].lower().startswith("record one concrete lesson")


def test_normalize_run_intent_rejects_meta_instruction_echo():
    out = normalize_run_intent(
        {
            "vision_directive": "Artwork tiers guide decisions across history.",
            "critique_directive": "Respond using exactly three lines.",
            "revision_directive": "soul_packet:{...}",
        }
    )
    assert out["vision_directive"] == DEFAULT_VISION_DIRECTIVE
    assert out["critique_directive"] == DEFAULT_CRITIQUE_DIRECTIVE
    assert out["revision_directive"] == DEFAULT_REVISION_DIRECTIVE


def test_normalize_action_vision_rewrites_generic_statement():
    vision = _normalize_action_vision(
        "Artwork tiers guide decisions: masterpieces reinforce strengths.",
        soul={"current_obsession": "Perfect squares and right angles"},
    )
    assert vision.lower().startswith("my vision for this run is to create")
    assert "perfect squares" in vision.lower()


def test_normalize_action_vision_keeps_labeled_command_output():
    vision = _normalize_action_vision(
        "RUN_VISION: My vision for this run is to create a crimson tower above a flooded plaza."
    )
    assert vision.startswith("My vision for this run is to")
    assert "crimson tower" in vision.lower()


def test_normalize_action_command_converts_to_imperative():
    action = _normalize_action_command("NEXT_ACTION: I will increase edge contrast around the focal square")
    assert action.lower().startswith("increase edge contrast")


def test_normalize_run_intent_rewrites_artist_name_possessive():
    out = normalize_run_intent(
        {
            "vision_directive": "Create something reflecting Lyra's ideals through geometry",
            "critique_directive": "Score by Lyra's values before aesthetics",
            "revision_directive": "Record learning that matches Lyra's priorities",
        },
        self_name="Lyra",
    )
    assert "lyra's" not in out["vision_directive"].lower()
    assert "my ideals" in out["vision_directive"].lower()
    assert "my values" in out["critique_directive"].lower()
    assert "my priorities" in out["revision_directive"].lower()


def test_normalize_action_vision_rewrites_artist_name_possessive():
    vision = _normalize_action_vision(
        "RUN_VISION: My vision for this run is to create an image reflecting Lyra's ideals with rigid perspective.",
        soul={"name": "Lyra", "current_obsession": "Rigid perspective"},
    )
    assert "lyra's" not in vision.lower()
    assert "my ideals" in vision.lower()


def test_normalize_action_command_rewrites_artist_name_possessive():
    action = _normalize_action_command(
        "NEXT_ACTION: Increase contrast to match Lyra's ideals in the focal area.",
        self_name="Lyra",
    )
    assert "lyra's" not in action.lower()
    assert "my ideals" in action.lower()


def test_mock_backend_can_generate_initial_render_prompt():
    backend = MockLLMBackend()
    prompt = backend.generate_initial_render_prompt(
        {"name": "Lyra"},
        "My vision for this run is to create a foggy bridge at dawn.",
        run_intent={"vision_directive": "Emphasize angular silhouettes and cool tones."},
    )
    assert "foggy bridge" in prompt.lower()
    assert "angular silhouettes" in prompt.lower()


def test_normalize_action_vision_strips_run_vision_placeholder():
    vision = _normalize_action_vision(
        "RUN_VISION: My vision for this run is to create an image that RUN_VISION.",
        soul={"name": "Lyra", "current_obsession": "geometric precision"},
    )
    assert "run_vision" not in vision.lower()
    assert "my vision for this run is to" in vision.lower()


def test_image_prompt_quality_checks_reject_wrappers_and_misalignment():
    bad = (
        'Run vision (fixed for this run): "My vision..."\n'
        'Iteration image prompt: "A meticulously aligned geometric masterpiece"\n'
        "Create a coherent 2D composition using the iteration image prompt while staying faithful to the fixed run vision."
    )
    assert not _is_usable_image_prompt(bad)
    assert not _is_prompt_aligned_with_vision("A geometric masterpiece", "create a blue fox under moonlight")


def test_critique_quality_checks_reject_trivial_outputs():
    assert not _meaningful_feedback("My vision for")
    assert not _meaningful_next_action("Create.")
    assert _meaningful_feedback("I lost depth in the foreground, so I should deepen shadow contrast around the focal arch.")
    assert _meaningful_next_action("Increase rim light on the left edge of the fox and darken the rear background.")
