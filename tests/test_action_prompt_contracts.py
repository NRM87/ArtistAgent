from artist_agent.backends import _normalize_action_command, _normalize_action_vision
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
