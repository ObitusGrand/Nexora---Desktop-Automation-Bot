from PIL import Image
from io import BytesIO

from desktop_bot.capture import image_from_bytes
from desktop_bot.grid import add_coordinate_grid
from desktop_bot.models import ActionCommand, ScreenFrame, Target
from desktop_bot.vlm import VLMError, _parse_command, _parse_verification


def jpeg(width=320, height=200):
    output = BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="JPEG")
    return output.getvalue()


def test_grid_preserves_dimensions_and_adds_pixels():
    result = add_coordinate_grid(jpeg())
    image = image_from_bytes(result)
    assert image.size == (320, 200)
    assert result != jpeg()


def test_target_maps_vlm_coordinates_back_to_screen():
    command = ActionCommand(
        thought="button is visible",
        action="click",
        target={"x": 400, "y": 200},
        expected_outcome="dialog opens",
    )
    frame = ScreenFrame(
        image_bytes=b"frame",
        width=400,
        height=200,
        scale_x=0.5,
        scale_y=0.5,
        monitor=1,
    )
    assert frame.to_screen_coordinates(command.target) == Target(x=800, y=400)
    assert command.action.value == "click"


def test_invalid_vlm_json_is_rejected():
    try:
        _parse_command("not json")
    except VLMError as exc:
        assert "valid JSON" in str(exc)
    else:
        raise AssertionError("invalid JSON was accepted")


def test_pipe_separated_action_choices_are_rejected_with_actionable_error():
    malformed = '{"thought":"choose", "action":"click|double_click|type|hotkey|scroll|wait|done", "target":{"x":0,"y":0}, "expected_outcome":"done"}'

    try:
        _parse_command(malformed)
    except VLMError as exc:
        assert "action choices as one string" in str(exc)
    else:
        raise AssertionError("malformed action choices were accepted")


def test_type_text_is_recovered_when_model_puts_single_characters_in_keys():
    malformed = '{"thought":"type it", "action":"type", "target":{"x":0,"y":0}, "text":"", "keys":["P","E","E","P","S"," ","B","O","B","O"], "expected_outcome":"text appears"}'

    command = _parse_command(malformed)

    assert command.text == "PEEPS BOBO"
    assert command.keys == []


def test_verification_requires_boolean_verified_field():
    try:
        _parse_verification('{"reason":"the screen changed"}')
    except VLMError as exc:
        assert "omitted the required boolean 'verified'" in str(exc)
    else:
        raise AssertionError("verification without verified was accepted")
