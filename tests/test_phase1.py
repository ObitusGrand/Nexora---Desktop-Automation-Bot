from PIL import Image
from io import BytesIO

from desktop_bot.capture import image_from_bytes
from desktop_bot.grid import add_coordinate_grid
from desktop_bot.models import ActionCommand, ScreenFrame, Target
from desktop_bot.vlm import VLMError, _parse_command


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
