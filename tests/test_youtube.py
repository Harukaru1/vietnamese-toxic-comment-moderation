import pytest

from toxic_moderation.youtube import extract_video_id


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=hGu7BVQy5G0",
    "https://www.youtube.com/watch?v=hGu7BVQy5G0&t=42s",
    "https://youtu.be/hGu7BVQy5G0?si=abc",
    "https://www.youtube.com/shorts/hGu7BVQy5G0",
    "https://www.youtube.com/live/hGu7BVQy5G0",
    "hGu7BVQy5G0",
])
def test_extract_video_id(url):
    assert extract_video_id(url) == "hGu7BVQy5G0"


def test_extract_video_id_rejects_garbage():
    with pytest.raises(ValueError):
        extract_video_id("https://example.com/not-youtube")
