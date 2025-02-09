import numpy as np
import os
import pytest
from sleap.io.dataset import Labels
from sleap.io.visuals import (
    save_labeled_video,
    resize_images,
    VideoMarkerThread,
    main as sleap_render,
)


def test_resize(small_robot_mp4_vid):
    imgs = small_robot_mp4_vid[:4]

    resized_imgs = resize_images(imgs, 0.25)

    assert resized_imgs.shape[0] == imgs.shape[0]
    assert resized_imgs.shape[1] == imgs.shape[1] // 4
    assert resized_imgs.shape[2] == imgs.shape[2] // 4
    assert resized_imgs.shape[3] == imgs.shape[3]


def test_serial_pipeline(centered_pair_predictions, tmpdir):
    frames = [0, 1, 2]
    video_idx = 0
    scale = 0.25

    video = centered_pair_predictions.videos[video_idx]
    frame_images = video.get_frames(frames)

    # Make sure we can resize
    small_images = resize_images(frame_images, scale=scale)

    _, height, width, _ = small_images.shape

    assert height == video.height // (1 / scale)
    assert width == video.width // (1 / scale)

    marker_thread = VideoMarkerThread(
        in_q=None,
        out_q=None,
        labels=centered_pair_predictions,
        video_idx=video_idx,
        scale=scale,
        color_manager=None,
    )

    # Make sure we can mark images
    marked_image_list = marker_thread._mark_images(
        frame_indices=frames,
        frame_images=small_images,
    )

    # There's a point at 201, 186 (i.e. 50.25, 46.5), so make sure it got marked
    assert not np.allclose(
        marked_image_list[0][44:48, 48:52, 0], small_images[0, 44:48, 48:52, 0]
    )

    # Make sure no change where nothing marked
    assert np.allclose(
        marked_image_list[0][10:20, :10, 0], small_images[0, 10:20, :10, 0]
    )


def test_sleap_render(centered_pair_predictions):
    args = f"-o testvis.avi -f 2 --scale 1.2 --frames 1,2 --video-index 0 tests/data/json_format_v2/centered_pair_predictions.json".split()
    sleap_render(args)
    assert os.path.exists("testvis.avi")


@pytest.mark.parametrize("crop", ["Half", "Quarter", None])
def test_write_visuals(tmpdir, centered_pair_predictions: Labels, crop: str):
    labels = centered_pair_predictions
    video = centered_pair_predictions.videos[0]

    # Determine crop size relative to original size and scale
    crop_size_xy = None
    w = int(video.backend.width)
    h = int(video.backend.height)
    if crop == "Half":
        crop_size_xy = (w // 2, h // 2)
    elif crop == "Quarter":
        crop_size_xy = (w // 4, h // 4)

    path = os.path.join(tmpdir, "clip.avi")
    save_labeled_video(
        filename=path,
        labels=centered_pair_predictions,
        video=video,
        frames=(0, 1, 2),
        fps=15,
        crop_size_xy=crop_size_xy,
    )
    assert os.path.exists(path)
