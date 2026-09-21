"""Render the shot list to a captioned MP4.

Each shot becomes one frame: a screenshot or a generated card, with a caption
bar carrying the text. ffmpeg holds each frame for its stated duration. There is
no narration - the captions are the narration - so they are sized to be read at
a glance and held long enough to finish reading.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import textwrap
from pathlib import Path

from neuromps.paths import REPO_ROOT, metrics_path, outputs_dir
from neuromps.video.shots import Shot, shot_list

log = logging.getLogger("neuromps.video")

WIDTH, HEIGHT = 1920, 1080
MARGIN = 64
CAPTION_HEIGHT = 190
BACKGROUND = (250, 250, 249)
CARD_BACKGROUND = (24, 30, 38)
INK = (22, 26, 32)
ACCENT = (0, 114, 178)
FPS = 30
FONT_DIRS = ("/usr/share/fonts", str(Path.home() / ".fonts"))


def _font(size: int, bold: bool = False):
    """A DejaVu face at the requested size, falling back to the default."""
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for root in FONT_DIRS:
        for path in Path(root).rglob(name):
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def _wrapped(draw, text: str, font, width: int) -> list[str]:
    """Break text to fit a pixel width."""
    if not text:
        return []
    average = max(draw.textlength("x" * 40, font=font) / 40, 1)
    return textwrap.wrap(text, width=max(int(width / average), 20))


def render_card(shot: Shot):
    """A text slide: title, body lines, section label."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), CARD_BACKGROUND)
    draw = ImageDraw.Draw(image)
    if shot.title:
        draw.text((MARGIN + 40, 150), shot.title, font=_font(62, bold=True), fill=(255, 255, 255))
    y = 270
    for line in shot.body:
        draw.text((MARGIN + 40, y), line, font=_font(34), fill=(226, 232, 240))
        y += 54
    _section_label(draw, shot.section, light=True)
    return image


def render_still(shot: Shot):
    """A screenshot or figure, scaled into the frame above its caption."""
    from PIL import Image, ImageDraw

    source = Image.open(REPO_ROOT / shot.image).convert("RGB")
    if shot.crop:
        source = source.crop(shot.crop)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    room = (WIDTH - 2 * MARGIN, HEIGHT - CAPTION_HEIGHT - MARGIN)
    scale = min(room[0] / source.width, room[1] / source.height)
    resized = source.resize((int(source.width * scale), int(source.height * scale)))
    top = (HEIGHT - CAPTION_HEIGHT - resized.height) // 2
    canvas.paste(resized, ((WIDTH - resized.width) // 2, max(top, MARGIN // 2)))
    draw = ImageDraw.Draw(canvas)
    _caption(draw, shot.caption)
    _section_label(draw, shot.section, light=False)
    return canvas


def _caption(draw, text: str) -> None:
    """The caption bar at the foot of a still."""
    draw.rectangle([0, HEIGHT - CAPTION_HEIGHT, WIDTH, HEIGHT], fill=(236, 239, 241))
    draw.rectangle([0, HEIGHT - CAPTION_HEIGHT, WIDTH, HEIGHT - CAPTION_HEIGHT + 5], fill=ACCENT)
    font = _font(34)
    y = HEIGHT - CAPTION_HEIGHT + 30
    for line in _wrapped(draw, text, font, WIDTH - 2 * MARGIN)[:3]:
        draw.text((MARGIN, y), line, font=font, fill=INK)
        y += 48


def _section_label(draw, section: str, light: bool) -> None:
    """A small section marker, so a viewer always knows where they are."""
    if not section:
        return
    colour = (148, 163, 184) if light else (100, 116, 139)
    draw.text((MARGIN, 40), section.upper(), font=_font(26, bold=True), fill=colour)


def render_frames(shots: list[Shot], target: Path) -> list[tuple[Path, float]]:
    """Write one PNG per shot; return the frames with their durations."""
    target.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, shot in enumerate(shots):
        image = render_still(shot) if shot.image else render_card(shot)
        path = target / f"frame_{index:03d}.png"
        image.save(path)
        frames.append((path, shot.seconds))
    return frames


def encode(frames: list[tuple[Path, float]], output: Path) -> None:
    """Hold each frame for its duration and encode to H.264."""
    listing = output.parent / "frames.txt"
    lines = []
    for path, seconds in frames:
        lines.append(f"file '{path.as_posix()}'")
        lines.append(f"duration {seconds:.2f}")
    lines.append(f"file '{frames[-1][0].as_posix()}'")  # concat needs the last frame twice
    listing.write_text("\n".join(lines) + "\n")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-vf",
            f"fps={FPS},format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
        capture_output=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Build outputs/video/neuromps_demo.mp4 from the shot list."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="accepted for run_all; no effect")
    p.add_argument("--keep-frames", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if shutil.which("ffmpeg") is None:
        log.warning("ffmpeg not installed; skipping the video build")
        return 0

    shots = shot_list(json.loads(metrics_path().read_text()))
    video_dir = outputs_dir() / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    frames = render_frames(shots, video_dir / "frames")
    output = video_dir / "neuromps_demo.mp4"
    encode(frames, output)
    if not args.keep_frames:
        shutil.rmtree(video_dir / "frames", ignore_errors=True)
        (video_dir / "frames.txt").unlink(missing_ok=True)
    total = sum(shot.seconds for shot in shots)
    log.info("wrote %s: %d shots, %.0f s", output, len(shots), total)
    print(json.dumps({"shots": len(shots), "seconds": total, "path": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
