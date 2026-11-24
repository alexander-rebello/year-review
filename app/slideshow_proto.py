#!/usr/bin/env python3
"""
Year-review slideshow - prototype
=================================

Turns a year folder into one movie:

    2026/
      01 Ostertrip - Wölflinge/     one folder per section, in name order
      02 Ostertrip - Pfadis/        "NN Title - Subtitle"; the number only sets the order
      03 Sommerlager/
      music/                        songs to choose from (the playlist is set on the review page)
      slideshow.yaml                manual fixes, written by the review page (review.py)

Every section opens with its title over the darkened hero photo (the first photo,
or the one set as hero), then the title fades away and the hero continues as the
first slide. Photos are shown in name order, with
  * face- and people-aware pan/zoom that ends on the people
  * portrait photos paired side by side
  * smooth sub-pixel motion and crossfades

Setup:   pip install -r requirements.txt     (plus ffmpeg on PATH, e.g. `brew install ffmpeg`)

Usually you work in the review page (python review.py 2026/) and render from there.
From the terminal:
  python slideshow_proto.py 2026/ --plan-only --debug debug/     # check framing, no render
  python slideshow_proto.py 2026/ -o test.mp4 --preview --only 03 # one section, low-res
  python slideshow_proto.py 2026/ -o jahresrueckblick.mp4         # the whole movie
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps

try:
    import pillow_heif  # iPhone HEIC support

    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None

try:
    import yaml
except ImportError:
    yaml = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff"}
MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
PEOPLE_URL = ("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
              "0.1.1rc0/yolox_s.onnx")
CACHE_NAME = ".slideshow_analysis.json"
PROJECT_NAME = "slideshow.yaml"          # manual fixes (old layout: in the year folder)
EDITS_NAME = "edits.yaml"                # manual fixes in a project folder
MUSIC_FOLDERS = {"music", "musik"}       # not a section
VIDEO_EXTS = {".mov", ".mp4", ".m4v"}
MUSIC_EXTS = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".aif", ".aiff", ".ogg"}
ANALYSIS_VERSION = 2  # bump when the detection step changes, so cached results are redone
MUSIC_VERSION = 3     # same for songs and video clips
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


# --------------------------------------------------------------------------- config

@dataclass
class Config:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    slide_dur: float = 4.5    # default time a slide is visible, including its fades (s)
    max_dur: float = 7.0      # long pans may stretch a slide up to this (s)
    fade: float = 1.0         # crossfade length (s)
    title_dur: float = 3.5    # title fully visible over the darkened hero photo (s)
    title_fade: float = 1.3   # title + darkening fade away, revealing the hero photo (s)
    title_dim: float = 0.5    # how dark the hero photo is behind the title (0..1)
    title_blur: float = 0.7   # how blurred the hero photo is behind the title (0..1)
    music_fade: float = 3.0   # fade when a song is cut short by the next one or the movie end (s)
    cut_xfade: float = 0.3    # crossfade over a cut inside a song (s)
    max_speed: float = 0.25   # motion limit in screen heights per second (zoom counts too)
    max_zoom: float = 1.35    # end frame at most this much tighter than the start frame
    max_upscale: float = 1.1  # never magnify source pixels more than this
    gap: int = 12             # px between paired portraits
    bg_gray: int = 12         # colour of the gap between paired portraits
    portrait_ar: float = 0.9  # width/height below this counts as portrait
    group_faces: int = 5      # this many people or more -> gentle push-in instead of a pan
    max_pan: float = 0.45     # longest pan, in screen heights (the old version crossed the whole photo)
    face_margin: float = 0.06 # faces stay at least this far (share of frame) inside the start frame
    face_y: float = 0.38      # where faces end up, from the top of the frame (0..1)
    pair_portraits: bool = True
    title_size: float = 0.1   # title height, share of the frame height
    subtitle_size: float = 0.045
    font: str = ""            # font file for titles, empty = built-in choice
    music_lufs: float = -16.0 # every song is brought to this loudness (EBU R128)
    clip_lufs: float = -18.0  # sound of video clips
    clip_music: float = 0.25  # music level while a clip with sound plays (0..1)
    end_fade: float = 2.5     # sound fade at the very end (s)
    bitrate: int = 16         # video quality, Mbit/s


# key, group, label, kind, min, max, step, help. kind: s | pct | num | bool | font | choice
SETTINGS = [
    ("slide_dur", "Timing", "Photo on screen", "s", 2, 12, 0.5,
     "How long a photo is shown, including its crossfades. Photos with a long pan may stay longer."),
    ("fade", "Timing", "Crossfade", "s", 0.2, 3, 0.1, "Length of the dissolve between photos."),
    ("max_dur", "Timing", "Longest time for one photo", "s", 3, 15, 0.5,
     "Upper limit for photos whose pan needs more time."),
    ("title_dur", "Section titles", "Title stays for", "s", 1, 10, 0.5,
     "How long the section title is fully visible over the title photo."),
    ("title_fade", "Section titles", "Title fades away over", "s", 0.3, 4, 0.1, ""),
    ("title_dim", "Section titles", "Darken the photo behind titles", "pct", 0, 0.9, 0.05, ""),
    ("title_blur", "Section titles", "Blur the photo behind titles", "pct", 0, 1, 0.05, ""),
    ("title_size", "Section titles", "Title size", "pct", 0.04, 0.2, 0.005, "Share of the picture height."),
    ("subtitle_size", "Section titles", "Subtitle size", "pct", 0.02, 0.1, 0.005, "Share of the picture height."),
    ("font", "Section titles", "Font", "font", None, None, None, "Used for all titles, intro and outro."),
    ("max_pan", "Motion", "Longest pan", "num", 0, 1.5, 0.05,
     "In screen heights. 0 means photos only zoom slowly, no panning."),
    ("max_speed", "Motion", "Speed limit", "num", 0.05, 0.6, 0.01,
     "Screen heights per second. Lower is calmer."),
    ("max_zoom", "Motion", "Strongest zoom", "num", 1.0, 2.0, 0.05,
     "How much tighter the end of a move may be than its start. 1 means no zoom."),
    ("face_y", "Motion", "Where faces end up", "pct", 0.2, 0.6, 0.01,
     "Height of the faces at the end of a move, from the top of the picture."),
    ("pair_portraits", "Layout", "Show portrait photos in pairs", "bool", None, None, None,
     "Two upright photos side by side instead of each alone."),
    ("gap", "Layout", "Gap between paired photos", "num", 0, 60, 2, "In pixels."),
    ("group_faces", "Layout", "Treat as group photo from", "num", 2, 30, 1,
     "Number of people from which a photo only zooms gently, so nobody is cut off."),
    ("music_lufs", "Sound", "Music loudness", "num", -30, -8, 1, "LUFS. All songs are matched to this."),
    ("clip_lufs", "Sound", "Loudness of video clip sound", "num", -30, -8, 1, "LUFS."),
    ("clip_music", "Sound", "Music while a clip plays", "pct", 0, 1, 0.05,
     "Default for clips with sound; each clip can override it."),
    ("music_fade", "Sound", "Fade when a song is cut short", "s", 0.5, 8, 0.5, ""),
    ("cut_xfade", "Sound", "Crossfade over a cut in a song", "s", 0.05, 2, 0.05, ""),
    ("end_fade", "Sound", "Fade at the end of the movie", "s", 0.5, 10, 0.5, ""),
    ("resolution", "Output", "Resolution", "choice", ["1280x720", "1920x1080", "2560x1440", "3840x2160"], None, None,
     "Higher takes longer to render."),
    ("fps", "Output", "Frames per second", "choice", [24, 25, 30, 50, 60], None, None, ""),
    ("bitrate", "Output", "Video quality", "num", 4, 60, 1, "Mbit/s. 16 is plenty for 1080p; use about 40 for 4K."),
]


def config_value(cfg: Config, key: str):
    return f"{cfg.width}x{cfg.height}" if key == "resolution" else getattr(cfg, key)


def make_config(project: dict, preview: bool = False) -> Config:
    """Defaults, changed by the settings saved on the review page."""
    cfg, saved = Config(), project.get("settings") or {}
    for key, _, _, kind, lo, hi, _, _ in SETTINGS:
        v = saved.get(key)
        if v is None:
            continue
        try:
            if kind == "choice":
                if key == "resolution" and v in lo:
                    cfg.width, cfg.height = map(int, v.split("x"))
                elif key == "fps" and int(v) in lo:
                    cfg.fps = int(v)
            elif kind == "bool":
                setattr(cfg, key, bool(v))
            elif kind == "font":
                cfg.font = str(v) if Path(str(v)).exists() else ""
            else:
                v = min(max(float(v), lo), hi)
                setattr(cfg, key, int(round(v)) if isinstance(getattr(cfg, key), int) else v)
        except (TypeError, ValueError):
            pass
    if preview:
        cfg.width, cfg.height, cfg.fps, cfg.gap = 960, 540, 25, cfg.gap // 2
    cfg.fade = max(1, round(cfg.fade * cfg.fps)) / cfg.fps  # whole frames, so cached parts line up
    return cfg


# --------------------------------------------------------------------------- geometry

@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def scaled(self, k: float) -> "Rect":
        return Rect(self.x * k, self.y * k, self.w * k, self.h * k)


def rect_around(cx, cy, w, h, W, H) -> Rect:
    """A w*h rectangle centred as close to (cx, cy) as possible, kept inside the W*H image."""
    w, h = min(w, W), min(h, H)
    return Rect(min(max(cx - w / 2, 0), W - w), min(max(cy - h / 2, 0), H - h), w, h)


def cover_size(W, H, aspect):
    """Largest rectangle of the given aspect ratio that fits inside a W*H image."""
    return (H * aspect, H) if W / H > aspect else (W, W / aspect)


def lerp_rect(a: Rect, b: Rect, u: float) -> Rect:
    """Size changes geometrically (zoom feels uniform), the centre moves linearly.
    If a and b lie inside the image, every in-between rect does too."""
    w = a.w * (b.w / a.w) ** u
    h = w * a.h / a.w
    cx = a.cx + (b.cx - a.cx) * u
    cy = a.cy + (b.cy - a.cy) * u
    return Rect(cx - w / 2, cy - h / 2, w, h)


def motion_amount(a: Rect, b: Rect) -> float:
    """How far the picture travels on screen, in screen heights. Zoom counts as motion."""
    shift = math.hypot(b.cx - a.cx, b.cy - a.cy) / ((a.h + b.h) / 2)
    return shift + 1.2 * abs(math.log(a.w / b.w))


def ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


# --------------------------------------------------------------------------- loading

def exif_info(im: Image.Image):
    exif = im.getexif()
    date = exif.get_ifd(0x8769).get(36867) or exif.get(306)  # DateTimeOriginal, DateTime
    date = str(date).replace("\x00", "").strip() if date else None
    return date, exif.get(274, 1)                             # 274 = orientation


def load_image(path: Path, max_side: int | None = None):
    """Load upright, in sRGB, as a BGR array.
    Returns (bgr, (W, H) of the full-size upright image, EXIF date or None)."""
    im = Image.open(path)
    date, orientation = exif_info(im)
    W, H = im.size
    if orientation in (5, 6, 7, 8):
        W, H = H, W
    icc = im.info.get("icc_profile")
    if max_side:
        im.draft("RGB", (max_side, max_side))  # JPEG: decode directly at reduced size (much faster)
    im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":
        im = im.convert("RGB")
    if icc:  # iPhone photos are Display P3; ignoring that makes colours look dull
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            im = ImageCms.profileToProfile(im, src, SRGB, outputMode="RGB")
        except Exception:
            pass
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), (W, H), date


# --------------------------------------------------------------------------- analysis

class FaceFinder:
    """YuNet face detector (open source, ships with OpenCV). Falls back to the much
    weaker Haar cascade if the model can't be downloaded."""

    def __init__(self, model: Path):
        self.model = model
        self.yunet = None
        if not model.exists():
            try:
                model.parent.mkdir(parents=True, exist_ok=True)
                print(f"Downloading face model -> {model}")
                urllib.request.urlretrieve(MODEL_URL, model)
            except Exception as e:
                print(f"  could not download ({e}).\n  Get it manually from {MODEL_URL}\n"
                      f"  Using the weaker Haar detector for now.")
                model.unlink(missing_ok=True)
        self.haar = None
        if model.exists():
            self.yunet = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.75, 0.3, 5000)
            return
        name = "haarcascade_frontalface_default.xml"
        bundled = getattr(getattr(cv2, "data", None), "haarcascades", None)
        for p in ([Path(bundled) / name] if bundled else []) + [model.parent / name]:
            if p.is_file():
                self.haar = cv2.CascadeClassifier(str(p))
                break
        if self.haar is None:
            print("  No face detector available: framing will use saliency only.")

    def clone(self) -> "FaceFinder":
        return FaceFinder(self.model)

    def find(self, bgr) -> list[Rect]:
        h, w = bgr.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(bgr)
            return [] if faces is None else [Rect(*map(float, f[:4])) for f in faces]
        if self.haar is None:
            return []
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        found = self.haar.detectMultiScale(gray, 1.1, 6, minSize=(max(24, w // 40),) * 2)
        return [Rect(*map(float, f)) for f in found]


def salient_point(bgr):
    """Fallback focus for photos without faces: spectral-residual saliency.
    Returns (x, y) as 0..1, deliberately pulled towards the centre."""
    g = cv2.cvtColor(cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(g.astype(np.float64))
    log_amp = np.log(np.abs(f) + 1e-6)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(f)))) ** 2
    sal = cv2.GaussianBlur(sal, (9, 9), 2.5)
    sal = np.where(sal >= np.quantile(sal, 0.85), sal, 0)
    ys, xs = np.mgrid[0:64, 0:64]
    cx = (sal * xs).sum() / sal.sum() / 63
    cy = (sal * ys).sum() / sal.sum() / 63
    return 0.5 + 0.6 * (cx - 0.5), 0.5 + 0.6 * (cy - 0.5)


class PeopleFinder:
    """YOLOX-S general object detector (Apache 2.0), used only for the class "person".
    Finds people whose faces are hidden: from behind, looking down, under hoods."""
    SIZE = 640

    def __init__(self, model: Path):
        if not model.exists():
            try:
                model.parent.mkdir(parents=True, exist_ok=True)
                print(f"Downloading people model -> {model}")
                urllib.request.urlretrieve(PEOPLE_URL, model)
            except Exception as e:
                model.unlink(missing_ok=True)
                raise RuntimeError(f"could not download {PEOPLE_URL} ({e})")
        self.model = model
        self.net = cv2.dnn.readNetFromONNX(str(model))
        grids, strides = [], []
        for st in (8, 16, 32):  # YOLOX predicts on three grids; outputs are grid-relative
            n = self.SIZE // st
            yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), st))
        self.grid, self.stride = np.concatenate(grids), np.concatenate(strides)

    def clone(self) -> "PeopleFinder":
        return PeopleFinder(self.model)

    def find(self, bgr, min_score=0.5) -> list[Rect]:
        h, w = bgr.shape[:2]
        r = min(self.SIZE / h, self.SIZE / w)
        canvas = np.full((self.SIZE, self.SIZE, 3), 114, np.uint8)  # letterbox, top-left aligned
        canvas[:int(h * r), :int(w * r)] = cv2.resize(bgr, (int(w * r), int(h * r)),
                                                      interpolation=cv2.INTER_AREA)
        self.net.setInput(cv2.dnn.blobFromImage(canvas))  # BGR, 0..255, as YOLOX expects
        out = self.net.forward().reshape(-1, 85)
        score = out[:, 4] * out[:, 5]                    # objectness * P(person)
        keep = score > min_score
        if not keep.any():
            return []
        xy = (out[keep, :2] + self.grid[keep]) * self.stride[keep] / r
        wh = np.exp(out[keep, 2:4]) * self.stride[keep] / r
        boxes = [[float(x - bw / 2), float(y - bh / 2), float(bw), float(bh)]
                 for (x, y), (bw, bh) in zip(xy, wh)]
        idx = cv2.dnn.NMSBoxes(boxes, score[keep].tolist(), min_score, 0.45)
        result = []
        for i in np.array(idx).flatten():
            x, y, bw, bh = boxes[i]
            x0, y0 = max(0.0, x), max(0.0, y)
            result.append(Rect(x0, y0, min(w, x + bw) - x0, min(h, y + bh) - y0))
        return result


def detect(path: Path, faces: FaceFinder, people: PeopleFinder | None) -> dict:
    """Raw detections for one photo (the slow part; cached between runs)."""
    bgr, (W, H), date = load_image(path, max_side=1600)
    k = W / bgr.shape[1]
    sx, sy = salient_point(bgr)
    if not date:
        date = time.strftime("%Y:%m:%d %H:%M:%S", time.localtime(path.stat().st_mtime))
    return {
        "W": W, "H": H, "date": date,
        "faces": [list(vars(f.scaled(k)).values()) for f in faces.find(bgr)],
        "bodies": [list(vars(b.scaled(k)).values()) for b in people.find(bgr)] if people else [],
        "focus": [sx * W, sy * H],
    }


def loudness(path: Path) -> float | None:
    """Integrated loudness in LUFS (EBU R128), None for silence."""
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn",
                          "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", err)
    return float(m[-1]) if m and float(m[-1]) > -60 else None


def probe_video(path: Path) -> dict:
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
                                      "-show_format", str(path)], capture_output=True, text=True).stdout)
    v = next(x for x in info["streams"] if x["codec_type"] == "video")
    W, H = int(v["width"]), int(v["height"])
    rot = int(float((v.get("tags") or {}).get("rotate", 0) or 0))
    for sd in v.get("side_data_list") or []:
        rot = int(float(sd.get("rotation", rot) or rot))
    if abs(rot) % 180 == 90:  # phone held upright
        W, H = H, W
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    audio = any(x["codec_type"] == "audio" for x in info["streams"])
    date = ((info.get("format") or {}).get("tags") or {}).get("creation_time", "")
    return {"W": W, "H": H, "date": date.replace("-", ":").replace("T", " ")[:19],
            "duration": float(info["format"].get("duration") or v.get("duration") or 0),
            "fps": float(num) / float(den or 1) if float(num) else 30.0,
            "hdr": v.get("color_transfer") in ("arib-std-b67", "smpte2084"),
            "audio": audio, "lufs": loudness(path) if audio else None}


def estimate_head(b: Rect, H: int) -> Rect:
    """Where the head of a person box probably is."""
    if b.w > 1.1 * b.h:                        # wider than tall: crouching, sitting, lying
        hh = 0.3 * b.h
    elif b.y + b.h > 0.98 * H:                 # cut off at the bottom: mostly upper body
        hh = 0.25 * b.h
    else:                                      # standing; kids' heads are ~1/6 of their height
        hh = 0.15 * b.h
    hh = min(hh, 0.6 * b.w)
    hw = 0.8 * hh
    return Rect(b.cx - hw / 2, b.y + 0.02 * b.h, hw, hh)


@dataclass
class Photo:
    path: Path
    key: str           # name used in overrides.yaml
    W: int             # full-size, upright
    H: int
    date: str
    faces: list        # main subjects: real faces and estimated heads, full-size pixel coords
    estimated: list    # per entry in faces: True if it is a head guessed from a body box
    n_people: int      # everyone found, including small background people
    focus: tuple       # fallback focus point (px) when nobody is found
    bodies: list = field(default_factory=list)
    ov: dict = field(default_factory=dict)
    video: dict | None = None  # set for video clips: duration, fps, hdr, audio, lufs

    def trim(self) -> tuple[float, float]:
        """The part of a video clip that is used, in seconds."""
        a, b = (self.ov.get("trim") or [0, self.video["duration"]])[:2]
        a = min(max(0.0, float(a)), self.video["duration"] - 0.5)
        return a, min(max(a + 0.5, float(b)), self.video["duration"])


def build_photo(path: Path, key: str, raw: dict, ov: dict) -> Photo:
    W, H = raw["W"], raw["H"]
    faces = [Rect(*f) for f in raw["faces"]]
    bodies = [Rect(*b) for b in raw["bodies"]]

    def has_face(b):  # a detected face inside this body box: the face is better
        return any(b.x <= f.cx <= b.x + b.w and b.y - 0.05 * b.h <= f.cy <= b.y + b.h for f in faces)

    heads = [estimate_head(b, H) for b in bodies if not has_face(b)]
    subjects = [(f, False) for f in faces] + [(h, True) for h in heads]
    if subjects:  # ignore people much smaller than the biggest one (background)
        biggest = max(r.h for r, _ in subjects)
        subjects = [s for s in subjects if s[0].h >= 0.4 * biggest]
    return Photo(path, key, W, H, raw["date"], [r for r, _ in subjects],
                 [e for _, e in subjects], len(faces) + len(heads), tuple(raw["focus"]), bodies, ov)


def analyse_all(files: list[tuple[str, Path]], cache: dict, faces: FaceFinder,
                people: PeopleFinder | None, photo_ov: dict, progress=None, save=None) -> dict[str, Photo]:
    """Detect (or reuse cached detections) for every (key, path).

    New photos are analysed on several cores at once, each worker with its own detectors.
    The cache is saved every few seconds, so stopping halfway doesn't lose the work."""
    detectors = f"{'yunet' if faces.yunet is not None else 'haar'}+{'yolox_s' if people else 'none'}"
    jobs = []
    for key, f in files:
        if (photo_ov.get(key) or {}).get("skip"):
            continue
        st = f.stat()
        is_video = f.suffix.lower() in VIDEO_EXTS
        stamp = [MUSIC_VERSION if is_video else ANALYSIS_VERSION, "video" if is_video else detectors,
                 st.st_mtime, st.st_size]
        entry = cache.get(key)
        if not (entry and entry.get("stamp") == stamp):
            jobs.append((key, f, is_video, stamp))

    if jobs:
        cores = os.cpu_count() or 2
        workers = max(1, min(len(jobs), cores // 2, 6))
        local = threading.local()
        spare = [(faces, people)]  # the first worker uses the detectors already loaded

        def work(job):
            key, f, is_video, stamp = job
            if not hasattr(local, "det"):
                with lock:
                    local.det = spare.pop() if spare else (faces.clone(), people.clone() if people else None)
            entry = probe_video(f) if is_video else detect(f, *local.det)
            return key, {"stamp": stamp, **entry}

        lock = threading.Lock()
        threads_before = cv2.getNumThreads()
        cv2.setNumThreads(max(1, cores // workers))  # share the cores instead of fighting over them
        last_save, done = time.time(), 0
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for key, entry in pool.map(work, jobs):
                    cache[key] = entry
                    done += 1
                    report(progress, "Photos and videos", done / len(jobs),
                           f"{done} of {len(jobs)} new: {key}")
                    if not progress:
                        print(f"\r  analysing {done}/{len(jobs)}", end="", flush=True)
                    if save and time.time() - last_save > 10:
                        save()
                        last_save = time.time()
        finally:
            cv2.setNumThreads(threads_before)
            if save:
                save()
        if not progress:
            print()

    photos = {}
    for key, f in files:
        ov = photo_ov.get(key) or {}
        if ov.get("skip") or key not in cache:
            continue
        entry = cache[key]
        if f.suffix.lower() in VIDEO_EXTS:
            photos[key] = Photo(f, key, entry["W"], entry["H"], entry["date"], [], [], 0,
                                (entry["W"] / 2, entry["H"] / 2), [], ov, video=entry)
        else:
            photos[key] = build_photo(f, key, entry, ov)
    return photos


# --------------------------------------------------------------------------- planning

@dataclass
class Panel:
    photo: Photo
    x: int                     # slot position and size in the output frame
    w: int
    h: int
    mode: str                  # pan | push | still | blur
    start: Rect | None = None  # in full-size photo pixels
    end: Rect | None = None
    warnings: list = field(default_factory=list)


@dataclass
class Slide:
    dur: float
    panels: list = field(default_factory=list)
    title: str = ""            # set on a section's first (hero) slide
    subtitle: str = ""
    section: str = ""
    start: float = 0.0         # when the slide starts in the movie (s)
    card: bool = False         # intro / outro: the text stays, it doesn't fade away
    logo: Path | None = None


def plan_panel(p: Photo, x: int, w: int, h: int, cfg: Config, flip: bool) -> Panel:
    layout = p.ov.get("layout")
    if layout == "blur":
        return Panel(p, x, w, h, "blur")
    A = w / h
    W, H = p.W, p.H
    cw, ch = cover_size(W, H, A)
    mode = layout if layout in ("pan", "push", "still") else (
        "push" if p.n_people >= cfg.group_faces else "pan")
    warnings = []
    if cw * cfg.max_upscale < w:
        warnings.append("low resolution")

    # tightest allowed end frame: limited by zoom amount and by source resolution
    min_w = min(cw, max(cw / cfg.max_zoom, w / cfg.max_upscale))
    if mode == "push" and p.n_people >= cfg.group_faces:
        min_w = max(min_w, cw / 1.12)  # groups: barely zoom, everyone stays in frame

    focus = p.ov.get("focus")
    if focus:                                      # manual focus point, 0..1
        ew, cx, cy, face_cy = cw / 1.2, focus[0] * W, focus[1] * H, None
    elif p.faces:
        x0 = min(f.x for f in p.faces)
        x1 = max(f.x + f.w for f in p.faces)
        y0 = min(f.y for f in p.faces)
        y1 = max(f.y + f.h for f in p.faces)
        fh = float(np.median([f.h for f in p.faces]))
        want_h = max((y1 - y0) + 2.5 * fh, 4.0 * fh)  # room for hair above, shoulders below
        want_w = (x1 - x0) + 2.0 * fh
        ew = max(want_w, want_h * A)
        cx, face_cy = (x0 + x1) / 2, (y0 + y1) / 2
        if x1 - x0 > cw:
            warnings.append("people don't fit")
        if all(p.estimated):
            warnings.append("no visible faces, framed by body")
        elif any(p.estimated):
            warnings.append("some heads estimated")
    else:
        ew, face_cy = cw / 1.15, None
        cx, cy = p.focus
        warnings.append("nobody found")

    ew = min(max(ew, min_w), cw)
    eh = ew / A
    if face_cy is not None:
        cy = face_cy + eh * (0.5 - cfg.face_y)  # faces a bit above centre, near the upper third
    end = rect_around(cx, cy, ew, eh, W, H)

    def manual(v):  # [x, y, w] as fractions of the photo -> frame in this slot's shape
        rw = max(min(v[2] * W, cw), 16)
        return rect_around(v[0] * W + rw / 2, v[1] * H + rw / A / 2, rw, rw / A, W, H)

    if p.ov.get("end"):
        end = manual(p.ov["end"])
    if p.ov.get("start"):
        start = manual(p.ov["start"])
    elif mode == "still":
        start = end
    elif mode == "push":
        start = rect_around(end.cx, end.cy, cw, ch, W, H)
    else:
        start = pan_start(p, end, cw, ch, cfg, flip)

    if mode != "still" and not p.ov.get("end") and motion_amount(start, end) < 0.08:
        # photo already matches the frame and the faces fill it: add a slight push-in
        tight = min(cw, max(cw / 1.1, w / cfg.max_upscale))
        end = rect_around(end.cx, end.cy, tight, tight / A, W, H)

    return Panel(p, x, w, h, mode, start, end, warnings)


def pan_start(p: Photo, end: Rect, cw: float, ch: float, cfg: Config, flip: bool) -> Rect:
    """Start frame for a pan: same subject already visible, a short distance away.

    The start frame must contain the faces (or the focus point), so a pan never
    begins on empty ground or sky and then travels a long way to find the person.
    It moves towards the side with more photo to show, at most cfg.max_pan."""
    W, H = p.W, p.H
    horizontal = (W - cw) > (H - ch)               # a cover-size frame can only slide one way
    size, span = (cw, W - cw) if horizontal else (ch, H - ch)
    centre = end.cx if horizontal else end.cy
    home = min(max(centre - size / 2, 0), span)  # cover-size frame centred on the end frame
    if span < 1:
        return Rect(home, 0, cw, ch) if horizontal else Rect(0, home, cw, ch)

    # what has to stay visible: the faces, or else the middle of the end frame
    if p.faces and not p.ov.get("focus"):
        lo = min((f.x if horizontal else f.y) for f in p.faces)
        hi = max((f.x + f.w if horizontal else f.y + f.h) for f in p.faces)
    else:
        lo, hi = centre - 0.15 * size, centre + 0.15 * size
    m = cfg.face_margin * size
    first, last = max(0.0, hi + m - size), min(span, lo - m)  # allowed start positions

    room_before, room_after = home - first, last - home
    if max(room_before, room_after) <= 0:           # subject too big to move around it
        pos = home
    else:
        if abs(room_before - room_after) < 0.1 * size:
            before = flip                           # similar room: alternate direction
        else:
            before = room_before > room_after
        room = room_before if before else room_after
        shift = min(room, cfg.max_pan * ch)         # shift / ch = distance in screen heights
        pos = home - shift if before else home + shift
    return Rect(pos, 0, cw, ch) if horizontal else Rect(0, pos, cw, ch)


def slide_duration(panels: list[Panel], cfg: Config) -> float:
    override = max((float(pn.photo.ov.get("duration", 0)) for pn in panels), default=0)
    if override:
        return override
    amount = max((motion_amount(pn.start, pn.end) for pn in panels if pn.start), default=0)
    need = amount / cfg.max_speed + 0.5 * cfg.fade
    return min(max(cfg.slide_dur, need), cfg.max_dur)


def limit_speed(pn: Panel, dur: float, cfg: Config):
    """If the move is still too fast for the slide's duration, start closer to the end."""
    if pn.start is None or pn.mode == "still" or pn.photo.ov.get("start"):
        return  # a start frame set by hand is kept as it is
    allowed = cfg.max_speed * (dur - 0.5 * cfg.fade)
    amount = motion_amount(pn.start, pn.end)
    if amount > allowed:
        pn.start = lerp_rect(pn.end, pn.start, allowed / amount)
        pn.warnings.append("pan shortened")


def plan_section(photos: list[Photo], cfg: Config, title: str, subtitle: str,
                 section: str, flip: bool) -> list[Slide]:
    """photos[0] is the hero: shown alone, full screen, with the section title over it."""
    W, H = cfg.width, cfg.height
    half = (W - cfg.gap) // 2

    def pairable(p):
        return (cfg.pair_portraits and not p.video and p.W / p.H < cfg.portrait_ar
                and p.ov.get("layout") in (None, "pair"))

    slides, i = [], 0
    while i < len(photos):
        p = photos[i]
        if p.video:
            a, b = p.trim()
            slide = Slide(b - a, [Panel(p, 0, W, H, "video")], section=section)
            if not slides:  # a clip as first item: the title sits on its first seconds
                slide.title, slide.subtitle = title, subtitle
            slides.append(slide)
            i += 1
            continue
        if i > 0 and pairable(p) and i + 1 < len(photos) and pairable(photos[i + 1]):
            panels = [plan_panel(p, 0, half, H, cfg, flip),
                      plan_panel(photos[i + 1], W - half, half, H, cfg, not flip)]
            i += 2
        else:
            panels = [plan_panel(p, 0, W, H, cfg, flip)]
            i += 1
        dur = slide_duration(panels, cfg)
        slide = Slide(dur, panels, section=section)
        if len(slides) == 0:  # hero: the title sits on it first, then fades away
            slide.dur += cfg.title_dur + cfg.title_fade
            slide.title, slide.subtitle = title, subtitle
        for pn in panels:
            limit_speed(pn, slide.dur, cfg)
        slides.append(slide)
        flip = not flip
    return slides


# --------------------------------------------------------------------------- rendering

def blurred_cover(bgr, w, h, darken):
    """The photo cropped to fill w*h, heavily blurred and darkened (for backgrounds)."""
    H0, W0 = bgr.shape[:2]
    cw, ch = cover_size(W0, H0, w / h)
    c = rect_around(W0 / 2, H0 / 2, cw, ch, W0, H0)
    crop = bgr[int(c.y):int(c.y + c.h), int(c.x):int(c.x + c.w)]
    small = cv2.resize(crop, (max(1, w // 16), max(1, h // 16)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.5)
    return cv2.convertScaleAbs(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), alpha=darken)


def affine(scale, tx, ty):
    """Matrix mapping source pixel centres to output pixel centres."""
    return np.array([[scale, 0, tx + 0.5 * scale - 0.5],
                     [0, scale, ty + 0.5 * scale - 0.5]], dtype=np.float64)


class PanelRenderer:
    def __init__(self, pn: Panel):
        self.pn = pn
        bgr, _, _ = load_image(pn.photo.path)
        k = bgr.shape[1] / pn.photo.W  # loaded size vs. planned size (normally 1)
        if pn.mode == "blur":
            self.bg = blurred_cover(bgr, pn.w, pn.h, 0.55)
            fit = min(pn.w / bgr.shape[1], pn.h / bgr.shape[0]) * 0.96
            if fit * 1.05 < 1:  # shrink once, leaving detail for the slight push-in
                bgr = cv2.resize(bgr, None, fx=fit * 1.05, fy=fit * 1.05, interpolation=cv2.INTER_AREA)
                fit = 1 / 1.05
            self.img, self.fit = bgr, fit
            return
        a, b = pn.start.scaled(k), pn.end.scaled(k)
        # pre-shrink the photo so the tightest frame is about 1:1 with the output.
        # Every frame then needs at most max_zoom of downscaling, which bilinear handles cleanly.
        q = min(1.0, pn.w / min(a.w, b.w))
        if q < 1:
            small = cv2.resize(bgr, None, fx=q, fy=q, interpolation=cv2.INTER_AREA)
            q = small.shape[1] / bgr.shape[1]
            bgr, a, b = small, a.scaled(q), b.scaled(q)
        self.img, self.a, self.b = bgr, a, b

    def frame(self, u: float, t: float = 0.0):
        pn = self.pn
        if pn.mode == "blur":
            z = self.fit * (1 + 0.04 * u)
            h, w = self.img.shape[:2]
            out = self.bg.copy()
            cv2.warpAffine(self.img, affine(z, pn.w / 2 - w * z / 2, pn.h / 2 - h * z / 2),
                           (pn.w, pn.h), dst=out, flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_TRANSPARENT)
            return out
        r = lerp_rect(self.a, self.b, u)
        s = pn.w / r.w
        return cv2.warpAffine(self.img, affine(s, -r.x * s, -r.y * s), (pn.w, pn.h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def load_font(path: str | None, size: int):
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = [path, "/System/Library/Fonts/Supplemental/Futura.ttc",
                  "/System/Library/Fonts/Helvetica.ttc",
                  str(windows / "segoeui.ttf"), str(windows / "arial.ttf"),
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "/usr/share/fonts/TTF/DejaVuSans.ttf"]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)


def text_layer(W, H, title, subtitle, font_path, logo_h=0, sizes=(0.1, 0.045)):
    """White title (+ subtitle lines) as an alpha mask, plus a soft shadow mask.
    logo_h reserves room above the text. Returns (alpha, shadow, top of the block)."""
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    f1, f2 = load_font(font_path, int(H * sizes[0])), load_font(font_path, int(H * sizes[1]))
    lines = [(title, f1)] if title else []
    lines += [(t, f2) for t in (subtitle or "").split("\n") if t.strip()]
    boxes = [d.textbbox((0, 0), t, font=f) for t, f in lines]
    heights = [max(b[3] - b[1], int(H * sizes[1])) for b in boxes]
    spacing = int(H * 0.025)
    gap = int(H * 0.05) if logo_h else 0
    total = sum(heights) + spacing * max(0, len(lines) - 1) + logo_h + gap
    top = (H - total) / 2
    y = top + logo_h + gap
    for (t, f), b, hgt in zip(lines, boxes, heights):
        d.text(((W - (b[2] - b[0])) / 2 - b[0], y - b[1]), t, font=f, fill=255)
        y += hgt + spacing
    a = np.asarray(mask, np.float32)[..., None] / 255
    shadow = cv2.GaussianBlur(a[..., 0], (0, 0), H * 0.008)[..., None]
    return a, np.clip(shadow * 1.5, 0, 1), int(top)


class SlideRenderer:
    def __init__(self, slide: Slide, cfg: Config, font: str | None):
        self.slide, self.cfg = slide, cfg
        self.panels = [VideoRenderer(pn, cfg) if pn.mode == "video" else PanelRenderer(pn)
                       for pn in slide.panels]
        self.text = None
        if slide.title or slide.subtitle:
            W, H = cfg.width, cfg.height
            logo = None
            if slide.logo:
                lh = int(H * 0.16)
                im = load_logo(slide.logo, lh)
                logo = np.asarray(im, np.float32)
            text, shadow, top = text_layer(W, H, slide.title, slide.subtitle, font or cfg.font or None,
                                           logo.shape[0] if logo is not None else 0,
                                           (cfg.title_size, cfg.subtitle_size))
            if logo is not None:  # put the logo into the text layer: colour + alpha
                lh, lw = logo.shape[:2]
                x0 = (W - lw) // 2
                self.logo = (top, top + lh, x0, x0 + lw, logo[..., [2, 1, 0]], logo[..., 3:] / 255)
            else:
                self.logo = None
            ys, xs = np.nonzero(shadow[..., 0] > 0.004)  # only blend where the text is
            if ys.size:
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            else:
                y0, y1, x0, x1 = 0, 1, 0, 1
            if self.logo:
                y0, y1 = min(y0, self.logo[0]), max(y1, self.logo[1])
                x0, x1 = min(x0, self.logo[2]), max(x1, self.logo[3])
            self.box = (y0, y1, x0, x1)
            self.text, self.shadow = text[y0:y1, x0:x1], shadow[y0:y1, x0:x1]

    def close(self):
        for pr in self.panels:
            if hasattr(pr, "close"):
                pr.close()

    def frame(self, t: float):
        cfg = self.cfg
        u = ease(t / max(0.1, self.slide.dur - 0.5 * cfg.fade))  # settle before fading out
        out = np.full((cfg.height, cfg.width, 3), cfg.bg_gray, np.uint8)
        for pr in self.panels:
            out[:, pr.pn.x:pr.pn.x + pr.pn.w] = pr.frame(u, t)
        if self.text is None:
            return out
        # title over the photo: blurred + darkened; on section titles everything then fades away
        o = 1.0 if self.slide.card else 1 - ease((t - cfg.title_dur) / cfg.title_fade)
        if o <= 0:
            return out
        if cfg.title_blur > 0 and self.panels:
            h, w = out.shape[:2]
            soft = cv2.resize(out, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
            soft = cv2.resize(cv2.GaussianBlur(soft, (0, 0), 1.5), (w, h), interpolation=cv2.INTER_LINEAR)
            out = cv2.addWeighted(out, 1 - o * cfg.title_blur, soft, o * cfg.title_blur, 0)
        out = cv2.convertScaleAbs(out, alpha=1 - cfg.title_dim * o)
        ta = o * ease((t - 0.4) / 0.9)  # text appears slightly after the slide starts
        y0, y1, x0, x1 = self.box
        region = out[y0:y1, x0:x1].astype(np.float32) * (1 - 0.6 * ta * self.shadow)
        region = region * (1 - ta * self.text) + 255 * ta * self.text
        if self.logo:
            ly0, ly1, lx0, lx1, rgb, alpha = self.logo
            sub = region[ly0 - y0:ly1 - y0, lx0 - x0:lx1 - x0]
            sub[:] = sub * (1 - ta * alpha) + rgb * ta * alpha
        out[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)
        return out


@dataclass
class Workspace:
    """Where a movie's material comes from and where its working files go."""
    name: str
    photos: Path              # one subfolder per section
    music: Path | None        # songs, albums one level deep
    logo: Path | None
    work: Path                # edits, analysis cache, renders, the finished movie
    project_file: Path        # the edits (slideshow.yaml)

    @classmethod
    def from_year(cls, year: Path) -> "Workspace":
        """Old layout: everything inside one year folder."""
        music = next((d for d in year.iterdir() if d.is_dir() and d.name.casefold() in MUSIC_FOLDERS), None)
        return cls(year.name, year, music, find_logo(year), year, year / PROJECT_NAME)

    @classmethod
    def from_project(cls, folder: Path) -> "Workspace":
        """A project folder: project.json (folders, name), edits.yaml, and cache/ (or a local cache)."""
        migrate_project(folder)
        info = read_project(folder)
        photos = project_path(info, "photos", folder)
        logo = project_path(info, "logo", folder) or (find_logo(photos) if photos else None)
        if photos is None:
            raise RuntimeError("The photos folder can't be found. Choose it again on the start page.")
        work = cache_dir(info, folder)
        work.mkdir(parents=True, exist_ok=True)
        return cls(info["name"], photos, project_path(info, "music", folder),
                   logo if logo and logo.exists() else None, work, folder / EDITS_NAME)

    @property
    def movie(self) -> Path:
        return self.work / "movie.mp4" if self.work != self.photos else self.work / f"{self.name}.mp4"


def read_project(folder: Path) -> dict:
    return json.loads((folder / "project.json").read_text(encoding="utf-8"))


def write_project(folder: Path, info: dict):
    tmp = folder / "project.json.tmp"
    tmp.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, folder / "project.json")


def store_path(info: dict, key: str, path: Path | None, folder: Path):
    """Remember a folder both absolutely and relative to the project, so the project keeps
    working when a drive gets another letter or the whole thing moves together."""
    info.pop(key + "_rel", None)
    if path is None:
        info[key] = None
        return
    info[key] = str(Path(path).resolve())
    try:
        info[key + "_rel"] = Path(os.path.relpath(Path(path).resolve(), folder.resolve())).as_posix()
    except ValueError:  # another drive on Windows
        pass


def project_path(info: dict, key: str, folder: Path) -> Path | None:
    for candidate in (info.get(key), info.get(key + "_rel") and (folder / info[key + "_rel"])):
        if candidate and Path(candidate).exists():
            return Path(candidate).resolve()
    return None


def user_dir(kind: str) -> Path:
    """config: the small list of projects. cache: projects that keep their cache off the project folder."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / ("Library/Application Support/Year Review" if kind == "config" else "Library/Caches/Year Review")
    if os.name == "nt":
        base = os.environ.get("APPDATA" if kind == "config" else "LOCALAPPDATA") or str(home)
        return Path(base) / "Year Review" / ("" if kind == "config" else "Cache")
    base = os.environ.get("XDG_CONFIG_HOME" if kind == "config" else "XDG_CACHE_HOME")
    return Path(base or home / (".config" if kind == "config" else ".cache")) / "year-review"


def cache_dir(info: dict, folder: Path) -> Path:
    """Analysis, thumbnails, previews, rendered parts, mixes and the movie: everything that
    can be made again. Inside the project, or on this computer for synced project folders."""
    if info.get("cache") == "local":
        return user_dir("cache") / info.get("id", folder.name)
    return folder / "cache"


def migrate_project(folder: Path):
    """Projects from the previous version kept edits and caches at the top of the folder."""
    if (folder / PROJECT_NAME).exists() and not (folder / EDITS_NAME).exists():
        os.replace(folder / PROJECT_NAME, folder / EDITS_NAME)
    old = [folder / n for n in (CACHE_NAME, RENDER_DIR, ".slideshow_preview", ".slideshow_thumbs", "movie.mp4")]
    if any(p.exists() for p in old):
        (folder / "cache").mkdir(exist_ok=True)
        for p in old:
            if p.exists() and not (folder / "cache" / p.name).exists():
                shutil.move(str(p), str(folder / "cache" / p.name))


def report(progress, step: str, frac: float | None = None, detail: str = ""):
    if progress:
        progress(step, frac, detail)


def find_logo(year: Path) -> Path | None:
    for name in ("logo.svg", "logo.png"):
        if (year / name).exists():
            return year / name
    return None


def load_logo(path: Path, height: int) -> Image.Image:
    """The logo as an RGBA image of the given height. SVG is drawn at that size, so it stays sharp."""
    if path.suffix.lower() == ".svg":
        data = None
        try:
            import resvg_py
            data = bytes(resvg_py.svg_to_bytes(svg_path=str(path), height=height * 2))
        except Exception:
            pass
        try:
            if data is None:
                import cairosvg
                data = cairosvg.svg2png(url=str(path), output_height=height * 2)
        except Exception:
            if shutil.which("rsvg-convert"):
                data = subprocess.run(["rsvg-convert", "-h", str(height * 2), str(path)],
                                      capture_output=True, check=True).stdout
        if data is None:
            raise RuntimeError("To use an SVG logo, install resvg-py (pip install resvg-py)")
        im = Image.open(io.BytesIO(data)).convert("RGBA")
    else:
        im = Image.open(path).convert("RGBA")
    return im.resize((max(1, round(im.width * height / im.height)), height), Image.LANCZOS)


def zscale_available() -> bool:
    if not hasattr(zscale_available, "ok"):
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        zscale_available.ok = " zscale " in out and " tonemap " in out
    return zscale_available.ok


class VideoRenderer:
    """Streams the used part of a clip from ffmpeg, already at the output frame rate.
    Clips close to 16:9 fill the frame, others sit on a blurred copy of themselves."""

    def __init__(self, pn: Panel, cfg: Config):
        self.pn, v, p = pn, pn.photo.video, pn.photo
        a, b = p.trim()
        A, out_a = p.W / p.H, pn.w / pn.h
        self.fill = abs(A / out_a - 1) < 0.12
        if self.fill:
            self.fw, self.fh = pn.w, pn.h
            size = f"scale={pn.w}:{pn.h}:force_original_aspect_ratio=increase,crop={pn.w}:{pn.h}"
        else:
            k = min(pn.w / p.W, pn.h / p.H)
            self.fw, self.fh = int(p.W * k) // 2 * 2, int(p.H * k) // 2 * 2
            size = f"scale={self.fw}:{self.fh}"
        vf = []
        if v["hdr"]:
            if zscale_available():  # iPhone HDR (HLG) -> normal colours
                vf.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                          "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
            else:
                print(f"\n  {p.key}: HDR clip, but this ffmpeg can't convert HDR; colours may look pale")
        vf += [size, f"fps={cfg.fps}"]
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}",
                                      "-i", str(p.path), "-vf", ",".join(vf), "-an",
                                      "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                                     stdout=subprocess.PIPE)
        self.fps, self.n, self.cur = cfg.fps, -1, None

    def _read(self):
        size = self.fw * self.fh * 3
        buf = self.proc.stdout.read(size)
        if len(buf) < size:
            return False
        self.cur = np.frombuffer(buf, np.uint8).reshape(self.fh, self.fw, 3)
        self.n += 1
        return True

    def frame(self, u: float, t: float):
        want = int(t * self.fps + 1e-6)
        while self.n < want and self._read():
            pass
        if self.cur is None:
            return np.zeros((self.pn.h, self.pn.w, 3), np.uint8)
        if self.fill:
            return self.cur
        out = blurred_cover(self.cur, self.pn.w, self.pn.h, 0.55)
        y, x = (self.pn.h - self.fh) // 2, (self.pn.w - self.fw) // 2
        out[y:y + self.fh, x:x + self.fw] = self.cur
        return out

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


HW_ENCODERS = {"videotoolbox": "h264_videotoolbox", "nvenc": "h264_nvenc", "qsv": "h264_qsv", "amf": "h264_amf"}


def ensure_ffmpeg():
    """Use the installed ffmpeg; without one, fetch a ready-made copy once (static-ffmpeg).
    Started by double-click, the usual install folders (Homebrew, MacPorts) may be missing
    from PATH, so they're checked first: an installed ffmpeg can use the graphics chip."""
    extra = ["/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin", "/usr/bin"]
    if os.name == "nt":
        extra = [str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "ffmpeg/bin"),
                 str(Path.home() / "scoop/shims"), "C:/ProgramData/chocolatey/bin"]
    for d in extra:
        if (Path(d) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")).exists() and d not in os.environ["PATH"]:
            os.environ["PATH"] = os.environ["PATH"] + os.pathsep + d
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    try:
        import static_ffmpeg
        print("Getting ffmpeg (only the first time, about 80 MB) ...", flush=True)
        static_ffmpeg.add_paths(weak=False)
    except Exception as e:
        sys.exit(f"ffmpeg is missing and couldn't be downloaded ({e}). Install it from https://ffmpeg.org")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is missing. Install it from https://ffmpeg.org")


def pick_encoder(encoder: str = "auto") -> str:
    """The video encoder to use: the graphics chip's if it works on this computer, else the CPU's."""
    if encoder != "auto":
        return encoder
    if not hasattr(pick_encoder, "found"):
        order = ["videotoolbox"] if sys.platform == "darwin" else ["nvenc", "qsv", "amf"]
        pick_encoder.found = "x264"
        for name in order:  # a tiny test encode tells whether the hardware is really there
            r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=640x360:d=0.2",
                                "-c:v", HW_ENCODERS[name], "-f", "null", "-"], capture_output=True)
            if r.returncode == 0:
                pick_encoder.found = name
                break
    return pick_encoder.found


ENCODER_NAMES = {"videotoolbox": "VideoToolbox (Apple graphics chip)", "nvenc": "NVIDIA graphics card",
                 "qsv": "Intel graphics", "amf": "AMD graphics card", "x264": "CPU (slower)"}


def system_info() -> dict:
    """Which ffmpeg and video encoder this computer uses, for the start window and Settings."""
    enc = pick_encoder()
    return {"ffmpeg": shutil.which("ffmpeg"), "encoder": enc, "encoder_name": ENCODER_NAMES.get(enc, enc),
            "hdr": zscale_available()}


def open_writer(path: Path, cfg: Config, encoder: str):
    encoder = pick_encoder(encoder)
    rate = f"{cfg.bitrate if cfg.width >= 1280 else max(4, cfg.bitrate // 3)}M"
    if encoder in HW_ENCODERS:
        venc = ["-c:v", HW_ENCODERS[encoder], "-b:v", rate]
    else:
        venc = ["-c:v", "libx264", "-preset", "medium", "-b:v", rate, "-maxrate", rate, "-bufsize", rate]
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{cfg.width}x{cfg.height}",
           "-r", str(cfg.fps), "-i", "-",
           # convert with the HD colour matrix and tag it, so players don't shift colours
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           *venc, "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-g", str(cfg.fps * 2), str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render(slides: list[Slide], cfg: Config, out: Path, encoder: str, font: str | None,
           t0: float = 0.0, t1: float | None = None, total: float | None = None, label: str = "",
           on_frame=None):
    """Render the frames from t0 to t1 (seconds in the movie) into one video file."""
    total = slides[-1].start + slides[-1].dur if total is None else total
    t1 = total if t1 is None else t1
    n0, n1 = int(round(t0 * cfg.fps)), int(round(t1 * cfg.fps))
    proc = open_writer(out, cfg, encoder)
    pool = ThreadPoolExecutor(max_workers=2)
    jobs = {}

    def renderer(i):
        for j in (i, i + 1, i + 2):  # decode upcoming photos in the background
            if j < len(slides) and j not in jobs and slides[j].start < t1:
                jobs[j] = pool.submit(SlideRenderer, slides[j], cfg, font)
        return jobs[i].result()

    clock0 = time.time()
    first = next((i for i, sl in enumerate(slides) if sl.start + sl.dur > t0 + 1e-6), len(slides) - 1)
    try:
        for n in range(n0, n1):
            t = n / cfg.fps
            while first < len(slides) - 1 and t >= slides[first].start + slides[first].dur:
                job = jobs.pop(first, None)  # free memory of finished slides
                if job is not None and job.done() and not job.exception():
                    job.result().close()
                first += 1
            active = [i for i in (first, first + 1)
                      if i < len(slides) and slides[i].start <= t < slides[i].start + slides[i].dur] or [first]
            frames = [renderer(i).frame(t - slides[i].start) for i in active]
            img = frames[0]
            if len(frames) == 2:
                a = ease((t - slides[active[1]].start) / cfg.fade)
                img = cv2.addWeighted(frames[0], 1 - a, frames[1], a, 0)
            g = min(1.0, t / 0.6, (total - t) / 0.8)  # fade from / to black
            if g < 1:
                img = cv2.convertScaleAbs(img, alpha=max(g, 0.0))
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
            if on_frame:
                on_frame()
            elif (n - n0) % (cfg.fps * 2) == 0 or n == n1 - 1:
                el = time.time() - clock0
                print(f"\r  {label}rendering {100 * (n - n0 + 1) / max(1, n1 - n0):3.0f}%   "
                      f"({(n - n0 + 1) / max(el, 1e-6):.0f} fps)", end="", flush=True)
    finally:
        proc.stdin.close()
        proc.wait()
        for job in jobs.values():
            if job.done() and not job.exception():
                job.result().close()
        pool.shutdown(wait=False, cancel_futures=True)
    if not on_frame:
        print()
    if proc.returncode:
        raise RuntimeError("the video encoder failed")


RENDER_VERSION = 1      # bump when rendering changes, so cached parts are redone
RENDER_DIR = ".slideshow_render"


def stamp(path) -> list:
    st = Path(path).stat()
    return [str(path), st.st_mtime, st.st_size]


def movie_parts(plan: Plan, cfg: Config, encoder: str, font: str | None) -> list:
    """Split the movie at section starts (right after each crossfade) into parts that can be
    rendered and cached on their own. A part's fingerprint covers everything visible in it,
    measured from the part's own start, so moving or changing other sections doesn't touch it."""
    secs = plan.sections
    bounds = [0.0] + [x.start + cfg.fade for x in secs[1:]] + [plan.total]
    settings = {k: v for k, v in vars(cfg).items()}
    font_sig = stamp(font or cfg.font) if (font or cfg.font) else None
    parts = []
    for i, sec in enumerate(secs):
        b0, b1 = bounds[i], bounds[i + 1]
        idx = [k for k, sl in enumerate(plan.slides) if sl.start < b1 - 1e-6 and sl.start + sl.dur > b0 + 1e-6]
        slides = []
        for k in idx:
            sl = plan.slides[k]
            slides.append({
                "t": round(sl.start - b0, 4), "dur": round(sl.dur, 4), "title": sl.title, "sub": sl.subtitle,
                "card": sl.card, "logo": stamp(sl.logo) if sl.logo else None,
                "panels": [{"file": stamp(pn.photo.path), "mode": pn.mode, "slot": [pn.x, pn.w, pn.h],
                            "a": None if pn.start is None else [round(v, 2) for v in vars(pn.start).values()],
                            "b": None if pn.end is None else [round(v, 2) for v in vars(pn.end).values()],
                            "trim": list(pn.photo.trim()) if pn.photo.video else None} for pn in sl.panels]})
        sig = {"v": RENDER_VERSION, "cfg": settings, "font": font_sig, "encoder": encoder,
               "len": round(b1 - b0, 4), "from_black": b0 == 0, "to_end": round(plan.total - b0, 4) if i == len(secs) - 1 else None,
               "slides": slides}
        digest = hashlib.sha1(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()[:20]
        parts.append({"name": sec.name, "t0": b0, "t1": b1, "file": f"{cfg.width}x{cfg.height}_{digest}.mp4"})
    return parts


def movie_signature(plan: Plan, cfg: Config, ws: Workspace, encoder: str, font: str | None, with_sound: bool):
    """(parts, fingerprint of the whole movie including its sound)."""
    encoder = pick_encoder(encoder)
    parts = movie_parts(plan, cfg, encoder, font)
    with_sound = with_sound and bool(plan.songs or plan.clips)
    sound = sound_signature(plan, cfg, ws.music) if with_sound else None
    sig = hashlib.sha1(json.dumps({"parts": [p["file"] for p in parts], "sound": sound},
                                  sort_keys=True).encode()).hexdigest()
    return parts, sig, encoder, with_sound


def movie_state(plan: Plan, cfg: Config, ws: Workspace, out: Path) -> str:
    """missing | current | outdated: whether the rendered movie matches the edits."""
    if not out.exists():
        return "missing"
    try:
        state = json.loads((ws.work / RENDER_DIR / "renders.json").read_text())
    except (OSError, ValueError):
        return "outdated"
    prev = state.get(str(out.resolve()))
    _, sig, _, _ = movie_signature(plan, cfg, ws, "auto", None, True)
    return "current" if prev and prev["sig"] == sig and prev["mtime"] == out.stat().st_mtime else "outdated"


def render_movie(plan: Plan, cfg: Config, ws: Workspace, out: Path, encoder: str, font: str | None,
                 with_sound: bool, progress=None):
    """Render only the parts that changed since an earlier render, then join them and add the sound.
    If nothing changed at all, the existing file is kept."""
    cache = ws.work / RENDER_DIR
    cache.mkdir(exist_ok=True)
    parts, movie_sig, encoder, with_sound = movie_signature(plan, cfg, ws, encoder, font, with_sound)
    state_file = cache / "renders.json"
    try:
        state = json.loads(state_file.read_text())
    except (OSError, ValueError):
        state = {}
    prev = state.get(str(out.resolve()))
    if out.exists() and prev and prev["sig"] == movie_sig and prev["mtime"] == out.stat().st_mtime:
        print("Nothing changed since the last render: the movie is up to date.")
        return

    todo = [p for p in parts if not (cache / p["file"]).exists()]
    print(f"{len(parts) - len(todo)} of {len(parts)} parts unchanged, rendering {len(todo)}")
    t_start = time.time()
    frames = sum(int(round(p["t1"] * cfg.fps)) - int(round(p["t0"] * cfg.fps)) for p in todo)
    done = [0]
    for n, p in enumerate(todo, 1):
        tmp = cache / ("tmp_" + p["file"])  # renamed only when complete
        name = p["name"].lstrip("_").title() if p["name"].startswith("_") else p["name"]
        step = f"Rendering {name}" + (f" (part {n} of {len(todo)})" if len(todo) > 1 else "")

        def tick(step=step):
            done[0] += 1
            if done[0] % max(1, cfg.fps // 2) == 0:
                report(progress, step, done[0] / max(1, frames))
        label = "" if len(parts) == 1 else f"part {n}/{len(todo)} ({name}): "
        render(plan.slides, cfg, tmp, encoder, font, p["t0"], p["t1"], plan.total, label,
               on_frame=tick if progress else None)
        tmp.rename(cache / p["file"])

    report(progress, "Joining the parts", None)
    video = out.with_name(out.stem + ".video-only.mp4")
    if len(parts) == 1:
        shutil.copyfile(cache / parts[0]["file"], video)
    else:
        listing = cache / "join.txt"
        quoted = lambda f: (cache / f).resolve().as_posix().replace("'", "'\\''")
        listing.write_text("".join(f"file '{quoted(p['file'])}'\n" for p in parts), encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", "-movflags", "+faststart", str(video)], check=True)
        listing.unlink()
    if with_sound:
        audio = soundtrack(plan, cfg, ws, progress)
        report(progress, "Putting picture and sound together", None)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(audio), "-map", "0:v",
                        "-map", "1:a", "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
        video.unlink()
    else:
        os.replace(video, out)  # also replaces an older movie (shutil.move can't on Windows)
    state[str(out.resolve())] = {"sig": movie_sig, "mtime": out.stat().st_mtime, "parts": [p["file"] for p in parts]}
    # keep parts that some render still uses; also keep recent leftovers in case you undo a change
    used = {f for v in state.values() for f in v.get("parts", [])}
    spare = sorted((f for f in cache.glob("*.mp4") if f.name not in used), key=lambda f: f.stat().st_mtime)
    for f in spare[:-12]:
        f.unlink()
    state_file.write_text(json.dumps(state))
    print(f"  done in {time.time() - t_start:.0f} s")


# --------------------------------------------------------------------------- debug output

def write_debug(slides: list[Slide], out_dir: Path):
    """One image per photo with faces (green), estimated heads (pink), people (thin blue),
    start frame (white) and end frame (orange),
    plus a contact sheet of all of them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for si, s in enumerate(slides):
        for pn in s.panels:
            if pn.photo.video:
                continue
            img, _, _ = load_image(pn.photo.path, max_side=900)
            k = img.shape[1] / pn.photo.W
            lw = max(2, img.shape[1] // 300)
            for b in pn.photo.bodies:
                b = b.scaled(k)
                cv2.rectangle(img, (int(b.x), int(b.y)), (int(b.x + b.w), int(b.y + b.h)), (230, 160, 60), 1)
            for f, est in zip(pn.photo.faces, pn.photo.estimated):
                f = f.scaled(k)  # green: real face, pink: head estimated from a body box
                cv2.rectangle(img, (int(f.x), int(f.y)), (int(f.x + f.w), int(f.y + f.h)),
                              (200, 80, 255) if est else (90, 200, 30), lw)
            if pn.start:
                a, b = pn.start.scaled(k), pn.end.scaled(k)
                cv2.rectangle(img, (int(a.x), int(a.y)), (int(a.x + a.w - 1), int(a.y + a.h - 1)), (255, 255, 255), lw)
                cv2.rectangle(img, (int(b.x), int(b.y)), (int(b.x + b.w - 1), int(b.y + b.h - 1)), (0, 150, 255), lw + 1)
                cv2.arrowedLine(img, (int(a.cx), int(a.cy)), (int(b.cx), int(b.cy)), (0, 150, 255), lw, tipLength=0.08)
            label = f"#{si + 1} {pn.photo.key} | {pn.mode} {s.dur:.1f}s"
            if pn.warnings:
                label += " | " + ", ".join(pn.warnings)
            cv2.rectangle(img, (0, 0), (img.shape[1], 30), (20, 20, 20), -1)
            cv2.putText(img, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (60, 190, 255) if pn.warnings else (235, 235, 235), 1, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / f"{si + 1:03d}_{Path(pn.photo.key).stem}.jpg"), img)
            tiles.append(img)
    tile, cols = 320, 5
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * (tile + 8) + 8, cols * (tile + 8) + 8, 3), 30, np.uint8)
    for i, t in enumerate(tiles):
        f = tile / max(t.shape[:2])
        t = cv2.resize(t, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        y, x = 8 + (i // cols) * (tile + 8), 8 + (i % cols) * (tile + 8)
        sheet[y:y + t.shape[0], x:x + t.shape[1]] = t
    cv2.imwrite(str(out_dir / "contact_sheet.jpg"), sheet)
    print(f"Debug images: {out_dir}/")


# --------------------------------------------------------------------------- main

def natural_key(name: str):
    """Sort like a person would: "IMG_2" before "IMG_10", case-insensitive."""
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", name)]


def section_title(folder: str):
    """ "03 Sommerlager - Pfadis" -> ("Sommerlager", "Pfadis"). The number only sets the order."""
    name = re.sub(r"^\d+[\s._-]*", "", folder).strip() or folder
    title, _, subtitle = name.partition(" - ")
    return title.strip(), subtitle.strip()


# --------------------------------------------------------------------------- project

def load_project(ws: Workspace) -> dict:
    f = ws.project_file
    if not f.exists():
        return {}
    if yaml is None:
        sys.exit("pip install pyyaml to read " + PROJECT_NAME)
    return yaml.safe_load(f.read_text()) or {}


def save_project(ws: Workspace, project: dict):
    def clean(v):  # drop empty entries so the file stays readable
        if isinstance(v, dict):
            v = {k: clean(x) for k, x in v.items()}
            return {k: x for k, x in v.items() if x not in (None, "", {}, [])}
        return v
    text = yaml.safe_dump(clean(project), allow_unicode=True, sort_keys=False, width=100)
    ws.project_file.write_text(
        "# Written by the review page. Hand edits are fine; comments get lost on the next save.\n" + text)


def load_cache(ws: Workspace) -> dict:
    try:
        return json.loads((ws.work / CACHE_NAME).read_text())
    except (OSError, ValueError):
        return {}


def save_cache(ws: Workspace, cache: dict):
    try:
        tmp = ws.work / (CACHE_NAME + ".tmp")  # never leave a half-written cache behind
        tmp.write_text(json.dumps(cache))
        os.replace(tmp, ws.work / CACHE_NAME)
        return
    except OSError:
        pass  # read-only folder: just no cache


# --------------------------------------------------------------------------- music

def analyse_song(path: Path) -> dict:
    """Length, loudness, silent start/end, and a coarse waveform (20 values per second, 0..100)."""
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "4000",
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    pcm = np.frombuffer(pcm, np.int16)
    n = len(pcm) // 200
    peaks = np.abs(pcm[:n * 200].reshape(n, 200).astype(np.int32)).max(1) / 32768
    loud = np.nonzero(peaks > 0.02)[0]  # anything above about -34 dB counts as sound
    head = float(loud[0]) * 0.05 if loud.size else 0.0
    tail = float(n - 1 - loud[-1]) * 0.05 if loud.size else 0.0
    return {"duration": len(pcm) / 4000, "head": max(0.0, head - 0.1), "tail": max(0.0, tail - 0.3),
            "lufs": loudness(path), "peaks": np.round(np.sqrt(peaks) * 100).astype(int).tolist()}


def song_parts(duration: float, cuts: list, d: float, head: float = 0.0, tail: float = 0.0) -> list:
    """The parts of a song that stay after the cuts, as [from, to] in song seconds.
    Silence at the very start and end (head, tail) is left out too."""
    cuts = list(cuts) + ([[0, head]] if head > 0.05 else []) + ([[duration - tail, duration]] if tail > 0.05 else [])
    kept, pos = [], 0.0
    for a, b in sorted((max(0.0, float(a)), min(duration, float(b))) for a, b in cuts):
        if b <= a:
            continue
        if a > pos:
            kept.append([pos, a])
        pos = max(pos, b)
    if pos < duration:
        kept.append([pos, duration])
    return [k for k in kept if k[1] - k[0] >= 2 * d] or [[head, duration - tail]]


def song_time(kept: list, d: float, pos: float) -> float:
    """Seconds into the edited song -> seconds in the original file."""
    acc = 0.0
    for i, (a, b) in enumerate(kept):
        if pos < acc + (b - a) or i == len(kept) - 1:
            return a + (pos - acc)
        acc += (b - a) - d
    return 0.0


def song_files(music_dir: Path) -> list:
    """Songs in the music folder and in its subfolders (albums), one level deep.
    Returned as paths relative to the music folder, loose songs first."""
    loose, albums = [], []
    for f in music_dir.iterdir():
        if f.name.startswith("."):
            continue
        if f.is_file() and f.suffix.lower() in MUSIC_EXTS:
            loose.append(f.name)
        elif f.is_dir():
            albums += [f"{f.name}/{g.name}" for g in f.iterdir()
                       if g.is_file() and g.suffix.lower() in MUSIC_EXTS and not g.name.startswith(".")]
    return sorted(loose, key=natural_key) + sorted(albums, key=natural_key)


def music_settings(project: dict, available: list) -> tuple[list, dict]:
    """(playlist in playing order, per-song settings). Songs not in the playlist aren't used."""
    m = project.get("music") or {}
    if "order" in m or "songs" in m:
        songs = m.get("songs") or {}
        seen, order = set(), []
        for f in m.get("order") or []:
            if f in available and f not in seen:
                seen.add(f)
                order.append(f)
        return order, songs
    # older files: one entry per song, every song used in name order
    songs = {k: {kk: vv for kk, vv in (v or {}).items() if kk != "skip"} for k, v in m.items()}
    order = [f for f in available if not (m.get(f) or {}).get("skip")] if m else []
    return order, songs


def edited_length(duration: float, cuts: list, cfg: Config, head=0.0, tail=0.0) -> tuple[list, float]:
    kept = song_parts(duration, cuts, cfg.cut_xfade, head, tail)
    return kept, sum(b - a for a, b in kept) - cfg.cut_xfade * (len(kept) - 1)


def song_gain(info: dict, ov: dict, cfg: Config) -> float:
    """dB to bring a song to the common loudness, plus your own adjustment."""
    auto = 0.0 if info.get("lufs") is None else min(max(cfg.music_lufs - info["lufs"], -12.0), 12.0)
    return round(auto + float(ov.get("gain") or 0), 2)


def music_timeline(playlist: list, song_ov: dict, sec_starts: dict, total: float, cfg: Config) -> list:
    """Place the playlist on the movie timeline.

    Songs play back to back. A pinned song starts exactly when its section starts;
    if the song before is still playing, it fades out there ("overrun"), if it
    ended earlier there is silence ("gap")."""
    d, out, prev = cfg.cut_xfade, [], None
    for name, info in playlist:
        ov = song_ov.get(name) or {}
        e = {"file": name, "duration": info["duration"], "cuts": ov.get("cuts") or [],
             "pin": ov.get("pin") or None, "gap_before": 0.0, "overrun": 0.0, "fade": 0.0,
             "gap_after": 0.0, "warnings": [], "markers": [],
             "gain": song_gain(info, ov, cfg)}
        e["kept"], e["length"] = edited_length(e["duration"], e["cuts"], cfg,
                                               info.get("head", 0), info.get("tail", 0))
        if e["pin"] and e["pin"] not in sec_starts:
            e["warnings"].append(f"section {e['pin']} not found")
            e["pin"] = None
        start = sec_starts[e["pin"]] if e["pin"] else (prev["natural_end"] if prev else 0.0)
        if prev and start < prev["start"] + 1:
            e["warnings"].append("pinned to a section before the previous song starts")
            start = prev["start"] + 1
        e["start"] = start
        e["natural_end"] = e["end"] = start + e["length"]
        if prev:
            if prev["natural_end"] > start:  # previous song still playing: fade it out here
                prev["overrun"] = prev["natural_end"] - start
                prev["end"] = min(prev["natural_end"], start + cfg.music_fade / 2)
                prev["fade"] = cfg.music_fade if prev["end"] < prev["natural_end"] else 0.0
            else:
                e["gap_before"] = start - prev["natural_end"]
        out.append(e)
        prev = e
    for e in out:
        if e["start"] >= total:
            e["warnings"].append("starts after the movie ends")
        elif e["end"] > total:
            e["overrun"] = max(e["overrun"], e["natural_end"] - total)
            e["end"], e["fade"] = total, cfg.music_fade
    if out and out[-1]["end"] < total:
        out[-1]["gap_after"] = total - out[-1]["end"]
    for e in out:  # where each section starts, in the song's own time (for the waveform)
        for sec, t in sec_starts.items():
            if e["start"] <= t < e["end"]:
                e["markers"].append([sec, song_time(e["kept"], d, t - e["start"])])
        e["fade_at"] = song_time(e["kept"], d, e["end"] - e["start"] - e["fade"]) if e["fade"] else None
    return out


def suggest_music(lengths: dict, sec_starts: list, total: float, beam: int = 80) -> tuple[list, dict]:
    """Pick and order songs so that song changes land on section starts and the music
    lasts the whole movie. lengths: {file: length after cuts}, sec_starts: [(name, t)].
    Returns (playlist, {file: section it is pinned to}).

    Costs, in seconds: a song that runs past its section change (fades out, or you cut a
    verse) costs its overrun; silence costs three times as much; a song change that
    isn't on a section start costs 12; cutting the last song short at the movie end
    costs half the cut."""
    bounds = [(n, t) for n, t in sec_starts if t > 1]
    states = [(0.0, 0.0, (), (), None)]  # cost, time, playlist, pins, pin for the next song
    best = None
    for _ in range(len(lengths)):
        grown = []
        for cost, t, seq, pins, pending in states:
            for f, d in lengths.items():
                if f in seq or d < 5:
                    continue
                pins2 = pins + ((f, pending),) if pending else pins
                e = t + d
                if e >= total - 1:  # this song carries the movie to its end
                    c = cost + 0.5 * (e - total)
                    if best is None or c < best[0]:
                        best = (c, seq + (f,), pins2)
                    continue
                near = [(n, b) for n, b in bounds if b > t + 0.4 * d]
                n, b = min(near, key=lambda x: abs(x[1] - e)) if near else (None, None)
                if n and abs(b - e) <= max(15.0, 0.2 * d):
                    c = cost + (e - b if e >= b else 3 * (b - e))
                    grown.append((c, b, seq + (f,), pins2, n))
                else:
                    grown.append((cost + 12, e, seq + (f,), pins2, None))
        if not grown:
            break
        # judge unfinished playlists by cost plus the silence still left to fill
        grown.sort(key=lambda x: x[0] + 0.05 * (total - x[1]))
        states = grown[:beam]
    if best is None:  # not enough music for the whole movie: take the longest playlist
        c, t, seq, pins, _ = max(states, key=lambda x: (x[1], -x[0]))
        best = (c, seq, pins)
    return list(best[1]), dict(best[2])


def clip_sounds(slides: list, cfg: Config) -> list:
    """Video clips whose own sound is used: where they play and how loud."""
    out = []
    for sl in slides:
        for pn in sl.panels:
            v, ov = pn.photo.video, pn.photo.ov
            if pn.mode == "video" and v["audio"] and ov.get("sound", True):
                a, b = pn.photo.trim()
                gain = 0.0 if v["lufs"] is None else min(max(cfg.clip_lufs - v["lufs"], -12.0), 15.0)
                out.append({"path": pn.photo.path, "a": a, "b": b, "start": sl.start, "dur": sl.dur,
                            "gain": gain + float(ov.get("gain") or 0),
                            "music": float(ov.get("music", cfg.clip_music))})
    return out


def make_soundtrack(timeline: list, clips: list, music_dir: Path | None, total: float, cfg: Config, out: Path,
                    progress=None):
    """Mix songs and clip sounds as planned into one audio file as long as the movie."""
    songs = [e for e in timeline if e["start"] < total] if music_dir else []
    cmd, graph, d = ["ffmpeg", "-y", "-v", "error"], [], cfg.cut_xfade
    fmt = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    music = []
    for j, e in enumerate(songs):
        cmd += ["-i", str(music_dir / e["file"])]
        kept = e["kept"]
        base = f"[{j}:a]{fmt},volume={e['gain']:.2f}dB"
        if len(kept) == 1:
            graph.append(f"{base}[r{j}_0]")
        else:
            graph.append(base + f",asplit={len(kept)}" + "".join(f"[r{j}_{k}]" for k in range(len(kept))))
        for k, (a, b) in enumerate(kept):
            graph.append(f"[r{j}_{k}]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[t{j}_{k}]")
        cur = f"t{j}_0"
        for k in range(1, len(kept)):  # join the parts with a short crossfade over each cut
            graph.append(f"[{cur}][t{j}_{k}]acrossfade=d={d}:c1=tri:c2=tri[x{j}_{k}]")
            cur = f"x{j}_{k}"
        clip = e["end"] - e["start"]
        chain = [f"atrim=end={clip:.3f}", "asetpts=PTS-STARTPTS"]
        if kept[0][0] > 0.5:  # the song's own beginning was cut away: don't start abruptly
            chain.append("afade=t=in:d=0.4")
        if e["fade"]:
            chain.append(f"afade=t=out:st={max(0.0, clip - e['fade']):.3f}:d={e['fade']:.3f}")
        chain.append(f"adelay={int(round(e['start'] * 1000))}:all=1")
        graph.append(f"[{cur}]" + ",".join(chain) + f"[m{j}]")
        music.append(f"[m{j}]")
    bus = []
    if music:
        mix = "".join(music) + (f"amix=inputs={len(music)}:normalize=0:dropout_transition=0" if len(music) > 1 else "anull")
        ducks = [c for c in clips if c["music"] < 0.99]
        if ducks:  # music goes down while a clip with sound plays, ramping over the crossfades
            r = max(0.3, cfg.fade)
            expr = "*".join(f"(1-{1 - c['music']:.3f}*clip(min((t-{c['start']:.3f})/{r},"
                            f"({c['start'] + c['dur']:.3f}-t)/{r}),0,1))" for c in ducks)
            mix += f",volume='{expr}':eval=frame"
        graph.append(mix + "[music]")
        bus.append("[music]")
    for k, c in enumerate(clips, len(songs)):
        cmd += ["-i", str(c["path"])]
        n = c["b"] - c["a"]
        f = min(0.6, n / 3)
        graph.append(f"[{k}:a]{fmt},atrim=start={c['a']:.3f}:end={c['b']:.3f},asetpts=PTS-STARTPTS,"
                     f"volume={c['gain']:.2f}dB,afade=t=in:d={f:.2f},afade=t=out:st={n - f:.3f}:d={f:.2f},"
                     f"adelay={int(round(c['start'] * 1000))}:all=1[c{k}]")
        bus.append(f"[c{k}]")
    final = "".join(bus) + (f"amix=inputs={len(bus)}:normalize=0:dropout_transition=0," if len(bus) > 1 else "")
    final += (f"alimiter=limit=0.89:level=false,apad,atrim=end={total:.3f},"
              f"afade=t=out:st={max(0.0, total - cfg.end_fade):.3f}:d={cfg.end_fade}[aout]")
    if not bus:  # nothing to play: silence of the right length
        cmd += ["-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        final = "[0:a]anull[aout]"
    graph.append(final)
    cmd += ["-filter_complex", ";".join(graph), "-map", "[aout]", "-c:a", "aac", "-b:a", "192k",
            "-progress", "pipe:1", "-nostats", str(out)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:  # ffmpeg reports how far it got
        if line.startswith(("out_time_us=", "out_time_ms=")) and line.split("=")[1].strip().isdigit():
            report(progress, "Mixing the sound", min(1.0, int(line.split("=")[1]) / 1e6 / max(total, 1)))
    if proc.wait():
        raise RuntimeError("mixing the sound failed")


def sound_signature(plan: Plan, cfg: Config, music_dir: Path | None) -> str:
    """Fingerprint of everything that goes into the soundtrack."""
    sig = {"songs": [{**{k: e[k] for k in ("file", "start", "end", "fade", "kept", "gain")},
                      "stamp": stamp(music_dir / e["file"])} for e in plan.songs] if music_dir else [],
           "clips": [{**c, "path": stamp(c["path"])} for c in plan.clips],
           "fade": cfg.end_fade, "cut": cfg.cut_xfade, "total": round(plan.total, 4), "v": RENDER_VERSION}
    return hashlib.sha1(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()[:20]


def soundtrack(plan: Plan, cfg: Config, ws: Workspace, progress=None) -> Path:
    """The movie's mixed sound, made once per change and kept in the render cache."""
    music_dir = ws.music
    cache = ws.work / RENDER_DIR
    cache.mkdir(exist_ok=True)
    out = cache / f"sound_{sound_signature(plan, cfg, music_dir)}.m4a"
    if not out.exists():
        if not progress:
            print("  mixing sound ...", flush=True)
        tmp = cache / ("tmp_" + out.name)
        make_soundtrack(plan.songs, plan.clips, music_dir, plan.total, cfg, tmp, progress)
        tmp.rename(out)
        for old in sorted(cache.glob("sound_*.m4a"), key=lambda f: f.stat().st_mtime)[:-4]:
            old.unlink()  # keep a few recent mixes
    return out


# --------------------------------------------------------------------------- the whole plan

@dataclass
class Section:
    name: str
    title: str
    subtitle: str
    hero: str
    start: float
    dur: float
    slides: list   # indices into Plan.slides
    n_photos: int
    kind: str = "section"  # section | intro | outro
    bg: str = ""           # intro / outro: key of the background photo


@dataclass
class Plan:
    sections: list
    slides: list
    photos: dict
    total: float
    songs: list
    videos_skipped: int
    library: dict = field(default_factory=dict)  # every song in the music folder
    clips: list = field(default_factory=list)    # video clips whose sound is used
    cards: dict = field(default_factory=dict)    # intro / outro settings as used
    logo: bool = False


class Detectors:
    def __init__(self, face_model: Path, people_model: Path | None):
        self.faces = FaceFinder(face_model)
        self.people = None
        if people_model:
            try:
                self.people = PeopleFinder(people_model)
            except Exception as e:
                print(f"  People detection off: {e}")


def card_settings(project: dict, name: str) -> dict:
    """Intro and outro with their defaults filled in."""
    out = {}
    for kind, base in (("intro", {"title": f"Jahresrückblick {name}", "subtitle": "", "duration": 7.0}),
                       ("outro", {"title": "Danke!", "subtitle": "", "duration": 10.0})):
        ov = project.get(kind) or {}
        out[kind] = {**base, "enabled": True, "logo": True, "background": None,
                     **{k: v for k, v in ov.items() if v is not None}}
    return out


def card_slide(kind: str, c: dict, bg: Photo | None, cfg: Config, logo: Path | None) -> Slide:
    panels = []
    if bg is not None and not bg.video:
        still = Photo(**{**vars(bg), "ov": {"layout": "push"}})
        panels = [plan_panel(still, 0, cfg.width, cfg.height, cfg, False)]
        panels[0].warnings = []  # a blurred background doesn't need checking
    return Slide(max(3.0, float(c["duration"])), panels, title=c["title"], subtitle=c["subtitle"],
                 section=f"_{kind}", card=True, logo=logo if c["logo"] else None)


def make_plan(ws: Workspace, cfg: Config, project: dict, det: Detectors,
              only: str | None = None, use_cache: bool = True, progress=None) -> Plan:
    section_ov, photo_ov = project.get("sections") or {}, project.get("photos") or {}
    chosen = project.get("section_order") or []
    report(progress, "Looking through the folders", None)
    if not ws.photos.is_dir():
        raise RuntimeError(f"The photos folder {ws.photos} doesn't exist (anymore).")
    skip = {ws.music.resolve() if ws.music else None, ws.work.resolve()}
    dirs = sorted((d for d in ws.photos.iterdir() if d.is_dir() and not d.name.startswith((".", "@"))
                   and d.name.casefold() not in MUSIC_FOLDERS and d.resolve() not in skip),
                  key=lambda d: (chosen.index(d.name) if d.name in chosen else len(chosen), natural_key(d.name)))
    if only:
        dirs = [d for d in dirs if d.name.startswith(only)]

    files = []  # (key, path); key = "section folder/path inside it"
    for d in dirs:
        found = []
        for f in d.rglob("*"):
            rel = f.relative_to(d)
            if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS and f.is_file() \
                    and not any(part.startswith((".", "@")) for part in rel.parts):
                found.append((rel, f))
        # subfolders don't matter: one flat list, sorted by file name
        found.sort(key=lambda x: (natural_key(x[1].name), natural_key(x[0].as_posix())))
        files += [(f"{d.name}/{rel.as_posix()}", f) for rel, f in found]
    if pillow_heif is None and any(f.suffix.lower() in (".heic", ".heif") for _, f in files):
        sys.exit("HEIC photos found: pip install pillow-heif")

    cache = load_cache(ws) if use_cache else {}
    photos = analyse_all(files, cache, det.faces, det.people, photo_ov, progress, save=lambda: save_cache(ws, cache))
    logo = ws.logo
    cards = card_settings(project, ws.name)

    groups = []  # (Section, slides) in movie order
    flip = False
    for d in dirs:
        sec_photos = [photos[k] for k, f in files if k.split("/", 1)[0] == d.name and k in photos]
        if not sec_photos:
            continue
        ov = section_ov.get(d.name) or {}
        title, subtitle = section_title(d.name)
        title, subtitle = ov.get("title") or title, ov.get("subtitle", subtitle) or ""
        by_file = {p.key.split("/", 1)[1]: p for p in sec_photos}
        ordered = []
        for f in (ov.get("order") or []) + list(by_file):  # chosen order first, new photos after
            if f in by_file and by_file[f] not in ordered:
                ordered.append(by_file[f])
        if ov.get("hero") in by_file:  # older files: hero = first photo
            ordered.remove(by_file[ov["hero"]])
            ordered.insert(0, by_file[ov["hero"]])
        sec_slides = plan_section(ordered, cfg, title, subtitle, d.name, flip)
        if len(sec_slides) % 2:
            flip = not flip
        groups.append((Section(d.name, title, subtitle, ordered[0].key.split("/", 1)[1], 0, 0, [],
                               len(sec_photos)), sec_slides))

    if not only and groups:
        for kind in ("outro", "intro"):
            c = cards[kind]
            if not c["enabled"]:
                continue
            first = groups[0][1][0].panels[0].photo if kind == "intro" else groups[-1][1][-1].panels[-1].photo
            bg = photos.get(c["background"]) or first
            sl = card_slide(kind, c, bg, cfg, logo)
            groups.insert(0 if kind == "intro" else len(groups), (Section(f"_{kind}", c["title"], c["subtitle"], "", 0, 0, [], 0,
                                       kind=kind, bg=bg.key if not bg.video else ""), [sl]))

    for _, sec_slides in groups:  # whole frames, so cached parts line up exactly
        for sl in sec_slides:
            sl.dur = max(cfg.fade + 1 / cfg.fps, round(sl.dur * cfg.fps) / cfg.fps)
    slides, sections, t = [], [], 0.0
    for sec, sec_slides in groups:
        sec.start = t if not slides else t - cfg.fade
        sec.dur = sum(x.dur for x in sec_slides) - cfg.fade * (len(sec_slides) - 1)
        sec.slides = list(range(len(slides), len(slides) + len(sec_slides)))
        for sl in sec_slides:
            sl.start = t if not slides else t - cfg.fade
            t = sl.start + sl.dur
            slides.append(sl)
        sections.append(sec)
    total = t

    library = {}
    music_dir = ws.music if ws.music and ws.music.is_dir() else None
    if music_dir and not only:
        names = song_files(music_dir)
        for i, name in enumerate(names):
            f = music_dir / name
            st = f.stat()
            key, stamp = f"music/{name}", [MUSIC_VERSION, st.st_mtime, st.st_size]
            fresh = cache.get(key) and cache[key].get("stamp") == stamp
            report(progress, "Songs", i / max(1, len(names)), name if fresh else f"{name}: loudness and waveform")
            if not fresh:
                if not progress:
                    print(f"  reading {name}", flush=True)
                cache[key] = {"stamp": stamp, **analyse_song(f)}
            library[name] = {k: cache[key][k] for k in ("duration", "peaks", "head", "tail", "lufs")}
    save_cache(ws, cache)
    report(progress, "Planning the movie", None)
    order, song_ov = music_settings(project, list(library))
    for f, info in library.items():
        info["gain"] = song_gain(info, song_ov.get(f) or {}, cfg)
    timeline = music_timeline([(f, library[f]) for f in order], song_ov,
                              {s.name: s.start for s in sections}, total, cfg)
    return Plan(sections, slides, photos, total, timeline, 0, library,
                clip_sounds(slides, cfg), cards, logo is not None)


def plan_json(plan: Plan, cfg: Config) -> dict:
    """Everything the review page needs, in plain JSON."""
    r = lambda x: None if x is None else [round(x.x, 1), round(x.y, 1), round(x.w, 1), round(x.h, 1)]
    return {
        "config": {k: getattr(cfg, k) for k in ("width", "height", "fade", "title_dur", "title_fade",
                                                 "title_dim", "title_blur")},
        "total": plan.total,
        "videos_skipped": plan.videos_skipped,
        "sections": [vars(s) for s in plan.sections],
        "cards": plan.cards,
        "logo": plan.logo,
        "slides": [{"start": s.start, "dur": s.dur, "title": s.title, "subtitle": s.subtitle,
                    "section": s.section, "card": s.card, "logo": bool(s.logo),
                    "panels": [{"key": pn.photo.key, "x": pn.x, "w": pn.w, "h": pn.h, "mode": pn.mode,
                                "start": r(pn.start), "end": r(pn.end), "warnings": pn.warnings}
                               for pn in s.panels]} for s in plan.slides],
        "photos": {k: {"W": p.W, "H": p.H, "faces": [r(f) for f in p.faces], "estimated": p.estimated,
                       "bodies": [r(b) for b in p.bodies], "ov": p.ov,
                       "video": None if not p.video else {**p.video, "trim": list(p.trim())}}
                   for k, p in plan.photos.items()},
        "songs": plan.songs,
        "library": plan.library,
        "clips": [{"start": c["start"], "dur": c["dur"], "music": c["music"], "file": c["path"].name} for c in plan.clips],
    }


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Render a year-review movie.")
    ap.add_argument("year", type=Path, nargs="?", help="year folder with one subfolder per section (old layout)")
    ap.add_argument("--project", type=Path, help="a project folder made by the review app")
    ap.add_argument("-o", "--output", type=Path, help="default: the project's movie.mp4")
    ap.add_argument("--only", help="render only the section with this folder name")
    ap.add_argument("--debug", type=Path, help="write plan drawings and a contact sheet here")
    ap.add_argument("--plan-only", action="store_true", help="analyse and plan, don't render")
    ap.add_argument("--preview", action="store_true", help="960x540 @ 25 fps for quick checks")
    ap.add_argument("--no-music", action="store_true", help="leave the music out")
    ap.add_argument("--encoder", choices=["auto", "x264", *HW_ENCODERS], default="auto")
    ap.add_argument("--font", help="font file for titles (ttf/otf/ttc)")
    models = Path(__file__).resolve().parent / "models"
    ap.add_argument("--model", type=Path, default=models / "face_detection_yunet_2023mar.onnx",
                    help="face detector (downloaded automatically)")
    ap.add_argument("--people-model", type=Path, default=models / "yolox_s.onnx",
                    help="people detector (downloaded automatically)")
    ap.add_argument("--no-people", action="store_true", help="faces only, skip people detection")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached analysis and redo it")
    args = ap.parse_args()
    if not args.project and not args.year:
        ap.error("give a year folder or --project")
    ensure_ffmpeg()
    ws = Workspace.from_project(args.project) if args.project else Workspace.from_year(args.year)

    progress = None
    if os.environ.get("SLIDESHOW_PROGRESS"):  # machine-readable progress for the review app
        def progress(step, frac=None, detail=""):
            print(f"@@PROGRESS {'' if frac is None else f'{frac:.4f}'}|{step}|{detail}", flush=True)

    project = load_project(ws)
    cfg = make_config(project, args.preview)
    if args.font:
        cfg.font = args.font
    det = Detectors(args.model, None if args.no_people else args.people_model)
    plan = make_plan(ws, cfg, project, det, args.only, not args.no_cache, progress)
    if not plan.slides:
        sys.exit(f"No photos found in the section folders of {ws.photos}")

    for s in plan.sections:
        print(f"  {s.start // 60:3.0f}:{s.start % 60:04.1f}  {s.name if s.kind == 'section' else s.kind.title()}: "
              f"{s.n_photos} items, {s.dur // 60:.0f}:{s.dur % 60:02.0f} min")
    pairs = sum(1 for s in plan.slides if len(s.panels) == 2)
    print(f"{len(plan.photos)} photos -> {len(plan.slides)} slides ({pairs} portrait pairs), "
          f"{plan.total // 60:.0f}:{plan.total % 60:02.0f} min")
    if plan.library and not plan.songs:
        print("  no songs chosen yet: pick them on the Music tab of the review page")
    for e in plan.songs:
        note = f", fades out {e['overrun']:.1f} s early" if e["fade"] else ""
        note += f", {e['gap_before']:.1f} s silence before" if e["gap_before"] > 0.05 else ""
        print(f"  music {e['start'] // 60:3.0f}:{e['start'] % 60:04.1f}  {e['file']}{note}")
    flagged = [(i + 1, pn) for i, s in enumerate(plan.slides) for pn in s.panels if pn.warnings]
    if flagged:
        print("Needs attention:")
        for si, pn in flagged:
            print(f"  slide {si:3d}  {pn.photo.key}: {', '.join(pn.warnings)}")

    if args.debug:
        write_debug(plan.slides, args.debug)
    if args.plan_only:
        return
    out = args.output or ws.movie
    render_movie(plan, cfg, ws, out, args.encoder, args.font, not args.no_music and not args.only, progress)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
