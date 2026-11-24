#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "opencv-python-headless>=4.8",
#   "numpy",
#   "pillow>=10.1",
#   "pillow-heif",
#   "pyyaml",
#   "resvg-py",
#   "static-ffmpeg",
# ]
# ///
"""
The year-review app.

    python review.py

Opens the app in your browser. Start a project (name, photos folder, music folder,
logo), or open an earlier one. A project is a folder that can live anywhere:
project.json (folders, name), edits.yaml (all your changes) and cache/ (analysis,
previews, rendered parts, the movie). Your photo and music folders are only read.
The list of projects is kept in a small file in your user folder.
Runs only on your own computer (http://localhost).
"""
from __future__ import annotations

import argparse
import atexit
import datetime
import hashlib
import json
import mimetypes
import os
import re
import shutil
import socket
import string
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

import slideshow_proto as engine

HERE = Path(__file__).resolve().parent
REGISTRY = engine.user_dir("config") / "projects.json"   # just the list of project folders
OLD_PROJECTS = engine.user_dir("config") / "projects"     # where the previous version kept them
DOCUMENTS = Path.home() / "Documents"
DEFAULT_LOCATION = (DOCUMENTS if DOCUMENTS.is_dir() else Path.home()) / "Year Review"
PREVIEW_DIR = ".slideshow_preview"
THUMB_DIR = ".slideshow_thumbs"
mimetypes.add_type("audio/mp4", ".m4a")
mimetypes.add_type("audio/flac", ".flac")
mimetypes.add_type("image/svg+xml", ".svg")


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class Job:
    """A long task running in the background, with progress for the page."""

    def __init__(self, kind: str, label: str, **extra):
        self.info = {"kind": kind, "label": label, "frac": None, "detail": "", "running": True,
                     "ok": None, "error": "", "started": time.time(), **extra}
        self.proc = None

    def update(self, step, frac=None, detail=""):
        self.info.update(label=step, frac=frac, detail=detail)

    def finish(self, ok: bool, error: str = "", **extra):
        self.info.update(running=False, ok=ok, error=error, frac=1.0 if ok else self.info["frac"], **extra)

    def cancel(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self.info.update(cancelled=True)


def load_registry() -> dict:
    try:
        reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        reg = {"projects": [], "location": str(DEFAULT_LOCATION)}
        if OLD_PROJECTS.is_dir():  # projects of the previous version stay where they are
            reg["projects"] = [{"path": str(d)} for d in OLD_PROJECTS.iterdir() if (d / "project.json").exists()]
        save_registry(reg)
    return reg


def save_registry(reg: dict):
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, REGISTRY)


_sizes = {}


def folder_size(p: Path) -> int:
    hit = _sizes.get(str(p))
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    total = 0
    if p.is_dir():
        for f in p.rglob("*"):
            try:
                total += f.stat().st_size if f.is_file() else 0
            except OSError:
                pass
    _sizes[str(p)] = (time.time(), total)
    return total


class Locked(Exception):
    pass


class App:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.det = None
        self.ws = None          # the open project
        self.pid = None         # its folder, as a string
        self.edits = {}
        self.plan = self.plan_data = self.cfg = None
        self.phase = "start"    # start | opening | ready
        self.opening = {}       # progress while opening
        self.job = None         # render or mixing
        self.save_job = None    # copying the movie
        self.fonts = find_fonts()
        atexit.register(self.unlock)

    # ---- projects

    def projects(self) -> dict:
        reg, out = load_registry(), []
        for entry in reg["projects"]:
            folder = Path(entry["path"])
            try:
                info = engine.read_project(folder)
            except (OSError, ValueError):
                out.append({"id": str(folder), "path": str(folder), "name": folder.name, "lost": True,
                            "opened": entry.get("opened")})
                continue
            found = {k: engine.project_path(info, k, folder) for k in ("photos", "music", "logo")}
            out.append({"id": str(folder), "path": str(folder), "name": info.get("name", folder.name),
                        "opened": entry.get("opened"), "local_cache": info.get("cache") == "local",
                        **{k: str(v) if v else info.get(k) for k, v in found.items()},
                        "missing": [k for k in found if info.get(k) and found[k] is None],
                        "cache_size": folder_size(engine.cache_dir(info, folder))})
        out.sort(key=lambda x: x.get("opened") or "", reverse=True)
        return {"projects": out, "location": reg.get("location") or str(DEFAULT_LOCATION)}

    def register(self, folder: Path, opened: bool = False):
        reg = load_registry()
        entry = next((e for e in reg["projects"] if Path(e["path"]) == folder), None)
        if entry is None:
            entry = {"path": str(folder)}
            reg["projects"].append(entry)
        if opened:
            entry["opened"] = now()
        save_registry(reg)

    def check(self, info: dict) -> dict:
        name = (info.get("name") or "").strip()
        if not name:
            raise ValueError("Give the project a name.")
        photos = Path(info.get("photos") or "").expanduser()
        if not info.get("photos") or not photos.is_dir():
            raise ValueError("Choose the photos folder (the one with a subfolder for each part of the movie).")
        music = Path(info["music"]).expanduser() if info.get("music") else None
        if music and not music.is_dir():
            raise ValueError(f"The music folder {music} doesn't exist.")
        logo = Path(info["logo"]).expanduser() if info.get("logo") else None
        if logo and (not logo.is_file() or logo.suffix.lower() not in (".svg", ".png")):
            raise ValueError("The logo has to be an SVG or PNG file.")
        return {"name": name, "photos": photos, "music": music, "logo": logo,
                "cache": "local" if info.get("local_cache") else "here"}

    def write_info(self, folder: Path, info: dict, clean: dict):
        old_cache = engine.cache_dir(info, folder) if "cache" in info else None
        info["name"], info["cache"] = clean["name"], clean["cache"]
        for key in ("photos", "music", "logo"):
            engine.store_path(info, key, clean[key], folder)
        new_cache = engine.cache_dir(info, folder)
        if old_cache and old_cache != new_cache and old_cache.exists() and not new_cache.exists():
            new_cache.parent.mkdir(parents=True, exist_ok=True)  # the cache moves along when the option changes
            shutil.move(str(old_cache), str(new_cache))
        engine.write_project(folder, info)

    def create(self, info: dict) -> str:
        clean = self.check(info)
        location = Path(info.get("location") or DEFAULT_LOCATION).expanduser()
        location.mkdir(parents=True, exist_ok=True)
        base = re.sub(r'[<>:"/\\|?*]+', "-", clean["name"]).strip(" .-") or "Project"
        folder, n = location / base, 2
        while folder.exists():
            folder, n = location / f"{base} {n}", n + 1
        folder.mkdir()
        project = {"id": uuid.uuid4().hex[:12], "created": now()}
        self.write_info(folder, project, clean)
        # earlier versions kept edits and analysis inside the photos folder: take them over
        cache = engine.cache_dir(project, folder)
        cache.mkdir(parents=True, exist_ok=True)
        if (clean["photos"] / engine.PROJECT_NAME).exists():
            shutil.copyfile(clean["photos"] / engine.PROJECT_NAME, folder / engine.EDITS_NAME)
        # reuse the analysis of any other project with the same photos folder (largest wins)
        candidates = [clean["photos"] / engine.CACHE_NAME]
        for e in load_registry()["projects"]:
            try:
                other = Path(e["path"])
                o = engine.read_project(other)
                if engine.project_path(o, "photos", other) == clean["photos"].resolve():
                    candidates.append(engine.cache_dir(o, other) / engine.CACHE_NAME)
            except (OSError, ValueError, KeyError):
                pass
        found = [c for c in candidates if c.exists()]
        if found:
            shutil.copyfile(max(found, key=lambda c: c.stat().st_size), cache / engine.CACHE_NAME)
        reg = load_registry()
        reg["location"] = str(location)
        save_registry(reg)
        self.register(folder)
        return str(folder)

    def update(self, pid: str, info: dict):
        folder = Path(pid)
        self.write_info(folder, engine.read_project(folder), self.check(info))

    def add_existing(self, path: str) -> str:
        folder = Path(path).expanduser()
        if not (folder / "project.json").exists():
            raise ValueError("That folder isn't a project (it has no project.json).")
        self.register(folder)
        return str(folder)

    def locate(self, old: str, new: str):
        folder = Path(new).expanduser()
        if not (folder / "project.json").exists():
            raise ValueError("That folder isn't a project (it has no project.json).")
        reg = load_registry()
        for e in reg["projects"]:
            if e["path"] == old:
                e["path"] = str(folder)
        save_registry(reg)

    def forget(self, pid: str):
        if pid == self.pid:
            self.close()
        reg = load_registry()
        reg["projects"] = [e for e in reg["projects"] if e["path"] != pid]
        save_registry(reg)

    def delete(self, pid: str):
        folder = Path(pid)
        try:
            cache = engine.cache_dir(engine.read_project(folder), folder)
        except (OSError, ValueError):
            cache = None
        self.forget(pid)
        if cache and cache.exists():
            shutil.rmtree(cache, ignore_errors=True)
        if (folder / "project.json").exists():
            shutil.rmtree(folder, ignore_errors=True)

    def clear_cache(self, pid: str):
        if pid == self.pid:
            raise ValueError("Close the project first.")
        folder = Path(pid)
        cache = engine.cache_dir(engine.read_project(folder), folder)
        shutil.rmtree(cache, ignore_errors=True)
        _sizes.pop(str(cache), None)

    # ---- lock: one computer at a time (for project folders that are synced)

    def take_lock(self, folder: Path, force: bool):
        lock, me = folder / ".lock", socket.gethostname()
        if lock.exists() and not force:
            try:
                held = json.loads(lock.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                held = {}
            if held.get("host") and held["host"] != me and time.time() - held.get("time", 0) < 24 * 3600:
                since = datetime.datetime.fromtimestamp(held["time"]).strftime("%d.%m. %H:%M")
                raise Locked(f"This project is open on “{held['host']}” (since {since}).")
        lock.write_text(json.dumps({"host": me, "time": time.time()}), encoding="utf-8")

    def unlock(self):
        if self.pid:
            lock = Path(self.pid) / ".lock"
            try:
                if json.loads(lock.read_text(encoding="utf-8")).get("host") == socket.gethostname():
                    lock.unlink()
            except (OSError, ValueError):
                pass

    def open(self, pid: str, force: bool = False):
        with self.lock:
            if self.phase == "opening":
                raise ValueError("A project is being opened already.")
            self.close()
            folder = Path(pid)
            self.take_lock(folder, force)
            try:
                self.ws = engine.Workspace.from_project(folder)
            except Exception as e:
                (folder / ".lock").unlink(missing_ok=True)
                raise ValueError(str(e))
            self.pid = pid
            self.phase, self.opening = "opening", {"step": "Starting", "frac": None, "detail": "", "error": ""}
        threading.Thread(target=self._open, daemon=True).start()

    def _open(self):
        started, times = time.time(), {}

        def progress(step, frac=None, detail=""):
            if step != self.opening.get("step"):
                times[step] = time.time()
            self.opening.update(step=step, frac=frac, detail=detail)
        try:
            if self.det is None:
                progress("Loading the face and people detectors")
                self.det = engine.Detectors(self.args.model, None if self.args.no_people else self.args.people_model)
            self.edits = engine.load_project(self.ws)
            self.refresh(progress)
            self.register(Path(self.pid), opened=True)
            self.phase = "ready"
            steps = list(times.items()) + [("end", time.time())]
            print(f"Opened {self.ws.name} in {time.time() - started:.1f} s: " + ", ".join(
                f"{a} {b[1] - t:.1f} s" for (a, t), b in zip(steps, steps[1:])), flush=True)
        except Exception as e:
            self.opening.update(error=str(e))
            self.unlock()
            self.phase = "start"
            self.ws = self.pid = None

    def close(self):
        with self.lock:
            if self.job and self.job.info["running"]:
                self.job.cancel()
            self.unlock()
            self.ws = self.pid = self.plan = self.plan_data = None
            self.job = self.save_job = None
            self.phase = "start"

    # ---- the open project

    def refresh(self, progress=None):
        ws = self.ws
        self.cfg = engine.make_config(self.edits)
        plan = engine.make_plan(ws, self.cfg, self.edits, self.det, progress=progress)
        order, songs = engine.music_settings(self.edits, list(plan.library))
        self.edits["music"] = {"order": order, "songs": songs}
        self.plan = plan
        data = engine.plan_json(plan, self.cfg)
        data["year"] = ws.name
        data["project"] = self.edits
        data["skipped"] = sorted(k for k, v in (self.edits.get("photos") or {}).items() if (v or {}).get("skip"))
        data["defaults"] = {d.name: engine.section_title(d.name) for d in ws.photos.iterdir() if d.is_dir()}
        data["music_dir"] = bool(ws.music)
        data["sound_sig"] = engine.sound_signature(plan, self.cfg, ws.music)
        data["movie"] = engine.movie_state(plan, self.cfg, ws, ws.movie)
        data["save_dir"] = self.project_info().get("save_dir")
        data["system"] = getattr(self, "system", None)
        default = engine.Config()
        data["settings"] = {
            "spec": [dict(zip(("key", "group", "label", "kind", "min", "max", "step", "help"), x)) for x in engine.SETTINGS],
            "values": {x[0]: engine.config_value(self.cfg, x[0]) for x in engine.SETTINGS},
            "defaults": {x[0]: engine.config_value(default, x[0]) for x in engine.SETTINGS},
            "fonts": self.fonts,
        }
        self.plan_data = data

    def project_info(self) -> dict:
        try:
            return engine.read_project(Path(self.pid))
        except (OSError, ValueError, TypeError):
            return {}

    def save_edits(self, edits: dict):
        with self.lock:
            self.edits = edits
            engine.save_project(self.ws, edits)
            self.refresh()
            return self.plan_data

    def suggest(self):
        with self.lock:
            plan, music = self.plan, self.edits.setdefault("music", {})
            songs = music.setdefault("songs", {})
            lengths = {f: engine.edited_length(info["duration"], (songs.get(f) or {}).get("cuts") or [],
                                               self.cfg, info.get("head", 0), info.get("tail", 0))[1]
                       for f, info in plan.library.items()}
            order, pins = engine.suggest_music(lengths, [(x.name, x.start) for x in plan.sections], plan.total)
            music["order"] = order
            for f in plan.library:
                o = songs.setdefault(f, {})
                o.pop("pin", None)
                if pins.get(f):
                    o["pin"] = pins[f]
            engine.save_project(self.ws, self.edits)
            self.refresh()
            return self.plan_data

    # ---- background jobs

    def busy(self):
        return self.job is not None and self.job.info["running"]

    def start_soundtrack(self):
        """Mix the soundtrack (if it isn't mixed yet) for listening on the timeline."""
        with self.lock:
            plan, cfg, ws = self.plan, self.cfg, self.ws
            if not (plan.songs or plan.clips):
                raise ValueError("No songs in the movie yet.")
            sig = self.plan_data["sound_sig"]
            ready = ws.work / engine.RENDER_DIR / f"sound_{sig}.m4a"
            if ready.exists():
                return {"ready": True, "path": str(ready.relative_to(ws.work)), "sig": sig}
            if self.busy():
                raise ValueError("Wait until the current task has finished.")
            job = self.job = Job("mix", "Mixing the sound", sig=sig)

        def run():
            try:
                path = engine.soundtrack(plan, cfg, ws, job.update)
                job.finish(True, path=str(path.relative_to(ws.work)))
            except Exception as e:
                job.finish(False, str(e))
        threading.Thread(target=run, daemon=True).start()
        return {"ready": False}

    def start_render(self, kind: str, section: str | None):
        with self.lock:
            if self.busy():
                raise ValueError("Wait until the current task has finished.")
            ws = self.ws
            cmd = [sys.executable, "-u", str(HERE / "slideshow_proto.py"), "--project", self.pid,
                   "--model", str(self.args.model), "--people-model", str(self.args.people_model)]
            if self.args.no_people:
                cmd.append("--no-people")
            if kind == "section":
                out = ws.work / PREVIEW_DIR / (re.sub(r"[^\w-]+", "_", section) + ".mp4")
                out.parent.mkdir(exist_ok=True)
                cmd += ["--only", section, "--preview", "-o", str(out)]
                label = f"Preview of {section}"
            else:
                out = ws.movie
                cmd += ["-o", str(out)]
                label = "Rendering the movie"
            job = self.job = Job(kind, label, section=section, file=str(out.relative_to(ws.work)))
            job.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        env={**os.environ, "SLIDESHOW_PROGRESS": "1", "PYTHONIOENCODING": "utf-8"})
        threading.Thread(target=self._follow, args=(job,), daemon=True).start()

    def _follow(self, job: Job):
        lines = []
        for raw in job.proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("@@PROGRESS"):
                frac, step, detail = (line[len("@@PROGRESS "):].split("|") + ["", ""])[:3]
                job.update(step, float(frac) if frac else None, detail)
            elif line:
                lines.append(line)
        code = job.proc.wait()
        if job.info.get("cancelled"):
            job.finish(False, "Cancelled")
        elif code == 0:
            job.finish(True, uptodate=any("up to date" in x for x in lines))
            with self.lock:
                if self.ws is not None:
                    self.plan_data["movie"] = engine.movie_state(self.plan, self.cfg, self.ws, self.ws.movie)
        else:
            job.finish(False, " / ".join(x for x in lines[-4:] if "Traceback" not in x) or f"Failed ({code})")

    def save_movie(self, dest: str):
        """Copy the finished movie out of the project folder."""
        with self.lock:
            src, name = self.ws.movie, self.ws.name
            if not src.exists():
                raise ValueError("Render the movie first.")
            folder = Path(dest).expanduser()
            if not folder.is_dir():
                raise ValueError(f"{folder} is not a folder.")
            target, n = folder / f"{name}.mp4", 2
            while target.exists():
                target, n = folder / f"{name} {n}.mp4", n + 1
            info = engine.read_project(Path(self.pid))
            engine.write_project(Path(self.pid), {**info, "save_dir": str(folder)})
            self.plan_data["save_dir"] = str(folder)
            job = self.save_job = Job("save", f"Saving {target.name}", target=str(target))

        def run():
            try:
                size, done = src.stat().st_size, 0
                tmp = target.with_name("." + target.name + ".part")
                with open(src, "rb") as a, open(tmp, "wb") as b:
                    while chunk := a.read(8 << 20):
                        b.write(chunk)
                        done += len(chunk)
                        job.update(f"Saving {target.name}", done / size)
                tmp.rename(target)
                job.finish(True)
            except Exception as e:
                job.finish(False, str(e))
        threading.Thread(target=run, daemon=True).start()

    def status(self) -> dict:
        return {"phase": self.phase, "project": self.pid, "name": self.ws.name if self.ws else None,
                "opening": self.opening, "job": self.job.info if self.job else None,
                "save": self.save_job.info if self.save_job else None,
                "movie": self.plan_data["movie"] if self.plan_data else None}


# ---- choosing folders and files

def native_pick(kind: str, prompt: str, start: str | None) -> dict:
    """Show the system's own folder / file dialog. Falls back to the page's own browser."""
    start = start if start and Path(start).exists() else None
    try:
        if sys.platform == "darwin":
            what = "folder" if kind == "folder" else 'file of type {"public.svg-image", "public.png"}'
            where = f' default location (POSIX file "{start}")' if start else ""
            script = (f'tell application (path to frontmost application as text) to '
                      f'POSIX path of (choose {what} with prompt "{prompt}"{where})')
            r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            if r.returncode == 0:
                return {"path": r.stdout.strip().rstrip("/") or "/"}
            return {"cancel": True} if "-128" in r.stderr else {"fallback": True}
        if os.name == "nt":
            esc = lambda x: (x or "").replace("'", "''")
            if kind == "folder":
                dlg = (f"$d = New-Object System.Windows.Forms.FolderBrowserDialog; $d.Description = '{esc(prompt)}'; "
                       f"$d.ShowNewFolderButton = $true; $d.SelectedPath = '{esc(start)}'; "
                       "if ($d.ShowDialog($o) -eq 'OK') { $d.SelectedPath }")
            else:
                dlg = (f"$d = New-Object System.Windows.Forms.OpenFileDialog; $d.Title = '{esc(prompt)}'; "
                       "$d.Filter = 'Logo (*.svg;*.png)|*.svg;*.png'; "
                       f"$d.InitialDirectory = '{esc(start)}'; if ($d.ShowDialog($o) -eq 'OK') {{ $d.FileName }}")
            script = ("[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                      "Add-Type -AssemblyName System.Windows.Forms; "
                      "$o = New-Object System.Windows.Forms.Form -Property @{TopMost = $true}; " + dlg)
            r = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", script],
                               capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                return {"fallback": True}
            return {"path": r.stdout.strip()} if r.stdout.strip() else {"cancel": True}
        if shutil.which("zenity"):
            cmd = ["zenity", "--file-selection", f"--title={prompt}"]
            cmd += ["--directory"] if kind == "folder" else ["--file-filter=Logo | *.svg *.png"]
            if start:
                cmd.append(f"--filename={start.rstrip('/')}/")
            r = subprocess.run(cmd, capture_output=True, text=True)
            return {"path": r.stdout.strip()} if r.returncode == 0 else {"cancel": True}
        if shutil.which("kdialog"):
            where = start or str(Path.home())
            cmd = ["kdialog", "--title", prompt]
            cmd += ["--getexistingdirectory", where] if kind == "folder" else ["--getopenfilename", where, "*.svg *.png"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            return {"path": r.stdout.strip()} if r.returncode == 0 else {"cancel": True}
    except OSError:
        pass
    return {"fallback": True}


def browse(path: str | None, files: bool) -> dict:
    """For the page's own folder browser."""
    p = Path(path).expanduser() if path else Path.home()
    if not p.is_dir():
        p = Path.home()
    dirs, found = [], []
    try:
        for x in sorted(p.iterdir(), key=lambda x: x.name.casefold()):
            if x.name.startswith((".", "$")):
                continue
            if x.is_dir():
                dirs.append({"name": x.name, "path": str(x)})
            elif files and x.suffix.lower() in (".svg", ".png"):
                found.append({"name": x.name, "path": str(x)})
    except OSError:
        pass
    drives = [f"{d}:\\" for d in string.ascii_uppercase if os.name == "nt" and Path(f"{d}:\\").exists()]
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "dirs": dirs, "files": found,
            "drives": drives, "home": str(Path.home())}


def reveal(path: Path):
    """Show a file in Finder / Explorer / the file manager."""
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)])
    elif os.name == "nt":
        subprocess.run(["explorer", f"/select,{path}"])
    elif shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", str(path.parent)])


def find_fonts() -> list:
    """Font files on this computer, for the title font setting."""
    dirs = ["/System/Library/Fonts", "/System/Library/Fonts/Supplemental", "/Library/Fonts",
            str(Path.home() / "Library/Fonts"), "/usr/share/fonts", str(Path.home() / ".fonts"),
            str(Path.home() / ".local/share/fonts")]
    if os.name == "nt":
        dirs += [str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"),
                 str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Microsoft/Windows/Fonts")]
    found = {}
    for d in dirs:
        p = Path(d)
        if p.is_dir():
            for f in p.rglob("*"):
                if f.suffix.lower() in (".ttf", ".otf", ".ttc") and not f.name.startswith("."):
                    found.setdefault(f.stem, str(f))
    return [{"name": n, "path": found[n]} for n in sorted(found, key=str.casefold)]


class Handler(BaseHTTPRequestHandler):
    app: App = None

    def log_message(self, *a):  # keep the terminal quiet
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, body: bytes, ctype: str, cache=True):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path):
        """With HTTP range support, which browsers need to seek in audio and video."""
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        if m and (m[1] or m[2]):
            if m[1]:
                start = int(m[1])
                end = int(m[2]) if m[2] else size - 1
            else:
                start = max(0, size - int(m[2]))
            end = min(end, size - 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    @staticmethod
    def inside(path: Path, folder: Path | None) -> bool:
        if folder is None:
            return False
        try:
            path.resolve().relative_to(folder.resolve())
            return path.is_file()
        except ValueError:
            return False

    def do_GET(self):
        app = self.app
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                self.send_bytes((HERE / "review.html").read_bytes(), "text/html; charset=utf-8", cache=False)
            elif url.path == "/api/status":
                self.send_json(app.status())
            elif url.path == "/api/projects":
                self.send_json(app.projects())
            elif url.path == "/api/browse":
                self.send_json(browse(q.get("path"), q.get("files") == "1"))
            elif url.path == "/font":
                path = Path(q["path"])
                ok = any(f["path"] == str(path) for f in app.fonts)
                self.send_file(path) if ok else self.send_error(404)
            elif app.phase != "ready" or app.ws is None:
                self.send_error(409, "No project is open")
            elif url.path == "/api/plan":
                with app.lock:
                    self.send_json(app.plan_data)
            elif url.path == "/photo":
                self.photo(q["key"], int(q.get("size", 480)))
            elif url.path == "/logo":
                self.send_file(app.ws.logo) if app.ws.logo else self.send_error(404)
            elif url.path == "/media":
                base = {"music": app.ws.music, "photos": app.ws.photos, "work": app.ws.work}.get(q.get("kind"))
                path = (base / q["path"]) if base else None
                self.send_file(path) if path and self.inside(path, base) else self.send_error(404)
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def photo(self, key: str, size: int):
        ws = self.app.ws
        path = ws.photos / key
        if not self.inside(path, ws.photos):
            return self.send_error(404)
        size = min(max(size, 120), 2400)
        st = path.stat()
        name = hashlib.sha1(f"{key}|{st.st_mtime}|{st.st_size}|{size}".encode()).hexdigest()
        thumb = ws.work / THUMB_DIR / f"{name}.jpg"
        if not thumb.exists():
            thumb.parent.mkdir(exist_ok=True)
            tmp = thumb.with_name("tmp_" + thumb.name)
            if path.suffix.lower() in engine.VIDEO_EXTS:  # a frame from the clip
                photo = self.app.plan.photos.get(key) if self.app.plan else None
                info = (photo.video if photo else None) or {}
                at = min(1.0, info.get("duration", 2) / 2)
                vf = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
                      "zscale=t=bt709:m=bt709:r=tv,format=yuv420p," if info.get("hdr") and engine.zscale_available() else "")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{at}", "-i", str(path), "-frames:v", "1",
                                "-vf", vf + f"scale='min({size},iw)':-2", "-f", "image2", str(tmp)])
            else:
                bgr, _, _ = engine.load_image(path, max_side=size)
                cv2.imwrite(str(tmp), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if tmp.exists():
                tmp.replace(thumb)
        if thumb.exists():
            self.send_bytes(thumb.read_bytes(), "image/jpeg")
        else:
            self.send_error(404)

    def do_POST(self):
        app = self.app
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            if self.path == "/api/projects/new":
                pid = app.create(body)
                app.open(pid)
                self.send_json({"id": pid})
            elif self.path == "/api/projects/open":
                if body.get("info"):
                    app.update(body["id"], body["info"])
                app.open(body["id"], bool(body.get("force")))
                self.send_json({"ok": True})
            elif self.path == "/api/projects/add":
                self.send_json({"id": app.add_existing(body["path"])})
            elif self.path == "/api/projects/locate":
                app.locate(body["id"], body["path"])
                self.send_json({"ok": True})
            elif self.path == "/api/projects/forget":
                app.forget(body["id"])
                self.send_json({"ok": True})
            elif self.path == "/api/projects/delete":
                app.delete(body["id"])
                self.send_json({"ok": True})
            elif self.path == "/api/projects/clear-cache":
                app.clear_cache(body["id"])
                self.send_json({"ok": True})
            elif self.path == "/api/projects/close":
                app.close()
                self.send_json({"ok": True})
            elif self.path == "/api/pick":
                self.send_json(native_pick(body.get("kind", "folder"), body.get("prompt", "Choose"), body.get("start")))
            elif self.path == "/api/reveal":
                reveal(Path(body["path"]))
                self.send_json({"ok": True})
            elif app.phase != "ready" or app.ws is None:
                self.send_json({"error": "No project is open."}, 409)
            elif self.path == "/api/project":
                self.send_json(app.save_edits(body))
            elif self.path == "/api/suggest":
                self.send_json(app.suggest())
            elif self.path == "/api/soundtrack":
                self.send_json(app.start_soundtrack())
            elif self.path == "/api/render":
                app.start_render(body.get("kind", "movie"), body.get("section"))
                self.send_json({"ok": True})
            elif self.path == "/api/cancel":
                if app.job:
                    app.job.cancel()
                self.send_json({"ok": True})
            elif self.path == "/api/save-movie":
                app.save_movie(body["dest"])
                self.send_json({"ok": True})
            else:
                self.send_error(404)
        except Locked as e:
            self.send_json({"error": str(e), "locked": True}, 409)
        except ValueError as e:
            self.send_json({"error": str(e)}, 400)
        except Exception as e:
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    ap = argparse.ArgumentParser(description="The year-review app.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser")
    models = HERE / "models"
    ap.add_argument("--model", type=Path, default=models / "face_detection_yunet_2023mar.onnx")
    ap.add_argument("--people-model", type=Path, default=models / "yolox_s.onnx")
    ap.add_argument("--no-people", action="store_true")
    args = ap.parse_args()
    if engine.yaml is None:
        sys.exit("pip install pyyaml")
    url = f"http://localhost:{args.port}/"
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:  # already running? then just show it
        try:
            urllib.request.urlopen(url + "api/status", timeout=3)
            print(f"Year review is already running: {url}")
            if not args.no_browser:
                webbrowser.open(url)
            return
        except OSError:
            sys.exit(f"Port {args.port} is used by another program. Start with --port 8766")
    engine.ensure_ffmpeg()
    info = engine.system_info()
    print(f"ffmpeg: {info['ffmpeg']}\nVideo encoding: {info['encoder_name']}")
    if info["encoder"] == "x264" and sys.platform == "darwin":
        print("  Rendering would be much faster with Homebrew's ffmpeg:  brew install ffmpeg")
    if not info["hdr"]:
        print("  This ffmpeg can't convert iPhone HDR clips; they may look pale.")
    Handler.app = App(args)
    Handler.app.system = info
    print(f"Year review: {url}   (close this window or press Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
