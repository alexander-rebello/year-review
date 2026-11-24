# Year Review

Turns a year of photos, clips and music into one movie: faces stay in frame, portraits are
paired, music is matched to the sections, and you fix anything by hand in your browser.

## Start

Double-click the start file for your computer:

| Computer | File | The very first time |
|---|---|---|
| Mac | `Start (Mac).command` | macOS blocks downloaded scripts once. Double-click, close the warning, then **System Settings → Privacy & Security**, scroll down, **Open Anyway**. (Or in Terminal, once: `xattr -dr com.apple.quarantine` followed by a space and the `Year Review` folder dragged into the window.) |
| Windows | `Start (Windows).bat` | If Windows shows "Windows protected your PC": **More info**, then **Run anyway**. |
| Linux | `Start (Linux).sh` | Make it runnable once: right-click, Properties, "Allow executing", or `chmod +x "Start (Linux).sh"`. |

The first start needs the internet and takes a few minutes: it installs **uv** (a small
helper that sets up Python just for this app; nothing else on your computer changes), the
libraries, ffmpeg if you don't have it, and the face and people detection models. Later
starts take seconds.

On a Mac, rendering is several times faster with Homebrew's ffmpeg (`brew install ffmpeg`),
because it uses the Mac's graphics chip. The start window and the Settings tab show which
video encoder is used.

A small window stays open while the app runs, and the app opens in your browser. Closing that
window stops the app. Double-clicking the start file while it's running just opens the
browser again.

## Projects

On the start page, make a **new project**: a name, the **photos folder** (one subfolder per
part of the movie, e.g. `01 Ostern`, `02 Sommerlager`), optionally a **music folder** and a
**logo** (SVG or PNG), and where to save the project.

A project is a normal folder you can keep anywhere, move, back up or copy to another computer:

    Jahresrückblick 2026/
      project.json   which folders the project uses
      edits.yaml     everything you changed: order, framing, music, titles, settings
      cache/         analysis, previews, rendered parts and the finished movie

- **Your photo and music folders are only read**, never changed.
- Everything in `cache/` can be made again. **Free up …** on the start page empties it.
- If the project folder is **synced** (Nextcloud, Dropbox, …), tick **"Keep the cache on this
  computer only"**: then only the small files get synced, not gigabytes of video. A project
  that is open on one computer shows a warning when you open it on another.
- **Open a project folder …** adds a project that was made elsewhere. If a project or one of
  its folders has moved, the start page asks you to locate it.

## Working

- **Photos**: sections on the left (drag ⠿ to reorder), Intro and Outro. Drag photos to reorder,
  drag them into "Left out" to drop them, click one to adjust how it moves. Clips can be trimmed.
- **Music**: click anywhere on the timeline to hear the finished mix from there. "Suggest songs"
  picks songs that fit the sections; drag songs to reorder or to use / not use them.
- **Settings**: timing, titles, motion, sound, resolution.
- **Render movie**: only sections that changed are rendered again. **Save movie …** copies the
  movie to a folder of your choice.

## If something goes wrong

- **The start window shows an error**: it usually says what's missing. Without internet on the
  first start, connect and start again.
- **"Port 8765 is used by another program"**: start from a terminal with
  `uv run review.py --port 8766` inside the `app` folder.
- **Advanced**: with your own Python (3.10 or newer) you can also run
  `pip install -r app/requirements.txt` and `python app/review.py`.
