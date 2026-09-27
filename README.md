# loopcutter

Cut sample-accurate loops out of full tracks, driven by a plain-text manifest.

loopcutter is for live sets built from many short loops played at once, in the
spirit of Richie Hawtin's DE9 mixes. When eight loops run together for a
minute, a loop that is three samples too long drifts audibly. loopcutter
derives every loop's length from its tempo and bar count and rounds once. It
finds each track's beat grid, puts every start on a beat, and checks every
file before it calls a run a success.

The manifest is the source of truth. Keep it in version control and every
loop on disk can be rebuilt from one CSV file.

## Features

- **Exact lengths.** `samples = round(bars × beats_per_bar × sample_rate × 60 / bpm)`,
  rounded once at the end, never taken from a second timecode.
- **A beat grid per track.** `scan` fits a constant-tempo grid to the beats
  that [beat_this](https://github.com/CPJKU/beat_this) detects, finds which
  beat is the one, and moves the grid onto the audio's own attacks.
- **Mark loops by ear.** Set memory loops in rekordbox (or cues and saved loops
  in Serato) on a waveform, then `import` them. Each start is snapped onto the
  grid after the DJ app's own timing offset is removed, and every shift is
  reported.
- **Click-free edges.** The start moves back to the nearest zero crossing
  within 2 ms, never forward, so the attack is never clipped. The end moves
  with it, so the length never changes.
- **Roll variations.** One manifest row can produce 4, 2, 1 and ½-bar versions
  from the same start, ready to stack in a clip launcher for beat-division rolls.
- **Checks that can fail.** Every file is checked for exact length, sample
  rate, headroom against its source, silence and DC offset. The loop's own
  audio is checked for where its attacks sit against its beats, and its tempo
  against the analysis.
- **Tagged and filed.** AIFF loops carry BPM, key and stem tags (ID3v2.3) and
  are filed by stem and tempo band, ready for a sample browser.
- **Stems and one-shots.** `stems` separates a master into drums, bass,
  vocals and more at the master's own rate, and stem rows are cut from them.
  One-shot rows cut a phrase or stab from a start to an end.
- **Seamless tonal loops.** An optional pre-roll crossfade blends a loop's
  tail into the audio just before its start, so a pad wraps without a dip.
- **Session key planning.** From a list of keys, works out which one to three
  session keys cover the most loops within a transposition limit, and the
  exact shift each loop needs.

## Status

Version 0.2. The beat grid, the DJ-app import, stems, one-shots and the
checks are in place, tested, and proven on a real library. See
[Roadmap](#roadmap) for what's next and [Known issues](#known-issues).

## Install

Needs Python 3.12 or later.

```bash
git clone https://github.com/aaronlsmiles/loopcutter.git
cd loopcutter
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,analysis,markers]"
pytest -q
```

Optional extras:

| Extra | Adds | For |
|---|---|---|
| `analysis` | beat_this, librosa | `scan`: beat grids, attack phase |
| `markers` | pyrekordbox, serato-tools | `import` and the DJ-app grid cross-check |
| `verify` | librosa | the `beat_alignment` and measured-tempo checks in `cut` |
| `sheets` | openpyxl | reading XLSX key lists |
| `stems` | audio-separator, onnxruntime, audioread | the `stems` command |

beat_this runs on PyTorch and uses the Apple GPU when there is one; its model
weights download on first use.

Audio is read with libsndfile, which handles WAV, AIFF, FLAC and MP3. `prep`
reads AAC, M4A and anything else through [ffmpeg](https://ffmpeg.org/), so
install it if your sources include those. Key cross-checks use
[keyfinder-cli](https://github.com/EvanPurkhiser/keyfinder-cli) when it is on
your `PATH`.

## Workspace

Your masters, analysis and loops are private working data, so they live in a
workspace outside this repository:

```bash
loopcutter init ~/loops-workspace
cd ~/loops-workspace
```

`init` refuses to create a workspace inside a git repository that has a
remote, so private audio can't end up pushed by accident. Every command finds
the workspace the way git finds a repository, by walking up from the current
directory to the folder holding `loopcutter.toml`.

| Folder | Holds |
|---|---|
| `masters/` | Lossless working copies, one per source, all at one rate. |
| `analysis/` | `tracks.csv` (one row per track) and `beats/`, the cached beats. |
| `manifests/` | Your manifests. |
| `reports/` | Scan reports and manifest backups. |
| `loops/aiff/`, `loops/wav/` | Cut loops, filed as `<stem>/<tempo band>/`. |
| `stems/` | Separated stems. |

`loopcutter.toml` settings:

| Key | Default | What it does |
|---|---|---|
| `[audio] sample_rate` | `48000` | The rate of every master. A cut never resamples. |
| `[audio] subtype` | `"PCM_24"` | The masters' sample format. |
| `[audio] headroom_db` | `3.0` (0 if absent) | Gain taken off every master, so overs from decoding and resampling loud sources fit instead of clipping. Changing it only affects masters made afterwards. |
| `[prep] replaced_dir` | `"masters/.replaced"` | Where a copy of a master built from a lossy file is kept when a lossless copy replaces it. Point it outside `masters/` if you drag that whole folder into a DJ app. |
| `[sources] paths` | `[]` | Folders or files that `prep` reads when given none. |
| `[snap] max_shift_ms` | `60` | The furthest a start may be moved onto the grid. |
| `[marking] app` | `"rekordbox"` | What `import` reads when `--from` isn't given: `rekordbox`, `rekordbox-xml` or `serato`. |
| `[marking] playlist` | `""` | The rekordbox playlist that holds your masters. |

## Quick start

1. Make masters and analyse them:

   ```bash
   loopcutter prep ~/Music/sources
   loopcutter scan
   ```

2. Mark loops in rekordbox (see [Marking loops in DJ software](#marking-loops-in-dj-software))
   and import them, or write a manifest by hand. [manifests/example.csv](manifests/example.csv)
   shows the common columns.

3. Check that every row resolves, then cut:

   ```bash
   loopcutter cut manifests/session.csv --dry-run
   loopcutter cut manifests/session.csv
   ```

4. Spot-check an output. The bar count must come back whole:

   ```bash
   loopcutter inspect "loops/aiff/full/125-129/Some Artist - Some Track [A1][128][4bar][8A].aiff" --bpm 128
   #   at 128.0 BPM that is 16.0000 beats (4.0000 bars)
   ```

   If it reads 3.9996 bars, something upstream is using timecode arithmetic.

Prove the pipeline on a three-row manifest first. Load those loops into your
DAW, set them looping and listen for a full minute, because drift and seam
clicks only show on repeat. Then scale up. A batch that fails on row 280 has
wasted a lot of time that three rows would have saved.

## The manifest

| Column | Required | Notes |
|---|---|---|
| `source` | yes | Path to the full track, normally its master. |
| `label` | yes | Your cue or ID reference, such as `A1` or `16.2`. |
| `bars` | yes | Loop length in bars. |
| `bpm` | yes | The track's tempo. |
| `start` | one of | `M:SS.mmm`, `H:MM:SS.mmm` or bare seconds. |
| `downbeat` + `start_bar` | one of | A precise downbeat time plus a 1-based bar number counted from it. |
| `kind` | no | `loop` (the default) or `oneshot`. |
| `end` | one-shots | Where a one-shot ends. A one-shot needs `source`, `label`, `start` and `end`, not `bars` or `bpm`. |
| `xfade_ms` | no | Pre-roll crossfade length for this loop; overrides `--xfade-ms`. |
| `track_id` | no | The track's row in `tracks.csv`. `resolve` fills `source`, `bpm` and `key` from it. |
| `snap` | no | `beat` or `bar` asks `resolve` to move `start` onto the grid. Must be empty before `cut`. |
| `variations` | no | Extra bar lengths from the same start, such as `"2,1,0.5"`. |
| `beats_per_bar` | no | Defaults to 4. |
| `artist`, `track` | no | Used to build the output filename. |
| `stem` | no | Written into the filename and the tags, and picks the library folder. |
| `key` | no | Written into the filename and the tags, and read by the key planner. |
| `notes` | no | Ignored by the tool. For you. |

Give either `start` or `downbeat` plus `start_bar`, never both. The loader
rejects rows that give both, rows with no start, rows that still say
`snap=beat` or `snap=bar`, and manifests whose rows would produce the same
filename, and it names the row number every time. Relative `source` paths
resolve against the current directory. Use `--audio-root` to point somewhere
else.

For rows you type by hand, give a `track_id`, a rough `start` and
`snap=beat`, then run `loopcutter resolve manifests/set.csv`. It backs the
manifest up to `reports/`, fills the missing columns from `tracks.csv`, snaps
each start and reports how far it moved. A start further than `max_shift_ms`
from the nearest beat is refused, and the row is named along with the beats
and bar either side of it.

## Getting the start right

`cut` cuts exactly where the manifest says. It never moves a start by more
than the 2 ms zero-crossing search, and it never runs a model. Placement
happens earlier, in `import` and `resolve`, where every move is reported.

- **Mark on the master.** DJ apps decode MP3s with different padding, so a cue
  read off an MP3 can sit tens of milliseconds from the same moment in the
  audio this tool decodes. Marking on the lossless masters that `prep` makes
  removes that offset. For marks made on the original files, `scan --app`
  measures the app's offset per track and `import` removes it.
- **Give milliseconds.** A time typed to the whole second lands anywhere up to
  half a second from the downbeat, which at 125 BPM is the wrong beat about
  half the time.
- **Get the tempo exact.** A 4-bar loop cut at 134 BPM from a 136 BPM track is
  105 ms too long on every repeat. `scan` measures each track's tempo; the
  `bpm_match` check catches a manifest that disagrees.

## Marking loops in DJ software

1. **`loopcutter prep`** makes 24-bit masters at the workspace rate from your
   sources. Existing masters are not overwritten, because Mixed In Key or a
   DJ app may have written tags to them; the one exception is below. A changed
   source is reported instead; delete the master to rebuild it.
2. **Import the masters into rekordbox** as one playlist, name it in
   `loopcutter.toml` under `[marking] playlist`, and fix any grid whose first
   beat is wrong.
3. **Mark memory loops** with quantise on. Hot cues are ignored, so they stay
   free for performing. Comment each loop with its label, then its stem, then
   any roll lengths: `A1 bass 2,1` is loop A1, cut from the bass stem, with
   extra 2 and 1-bar versions. A memory cue with no loop gets the default
   length of 4 bars. Close rekordbox before importing.
4. **`loopcutter scan --app rekordbox`** analyses each master and compares the
   grid with rekordbox's.
5. **`loopcutter import --from rekordbox --playlist NAME`** reads the marks,
   snaps them and writes a resolved manifest to `manifests/`, reporting the
   median and range of the moves. Serato users run `import --from serato`;
   `--from rekordbox-xml --xml FILE` reads a rekordbox XML export instead of
   the database. Marks that can't be placed are skipped and named, and the
   command exits 1 so they can't be missed.
6. **Review the manifest.** Every row's `notes` say where the mark was and how
   far it moved.
7. **`loopcutter cut`** the manifest.

## Commands

### `init`

```bash
loopcutter init [DIR]
```

Creates a workspace with its folders and a default `loopcutter.toml`. Running
it again keeps your edits.

### `prep`

```bash
loopcutter prep [FILES OR FOLDERS ...]
```

Decodes each source once, resamples it once, takes `headroom_db` off, and
writes a 24-bit AIFF master, adding a row to `tracks.csv`. Folders are
searched recursively, skipping hidden folders and the workspace's own masters.
Masters are named after the source file, so a lossless
file and a lossy copy with the same name make the same master: the lossless
file wins, and the lossy one is skipped with a note (names are
compared as a case-insensitive filesystem sees them, an `.m4a` counts as
lossless only if it holds ALAC, and the two must be the same length to within
0.25 s). A master already built from a lossy file is rebuilt when a lossless
copy of the same track appears; a different track that shares a name is
reported as CHANGED and left alone. The rebuild happens beside the masters
and swaps in under the old master's name only when it's complete: a copy of
the old master is kept in `replaced_dir`, its tags (except DJ-app cue data)
are copied onto the new one, your `override_*` columns are kept, and the rest
of the track's analysis is left for `scan`. Re-analyse the track in your DJ
app too, because its cues were placed on the old audio. Any other pair of sources that would
make the same master is refused before anything is written. Loud MP3s
decode above full scale; the headroom keeps those peaks, and any sample still
over is clipped and counted.

### `scan`

```bash
loopcutter scan [TRACK_ID ...] [--app serato|rekordbox] [--force]
```

Analyses each master once: tempo, bar phase, attack phase, key, and with
`--app` the DJ app's tempo and offset. It writes them to `tracks.csv` with a
flag for anything doubtful:

| Flag | Means |
|---|---|
| `grid-fit` | Under 90% of the detected beats fit one steady grid: a tempo change, a long break or a swung rhythm. Snapping uses the whole-track grid wherever it still fits the beats around a mark, and a local grid only where one fits clearly better (a real tempo change). |
| `bar-phase` | The downbeats disagree about which beat is the one. |
| `phase` | The attacks don't agree on where the beat sits. Only rekordbox marks made on the master are kept as they are. |
| `key` | The key tag and keyfinder disagree. |
| `app-bpm` | The DJ app's tempo differs by more than 0.05 BPM. |
| `half-beat` | The DJ app's grid sits half a beat from ours. |
| `app-offset` | On a lossless file, the app's grid sits more than 20 ms from ours. |

`override_bpm` and `override_phase_ms` in `tracks.csv` are yours: `scan` never
touches them, and snapping uses them when set. Tracks already analysed are
skipped unless you pass `--force`. A report of each run goes in `reports/`.

### `import`

```bash
loopcutter import [--from rekordbox|rekordbox-xml|serato] [FILES ...] [--playlist NAME] [--xml FILE] [--out FILE] [--max-shift-ms N]
```

Turns marked loops into a resolved manifest. `--from` defaults to `[marking]
app`. An existing file is never overwritten.

### `resolve`

```bash
loopcutter resolve MANIFEST [--max-shift-ms N]
```

Fills rows from `tracks.csv` and snaps starts marked `snap=beat` or
`snap=bar`, after backing the manifest up.

### `stems`

```bash
loopcutter stems TRACK_ID ... [--engine audio-separator|demucs] [--model NAME]
```

Separates each master with [audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
(Demucs `htdemucs_6s` by default: drums, bass, vocals, guitar, piano and
other), or with the Demucs command line as a fallback. Separators work at
44.1 kHz, so each stem is resampled to the master's rate and written to
`stems/<track_id>/<stem>.aiff`, and the stems' sum is proven to line up with
the master to the sample before any loop is cut from them. Raw output is
cached, so a second run is instant.

### `cut`

```bash
loopcutter cut MANIFEST [options]
```

| Option | Default | What it does |
|---|---|---|
| `--out DIR` | `loops/<format>` in a workspace, else `loops` | Output directory. |
| `--layout` | `library` in a workspace, else `flat` | `library` files loops as `<stem>/<tempo band>/`. |
| `--tag`, `--no-tag` | on for AIFF in a workspace | Writes BPM, key and stem tags. |
| `--tracks FILE` | the workspace's `tracks.csv` | Analysis used for the tempo check. |
| `--audio-root DIR` | current directory | Base for relative `source` paths. |
| `--format` | `aiff` | `aiff`, `wav` or `flac`. |
| `--subtype` | `PCM_24` | Any libsndfile subtype. |
| `--snap-ms` | `2.0` | Zero-crossing search radius (backwards only). |
| `--fade-ms` | `0.5` | Edge fade length. |
| `--trim-db` | `0.0` | Gain for every loop, for example `-1.0`. |
| `--xfade-ms` | `0.0` | Crossfades each loop's tail into the audio before its start (a row's `xfade_ms` wins). |
| `--check-bpm` | off | Measures the tempo from the audio. Automatic for analysed tracks. |
| `--lenient` | off | Reports `beat_alignment` failures as warnings, for unsteady material. |
| `--dry-run` | off | Resolves every row without writing audio. |

32-bit float isn't the default because many CDJs won't play it.

### `inspect`

```bash
loopcutter inspect FILE [--bpm N]
```

Prints sample rate, channels, format, length, and the length in beats and bars
at the given tempo.

### `keys`

See [Session key planning](#session-key-planning).

## Output files

Files are named `Artist - Track [label][bpm][bars][key][stem].aiff`, for
example `Some Artist - Some Track [A1][128][4bar][8A][bass].aiff`. The brackets
make a library easy to sort and search in any browser. In a workspace they are
filed by stem and tempo band, such as `loops/aiff/bass/125-129/`; loops without
a stem go under `full/`, and one-shots under `<stem>/oneshots/`, named
`Artist - Track [label][oneshot].aiff`.

A row that names a `stem` and a `track_id` is cut from that track's separated
stem, while its timing is still checked on the full mix, where the attacks
are. `cut` names any stem that hasn't been separated yet. A `stem` row without
a `track_id` is cut from its `source` as given, with a note, so point `source`
at the stem file if you separated it yourself.

AIFF loops are tagged with BPM, key, stem (as the grouping), title, artist and
a comment holding the Camelot key and label. WAV loops are left untagged by
default, because some hardware samplers reject the tag chunk; pass `--tag` to
tag them anyway.

Ableton Live doesn't read the BPM in a filename. It assumes a clip is 1, 2, 4,
8 or 16 bars long and sets the clip's tempo from its length, so loops of those
lengths land on tempo because the lengths are exact. For any other length,
such as ½-bar rolls or 3-bar loops, `cut` prints a note: set that clip's tempo
by hand.

To make two copies of the same cuts, run the manifest twice: once as AIFF for
your DAW, and once with `--format wav` for a hardware sampler.

### Roll variations

A `variations` cell such as `"2,1,0.5"` turns one row into several loops that
share a start. Put them in one clip-launcher column, for example Ableton's
Session view played from a Launchpad. Set the clips to Legato launch so
switching keeps phase, and pressing down the column gives a roll that stays
locked. Each one is a real file cut to an exact sample count.

### Sample rates and tiling

Sample counts are whole numbers, so a sub-bar variation doesn't always divide
its parent exactly. Whether it does depends on the tempo as well as the rate:

| Rate | Common tempos where half bars tile exactly |
|---|---|
| 48 kHz | 120, 125, 128, 144, 150, 160 BPM |
| 44.1 kHz | 120, 125, 126, 135, 140, 144, 147, 150, 160 BPM |

Neither rate is always better. At other tempos a variation drifts by a few
samples per cycle: up to 4 for a half bar against 4 bars, and up to 8 against
8 or 16 bars, under 0.2 ms. `cut` reports each drifting variation set
and, when the other rate would be exact, names it. DAWs that warp clips
re-sync them every cycle anyway. loopcutter never resamples at cut time, so
the rate is chosen once, when `prep` makes the masters.

## Verification

An agent can't hear the output, and neither can a batch run, so every file is
checked before a run reports success. Every check has a test proving it can
fail.

- **`sample_count`, `sample_rate`:** the file is exactly the declared length at
  the source's rate.
- **`peak_headroom`:** the loop is no louder than its source, after any
  `--trim-db`. A master that already peaks at full scale passes; a trim that
  would push a loop past full scale fails.
- **`not_silent`, `dc_offset`.**
- **`beat_alignment`:** finds the strongest attack within 60 ms of each of the
  loop's beats, in the source around the loop (the full mix, for a stem).
  - *Placement* is where those attacks sit. It passes from −3 to +8 ms, warns
    out to −10 or +15 ms, and fails beyond. An attack before the start means
    the start cuts into it.
  - *Drift* is how the attacks move from the loop's first beat to its last,
    which is what a wrong tempo looks like. It warns above 4 ms and fails
    above 10 ms.
  - When under half the beats share a steady attack (breaks, pads, swung
    material), it reports *not judged* rather than guessing.
  - `--lenient` turns its failures into warnings.
- One-shots have no beats to align and no tempo to match, so they get the
  length, rate, headroom, silence and DC checks only.
- **`bpm_match`:** how far the loop's end lands from where the next bar
  begins. Against the tempo `scan` proposed it fails above 2 ms. On a
  `grid-fit` track, and with `--check-bpm` on an unanalysed one, it measures
  the tempo from the audio instead and fails above 5 ms; on a `grid-fit`
  track a half- or double-time tempo fails outright.

A failed check names the file and the reason, and the run exits with an error.

## Session key planning

Key detection is unreliable on short loops. Analyse the full source tracks
instead: `scan` reads the key your DJ software or Mixed In Key wrote to each
master, cross-checks it with keyfinder-cli, and carries it into the manifest's
`key` column. Every slice inherits its parent's key.

Then work out what to build the set around:

```bash
loopcutter keys library.xlsx --histogram
loopcutter keys library.xlsx --keys 3 --semitones 3 -v
loopcutter keys library.xlsx --sweep
```

`keys` reads CSV, XLSX or a manifest, using a column called `key`, `camelot`
or `initial key`, so exports from Mixed In Key, Serato and rekordbox work as
they are. It accepts Camelot (`8A`) and named keys (`Am`, `F# minor`, `Bb`).
It rejects Open Key rather than misreading it, because Open Key numbers sit
five places away from Camelot's.

The model is transposition, not harmonic mixing: a loop is usable in a session
key if you can pitch it there within the limit. It picks the key that covers
the most loops, then the best key for what's left, and reports the exact
semitone shift for each loop.

| Flag | Default | What it does |
|---|---|---|
| `--keys N` | 3 | How many session keys to plan for. |
| `--semitones N` | 3 | Transposition limit either way, 0 to 6. |
| `--comfort N` | 3 | Shifts beyond this are flagged for audition in the verbose listing. |
| `--sweep` | off | Tabulates coverage at every limit from 0 to 6. |
| `--relative` | off | Counts relative major/minor as a free match. |
| `--histogram` | off | Prints the key distribution as a bar chart. |
| `--sheet NAME` | first sheet | Worksheet to read in a multi-sheet XLSX. |
| `-v`, `--verbose` | off | Lists every loop with its Camelot key and shift. |

Read the sweep's "keys to cover all" column first. Relaxing the limit by one
semitone can drop you from three session keys to two, which changes how a set
is built far more than a few percent of coverage does. The best session key
is often not the most common one: a key between two clusters can reach more
material than the biggest cluster.

## Stems and seamless loops

One 4-bar window across six stems gives six loops you can EQ and launch apart:
run `loopcutter stems "<track_id>"` for the track, then give the rows a
`stem`. Comment a rekordbox memory loop `A1 bass` and `import` fills it in.

Edge fades stop a loop clicking, but on a sustained pad they dip to silence
at every wrap. `--xfade-ms 20` (or a row's `xfade_ms`) instead fades the
loop's tail into the audio just before its start, so the wrap plays exactly
what the source played there. The length doesn't change, and because the two
sides are similar audio, the linear blend never exceeds the source's peak.

## Known issues

- **rekordbox's database format changes between versions.** If
  `import --from rekordbox` can't read yours, export the collection as XML
  from rekordbox and use `--from rekordbox-xml --xml FILE`.
- **Tempo-changing tracks get local grids.** A track whose beats don't fit one
  steady grid is flagged `grid-fit`. Each mark is snapped to the whole-track
  grid where that still fits the nearby beats, and otherwise to a grid fitted
  to the 64 beats around it, provided that grid fits clearly better and its
  tempo isn't a half- or double-time reading. A mark where neither holds is
  refused. On a tempo-changing track the `tracks.csv` tempo is only an
  average, so `cut` checks those loops against the tempo measured from the
  audio.
- **Odd bar lengths need their tempo set in Live by hand** (see
  [Output files](#output-files)).
- **Starts in beatless passages can't be judged.** `beat_alignment` reports
  *not judged* there, so a hand-typed start in a breakdown relies on the grid
  alone.

## Roadmap

- An audition page: each master's waveform with the cutter's own bar lines,
  where a click previews the exact loop and a key accepts it into a manifest.
- Ableton integration: the right tempo for odd-length and half-bar clips, and
  a generated Live Set with each loop family in one Launchpad column.

## Development

```bash
pytest -q            # unit tests: synthetic audio only, no model weights
pytest -q -m slow    # the real beat detector and ffmpeg
```

```
src/loopcutter/
  model.py      shared data types: Grid, TrackRecord, Marker, SnapResult
  onsets.py     the one onset detector that the grid, the analysis and the checks share
  timing.py     sample arithmetic, no file I/O
  workspace.py  finding and creating workspaces
  audio_io.py   reading any audio file (libsndfile, else ffmpeg)
  prep.py       lossless working masters
  grid.py       beat-grid fit, bar phase, attack phase, local grids
  keydetect.py  key from tags, cross-checked with keyfinder-cli
  trackdb.py    tracks.csv
  analyse.py    scan: one analysis per master
  markers.py    reading marks and grids from rekordbox and Serato
  snap.py       app-offset correction, snapping, marks to manifest rows
  manifest.py   CSV to LoopSpec; all validation lives here
  cut.py        extraction, zero-crossing snap, edge fades
  verify.py     objective checks on every output
  tags.py       ID3v2.3 tags on AIFF and WAV loops
  naming.py     output filename convention and library layout
  keys.py       key parsing and session-cover arithmetic, no file I/O
  keyreport.py  spreadsheet loading and report rendering
  stems.py      separation, resampling to the master, alignment proof
  cli.py        command-line entry point
```

The code keeps a few rules:

- Loop length always comes from tempo and bar count, rounded once.
  `tests/test_timing.py` guards this against a well-meant "simplification".
- Analysis proposes and `cut` disposes: `scan` writes `tracks.csv`, only
  `import` and `resolve` move a start (reporting every move), and `cut` never
  moves a start or runs a model.
- Shared types go in `model.py`, and every part that looks for attacks uses
  `onsets.py`, so the grid and the checks can't disagree about where a beat is.
- `timing.py` and `keys.py` do arithmetic only, so they stay testable without
  audio fixtures.
- Failures are loud and name the manifest row. The tool never writes a short,
  silent or wrong-length file.
- Nothing is fixed silently: no trimming to fit, no resampling at cut time, no
  padding.

## Licence

MIT. See [LICENSE](LICENSE).

Loops cut from commercial releases are for your own practice and performance.
Releasing anything built from them needs the rights holders' clearance.
