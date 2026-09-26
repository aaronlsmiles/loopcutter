# loopcutter

Cut sample-accurate loops out of full tracks, driven by a plain-text manifest.

loopcutter is for live sets built from many short loops played at once, in the
spirit of Richie Hawtin's DE9 mixes. When eight loops run together for a
minute, a loop that is three samples too long drifts audibly. loopcutter
derives every loop's length from its tempo and bar count and rounds once. It
nudges the start onto a zero crossing and checks every file before it calls a
run a success.

The manifest is the source of truth. Keep it in version control and every
loop on disk can be rebuilt from one CSV file.

## Features

- **Exact lengths.** `samples = round(bars × beats_per_bar × sample_rate × 60 / bpm)`,
  rounded once at the end, never taken from a second timecode.
- **Click-free edges.** The start snaps to the nearest zero crossing within
  2 ms. The end moves with it, so the length never changes. A 0.5 ms fade at
  each edge catches whatever the snap missed.
- **Roll variations.** One manifest row can produce 4, 2, 1 and ½-bar versions
  from the same start, ready to stack in a clip launcher for beat-division rolls.
- **Verification.** Every file is checked for exact sample count, sample rate,
  headroom, silence, DC offset and a head-to-tail seam match.
- **Session key planning.** From a list of keys, works out which one to three
  session keys cover the most loops within a transposition limit, and the
  exact shift each loop needs.
- **Stems (optional).** A Demucs pre-pass, so one window of a track can give
  separate drum, bass and melodic loops.

## Status

Version 0.1. It works, and its 38 tests pass. It trusts the start times and
tempo you give it: it doesn't find the beat itself yet. Read
[Getting the start right](#getting-the-start-right) before cutting a large
batch, and see [Known issues](#known-issues) and [Roadmap](#roadmap).

## Install

Needs Python 3.10 or later.

```bash
git clone https://github.com/aaronlsmiles/loopcutter.git
cd loopcutter
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Optional extras:

| Extra | Adds | For |
|---|---|---|
| `verify` | librosa | the `--check-bpm` tempo cross-check |
| `sheets` | openpyxl | reading XLSX key lists |
| `stems` | Demucs | the stem-separation pre-pass |

```bash
pip install -e ".[dev,verify,sheets]"
```

Audio is read with libsndfile, which ships inside the `soundfile` package and
handles WAV, AIFF, FLAC and MP3. AAC and M4A files aren't supported, so
convert them first, for example `ffmpeg -i in.m4a out.wav`.

## Quick start

1. Write a manifest, one row per loop. [manifests/example.csv](manifests/example.csv)
   shows every column.

   ```csv
   source,label,artist,track,bars,bpm,start,variations,key
   audio/track.aiff,A1,Some Artist,Some Track,4,128,1:04.187,"2,1,0.5",Am
   audio/track.aiff,A2,Some Artist,Some Track,2,128,2:11.500,,Am
   ```

2. Check that every row resolves, then cut:

   ```bash
   loopcutter cut manifests/session.csv --out loops --dry-run
   loopcutter cut manifests/session.csv --out loops
   ```

3. Spot-check an output. The bar count must come back whole:

   ```bash
   loopcutter inspect "loops/Some Artist - Some Track [A1][128][4bar][Am].aiff" --bpm 128
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
| `source` | yes | Path to the full track. |
| `label` | yes | Your cue or ID reference, such as `A1` or `16.2`. |
| `bars` | yes | Loop length in bars. |
| `bpm` | yes | The track's tempo. Use the analysed value to two decimal places. |
| `start` | one of | `M:SS.mmm`, `H:MM:SS.mmm` or bare seconds. |
| `downbeat` + `start_bar` | one of | A precise downbeat time plus a 1-based bar number counted from it. |
| `variations` | no | Extra bar lengths from the same start, such as `"2,1,0.5"`. |
| `beats_per_bar` | no | Defaults to 4. |
| `artist`, `track` | no | Used to build the output filename. |
| `stem` | no | A tag for the filename. Run separation separately. |
| `key` | no | Written into the filename, and read by the key planner. |
| `notes` | no | Ignored by the tool. For you. |

Give either `start` or `downbeat` plus `start_bar`, never both. The loader
rejects rows that give both, rows with no start, and manifests whose rows
would produce the same filename, and it names the row number every time.
Relative `source` paths resolve against the current directory. Use
`--audio-root` to point somewhere else.

## Getting the start right

loopcutter cuts exactly where you tell it to. The zero-crossing snap moves a
start by 2 ms at most. That prevents clicks, but it can't rescue a start that
is off the beat.

- **Give milliseconds.** A time typed to the whole second lands anywhere up to
  half a second from the downbeat, which at 125 BPM is the wrong beat about
  half the time.
- **Prefer the grid form.** Read one downbeat precisely from a zoomed waveform,
  then address each loop by bar: `downbeat` `0:12.041`, `start_bar` `33`.
- **Get the tempo exact.** A 4-bar loop cut at 134 BPM from a 136 BPM track is
  105 ms too long on every repeat. Integer tempos are usually right for
  machine-made tracks. Check the grid by ear on anything played live or taken
  from vinyl.
- **Cut from lossless files.** MP3s begin with encoder padding that programs
  handle differently. A time read off an MP3 in DJ software can be tens of
  milliseconds from the same moment in the audio this tool decodes. Cut from
  WAV, AIFF or FLAC, and read your times from the same file you cut.

## Commands

### `cut`

```bash
loopcutter cut MANIFEST [--out DIR] [options]
```

| Option | Default | What it does |
|---|---|---|
| `--out DIR` | `loops` | Output directory. |
| `--audio-root DIR` | current directory | Base for relative `source` paths. |
| `--format` | `aiff` | `aiff`, `wav` or `flac`. |
| `--subtype` | `PCM_24` | Any libsndfile subtype. |
| `--snap-ms` | `2.0` | Zero-crossing search radius. |
| `--fade-ms` | `0.5` | Edge fade length. |
| `--check-bpm` | off | Adds a librosa tempo cross-check (slower). |
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
example `Some Artist - Some Track [A1][128][4bar][Am].aiff`. The brackets make
a library easy to sort and search in any browser.

Your DAW takes tempo from the audio, not the name. Ableton Live assumes a clip
is 1, 2, 4, 8 or 16 bars long and sets its tempo from the length, so loops of
those lengths land on tempo because the lengths are exact. Set other lengths,
such as ½-bar rolls or 3-bar loops, by hand.

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
its parent exactly. Whether it does depends on tempo and sample rate:

| Rate | Common tempos where half bars tile exactly |
|---|---|
| 48 kHz | 120, 125, 128, 144, 150 BPM |
| 44.1 kHz | 120, 125, 126, 135, 140, 144, 150 BPM |

At other tempos a variation drifts by 1 to 3 samples per cycle, under 0.07 ms.
The tool reports it for each variation set. DAWs that warp clips re-sync them
every cycle anyway. loopcutter never resamples, so pick a working rate and
convert your sources before cutting.

## Verification

Every output is checked before a run reports success:

- exact sample count;
- sample rate matches the source;
- peak headroom;
- not silent;
- DC offset;
- seam continuity, which compares the spectrum of the first and last 20 ms.

`--check-bpm` adds a librosa tempo estimate that allows for half- and
double-time detection. A failed check names the file and the reason, and the
run exits with an error.

## Session key planning

Key detection is unreliable on short loops. Analyse the full source tracks
instead, in Mixed In Key or your DJ software, and carry each key into the
manifest's `key` column. Every slice inherits its parent's key.

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

## Stems

```python
from loopcutter.stems import separate

stems = separate("audio/track.aiff", "stems/")   # {"drums": Path, "bass": Path, ...}
```

This calls Demucs (`htdemucs_6s` by default, six stems) and skips tracks
already separated. Point manifest rows at the stem files and tag them with
`stem`. One 4-bar window across six stems gives six loops you can EQ apart.
The original Demucs repository is archived; its author maintains a fork at
[adefossez/demucs](https://github.com/adefossez/demucs).

## Known issues

- **`--check-bpm` can't currently fail.** Its pass condition includes a
  comparison that is always true, and its 2% tolerance would let a 2 BPM error
  through anyway.
- **The headroom check fails loud masters.** Loops cut from a master that
  already peaks at full scale fail, even though the cut adds no gain.
- **The tiling note overpromises.** It says "48 kHz sources avoid this" even at
  tempos where 48 kHz doesn't tile (see the table above).
- **The seam check can't see placement.** It compares the first and last 20 ms,
  so a loop that starts off the beat can still pass.

## Roadmap

- Beat-grid detection with a per-track phase correction, so each start snaps
  to the nearest beat and the tool reports how far it moved.
- Importing loop markers from DJ software (rekordbox, Serato), so you mark
  loops by ear on a waveform instead of typing times.
- A one-shot mode for phrases and stabs, BPM, key and stem tags written into
  output files, and a prep step that makes lossless working copies at one
  sample rate.

## Development

```bash
pytest -q
```

```
src/loopcutter/
  timing.py     sample arithmetic, no file I/O
  manifest.py   CSV to LoopSpec; all validation lives here
  cut.py        extraction, zero-crossing snap, edge fades
  verify.py     objective checks on every output
  naming.py     output filename convention
  keys.py       key parsing and session-cover arithmetic, no file I/O
  keyreport.py  spreadsheet loading and report rendering
  stems.py      optional Demucs pre-pass
  cli.py        command-line entry point
```

The code keeps a few rules:

- Loop length always comes from tempo and bar count, rounded once.
  `tests/test_timing.py` guards this against a well-meant "simplification".
- `timing.py` and `keys.py` do arithmetic only, so they stay testable without
  audio fixtures.
- Failures are loud and name the manifest row. The tool never writes a short,
  silent or wrong-length file.
- Nothing is fixed silently: no trimming to fit, no resampling, no padding.

## Licence

MIT. See [LICENSE](LICENSE).

Loops cut from commercial releases are for your own practice and performance.
Releasing anything built from them needs the rights holders' clearance.
