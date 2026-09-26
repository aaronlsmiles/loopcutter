# From spreadsheet to loop library

A working order that avoids the common way this goes wrong: building a large
manifest before proving the pipeline on three rows.

## 0. Source and working formats

**Never upconvert MP3 to AIFF.** Lossy compression is irreversible—you get a
larger file containing the same missing data.

**The reason to source lossless isn't quality, it's timing.** MP3 encoders add
silence at the start of the file, typically 26ms or 50ms, and different
programs handle that padding differently. A time read off an MP3 in your DJ
software can be tens of milliseconds away from the same moment in the audio
this tool decodes, which shifts every start in your manifest. Slice from
lossless, and read your times from the same file you cut.

**Pick one working rate, 24-bit.** The tool never resamples, so convert
sources first. 48 kHz is the native rate of samplers such as Elektron's; 44.1
kHz is fine too. Whether sub-bar variations tile exactly depends on tempo:
see the README's section on sample rates. 24-bit for processing headroom.
Never 32-bit float, which many CDJs won't play.

**Two exports from one manifest**: run it to AIFF for your DAW, then again
with `--format wav` for a hardware sampler. Identical cuts, identical sample
counts, two destinations.

## 1. Pick the records and log the cues

Play the track, find the passage, note the timecode and how many bars you
want. One row per loop. Resist the urge to be clever about how many loops you
"should" take from a record—Hawtin's Parts LP pressed six locked grooves
from a track he used once in the mix.

Record BPM from your DJ software's analysis, but check the beatgrid by ear
first. A wrong declared BPM produces a loop of the wrong length that still
passes every check, because the tool believes you.

## 2. Prove the pipeline on three rows

```bash
loopcutter cut manifests/probe.csv --out loops --dry-run
loopcutter cut manifests/probe.csv --out loops
loopcutter inspect "loops/<one of them>" --bpm <declared>
```

The bar count must come back whole. Then load those three into your DAW or
DJ software, set them looping, and listen for a full minute. Drift and seam
clicks only show up on repeat.

## 2b. Add roll variations where you want them

For any loop you expect to stutter or ratchet live, add a `variations` cell:

    variations
    "2,1,0.5"

Put the resulting files in one clip-launcher column (Ableton's Session view
played from a Launchpad, for example), set the clips to Legato launch so
switching keeps phase, and pressing down the column gives you a roll that
stays locked. Do not do this for every loop - it costs grid space, and most
loops never need it.

Read the tiling note the tool prints for each variation set. See the README
on sample rates.

## 3. Separate stems where it earns its place

Not every loop needs it. Stems are worth it when you want the bassline without
the hats, or the pad without the kick—which for this style is most of the
time, but not all of it. Percussion loops are usually better taken full-range.

## 4. Scale up

Build the full manifest, run it, fix what the verifier names, re-run. The
whole output directory is disposable; only the manifest matters.

## 5. Tag and file

The filename carries label, BPM, bar count, key and stem, so a library sorts
and searches by eye in any browser. Your DAW takes tempo from the audio, not
the name: Ableton Live assumes a clip is 1, 2, 4, 8 or 16 bars long and sets
its tempo from the length, so exact lengths are what make loops land on
tempo. Beyond that, sort by BPM band rather than by source record—when you
are assembling, you reach for "a 4-bar bass loop at 128" far more often than
for a particular artist.

## Common failures

| Symptom | Cause |
|---|---|
| Loop drifts against the grid over a minute | declared BPM is slightly wrong, or the source has live timing |
| Click on every wrap | seam check should have caught it; raise `--fade-ms` or move the start |
| `too short` error | start is near the end of the track, or bars is too large |
| Loop is silent | cue landed in a breakdown; check the timecode |
| Two rows overwrite each other | same artist/track/label—the loader now rejects this upfront |
