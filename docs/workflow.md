# From record box to loop library

A working order that avoids the common way this goes wrong: building a large
manifest before proving the pipeline on three rows.

## 0. Sources and masters

**Source lossless where you can.** Converting an MP3 to AIFF gains no
quality: lossy compression is irreversible, and you get a larger file holding
the same missing data. Buy lossless copies of anything tonal you mean to loop.

**Cut from masters, not from your sources.** MP3 encoders add silence at the
start of the file, and different programs handle that padding differently, so
a time read off an MP3 in DJ software can be tens of milliseconds from the
same moment in the audio this tool decodes. `loopcutter prep` decodes each
source once and writes a 24-bit master at one working rate (48 kHz by
default, the native rate of samplers such as Elektron's). Every app then reads
identical samples. Masters aren't overwritten, so tags your key or DJ
software writes to them are safe; when a lossless copy replaces a lossy
original, the old master is set aside and its tags carried over.

**Never 32-bit float** for anything that goes near a CDJ.

**Two exports from one manifest**: run it to AIFF for your DAW, then again
with `--format wav` for a hardware sampler. Identical cuts, identical sample
counts, two destinations.

## 1. Analyse

`loopcutter scan --app rekordbox` (or `--app serato`) finds each master's
beat grid, bar phase and key, and compares the grid with your DJ app's. Read
the flags in its report before marking anything: `grid-fit` means the tempo
moves or the track has long breaks, `half-beat` means your app's grid sits
half a beat from the audio's, and `phase` means the attacks don't agree on
where the beat is. Fix what you can in the app, or set `override_bpm` and
`override_phase_ms` in `tracks.csv`.

## 2. Mark the loops by ear

Load the masters into rekordbox as one playlist and set memory loops with
quantise on, on the passages you want. Comment each one with its label, then
its stem, then any roll lengths: `A1 bass 2,1`. Resist the urge to be clever
about how many loops you "should" take from a record: Hawtin's Parts LP
pressed six locked grooves from a track he used once in the mix.

Then `loopcutter import --from rekordbox --playlist NAME`. Every mark is
snapped onto the grid and the move is reported; marks that can't be placed
are skipped with the reason.

## 3. Prove the pipeline on three rows

```bash
loopcutter cut manifests/probe.csv --dry-run
loopcutter cut manifests/probe.csv
loopcutter inspect "loops/aiff/<one of them>" --bpm <declared>
```

The bar count must come back whole. Then load those three into your DAW or
DJ software, set them looping, and listen for a full minute. Drift and seam
clicks only show up on repeat, and the checks are a substitute for your ears,
not a replacement.

## 3b. Add roll variations where you want them

For any loop you expect to stutter or ratchet live, add roll lengths to the
comment (`A1 2,1,0.5`) or a `variations` cell:

    variations
    "2,1,0.5"

Put the resulting files in one clip-launcher column (Ableton's Session view
played from a Launchpad, for example), set the clips to Legato launch so
switching keeps phase, and pressing down the column gives you a roll that
stays locked. Do not do this for every loop - it costs grid space, and most
loops never need it.

Read the tiling note the tool prints for each variation set. See the README
on sample rates.

## 4. Separate stems where it earns its place

Not every loop needs it. Stems are worth it when you want the bassline without
the hats, or the pad without the kick, which for this style is most of the
time, but not all of it. Percussion loops are usually better taken
full-range. Run `loopcutter stems "<track_id>"` for the tracks that need it;
a row that names a stem is then cut from it.

## 5. Scale up

Import the whole playlist, cut it, fix what the checks name, re-run. The
whole output directory is disposable; only the manifest matters.

## 6. Browse

Loops are filed by stem and tempo band and tagged with BPM, key and stem, so a
sample browser can sort and search them. Your DAW takes tempo from the audio,
not the name: Ableton Live assumes a clip is 1, 2, 4, 8 or 16 bars long and
sets its tempo from the length, so exact lengths are what make loops land on
tempo. When you are assembling, you reach for "a 4-bar bass loop at 128" far
more often than for a particular artist.

## Common failures

| Symptom | Cause |
|---|---|
| `import` skips a mark | the reason is printed: too far from a beat, before the file, an unsure grid phase, or a half-time reading |
| `beat_alignment` fails | the start is off the attack, or the tempo is wrong; check the mark, or set an override in `tracks.csv` |
| `bpm_match` fails | the declared tempo disagrees with the analysis or the audio |
| Loop drifts against the grid over a minute | the tempo is slightly wrong, or the source has live timing |
| Click on every wrap | raise `--fade-ms`, or use `--xfade-ms` on a sustained sound |
| `too short` error | the start is near the end of the track, or bars is too large |
| Loop is silent | the mark landed in a breakdown |
| Two rows overwrite each other | same artist, track, label and stem; the loader rejects this upfront |
