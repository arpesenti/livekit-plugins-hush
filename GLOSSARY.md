# LiveKit Hush plugin

A streaming speech enhancer that plugs into LiveKit's audio input as a `FrameProcessor`.
This glossary fixes the words used for audio timing and for the two ways audio can come out
untouched — the same words were previously used for both a bug and a feature.

## Language

### Stream boundaries

**Frame**:
The audio buffer the host hands the processor, and expects exactly one of back for. Any sample
rate, any length, one `_process` call each.
_Avoid_: chunk, block, buffer

**Hop**:
The 160-sample unit (10 ms at 16 kHz) that the model consumes and produces, one hop per
inference call.
_Avoid_: frame, window, chunk

**Native rate**:
The host's frame sample rate — 8/16/24/32/44.1/48 kHz. Output is delivered at the same rate as
input, so the stage returns to it after the model. 16 kHz is also the model rate and needs no
converter.
_Avoid_: host rate, output rate

**Model rate**:
16 kHz — the only rate the model runs at.
_Avoid_: native rate (when 16 kHz specifically is meant), network rate

### Timing

**Converter delay**:
The delay a streaming resampler declares: half its kernel, a constant known when it is built. It
replaced the old per-rate backlog probe, which measured how many samples a bursty converter held
back.
_Avoid_: backlog, burst, measured delay

**Algorithmic latency**:
The delay inherent to the model: 10 ms (one hop) of STFT lookahead on top of fully causal state.
_Avoid_: backlog, processing time

**Added delay**:
The total delay the processor inserts beyond the algorithm: the converters' declared delay plus
at most one hop of hop/frame quantisation. It is not quantised to whole frames.
_Avoid_: latency

**Prime**:
Silence placed at the head of a stream to pay the converters' declared delay. It is emitted
inside a frame — silence first, then the audio that is ready — not in whole frames.
_Avoid_: prefill, padding, warmup, backlog

**Underrun**:
A shortfall larger than one hop after the stream has been primed, meaning the enhanced path is
not keeping up. A smaller shortfall is the expected hop quantisation and is counted as prime.

### Modes

**Leak**:
Audio that reached the consumer without passing through the model, although the processor was
on and the model was supposed to shape it.
_Avoid_: bypass, passthrough, fallback

**Bypass**:
A configured-off state (`enabled=False`, `strength=0`) in which the processor returns frames
untouched — intentional, cheap, and not a defect.
_Avoid_: leak, passthrough

### Signal

**Wet / dry**:
Wet is the model's output for a hop; dry is that same hop's input. They are mixed, never switched.

**Strength**:
The wet fraction of the output: 0.0 is dry only (bypass), 1.0 is wet only.

**Attenuation limit** (`atten_lim_db`):
A cap on how much of the input spectrum the model may remove. 0 dB allows no removal, so the
output is the input again — which is why "output must never equal input" is not an invariant.
_Avoid_: gain, suppression level

**Enhanced path**:
Every sample that has passed through the model, the wet/dry blend and the rate conversion, as
opposed to the untouched input.
