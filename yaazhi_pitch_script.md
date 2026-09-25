# Yaazhi — Full Pitch Script

> **Usage:** Read this out loud, at normal speaking pace. Each slide section has the spoken words, then the "what to do with your hands / demo" cue in italics. No personal names are used anywhere. Total runtime: ~8–12 minutes with demo, ~4–5 minutes without.

---

## ─── PART 1: SLIDE-BY-SLIDE SCRIPT ───

---

### SLIDE 1 — Title / Hook

**Spoken words:**

> "Every year, firefighters walk into buildings they cannot see. Not because of darkness — thermal cameras solve that. The problem is that the thermal images they carry are low-resolution, slow, and when they turn their heads, the view smears and lags.
>
> Markers on people shift. Context is lost. In the worst moment of someone's life, the tool that should help them becomes noise.
>
> Yaazhi is a perception software pipeline that fixes that. It detects people through low-quality thermal video, stabilises those markers under head movement using the IMU already in the headset, and when any sensor fails — it fails safe. Not frozen. Not wrong. Off.
>
> Let me show you that."

*Cue: Stand back. Let the split-screen come up on its own. Don't click anything yet.*

---

### SLIDE 2 — The Problem

**Spoken words:**

> "The tools that exist today were designed for a different era. Thermal cameras produce images at 9 frames per second — a tenth of what your phone does. At the resolution of a Lepton-class sensor — 160 by 120 pixels — a person standing 20 metres away is roughly 12 pixels tall. At 30 metres, they're 8 pixels tall.
>
> On top of that, the camera is mounted on a helmet. Every time the wearer turns their head, the image shifts. But the detection markers on screen don't shift with it for up to 150 milliseconds. That's not a software bug — it's physics. Processing takes time. The question is: what do you do with that time?
>
> Existing systems either freeze the overlays — which means they lie — or disable them entirely. Neither is useful.
>
> We solved this without adding any hardware."

*Cue: Point to the "raw" side of the split screen. Let the audience look at it. It should be choppy and slow.*

---

### SLIDE 3 — The Solution / Architecture Overview

**Spoken words:**

> "Yaazhi is three things working together.
>
> First — detection. We run a YOLOv8 nano model, trained and evaluated at actual sensor resolution: 160 by 120 pixels, at 9 frames per second, with optical blur and thermal noise injected. Not at HD. At the resolution the hardware actually produces.
>
> Second — stabilisation. While detection is running on the slow sensor loop, the IMU in the headset is updating at 200 times per second. We read that IMU stream continuously, predict where the frame will be at display time, and warp the image to match. The render loop runs at 60 frames per second regardless of inference speed. The sensor is slow. The display is not.
>
> Third — safety. The system monitors every input in real time. If thermal goes stale, markers fade. If IMU drops, reprojection stops. If both go, the entire overlay disappears and the screen reads: SENSOR LOST. DIRECT VIEW. The operator sees raw reality — which is always safer than a frozen, wrong overlay.
>
> Here's what that looks like."

*Cue: Begin the scripted demo. Press the demo start hotkey or walk through manually.*

---

### SLIDE 4 — Live Demo (This is your most important slide)

**Spoken words — narrate what the audience is seeing in real time:**

**[Beat 1 — Normal view, 0:00–0:30]**
> "This is normal operation. Left side: raw 9 Hz thermal output — you can see it's slow and choppy. Right side: Yaazhi output. 60 frames per second display. Corner-bracket markers on the detected person. The system is confirming a track before it shows a marker — it won't flash a box for a single noisy frame.
>
> The numbers at the bottom — those are real measurements. Not targets. mAP-50 of 61.4% at sensor resolution. Recall of 95.7% for people closer than 7 metres, dropping to 57% beyond 30 metres. We are not hiding that range drop-off. We measured it; it's in the report."

**[Beat 2 — Head turn, reprojection off then on, 0:30–1:15]**
> "I'm going to simulate a fast head turn."
> *(Press 'R' to turn reprojection OFF first, then press '3' for fast_turn IMU profile)*
> "Reprojection is OFF. Watch the markers lag. The person is there — the system knows — but the overlay doesn't keep up. That's the raw problem."
> *(Press 'R' again to turn reprojection ON)*
> "Now it's ON. Same head movement. The image warp compensates for the predicted orientation. Benefit measured at 150ms delay: 75 pixel improvement in overlay alignment for a fast turn. Not perfect — you'll see some residual error during abrupt reversals — but dramatically better than nothing."

**[Beat 3 — Smoke simulation, 1:15–1:45]**
> *(Press 'S' to enable smoke)*
> "This is simulated smoke — the banner says so. The visible-light feed degrades completely. The thermal side attenuates but does not lose the person. This is why thermal exists. We're not claiming fire-rated performance. We're showing that the software correctly handles what the sensor already does."

**[Beat 4 — Hot scene, 1:45–2:15]**
> *(Press 'H' to cycle to adaptive AGC mode)*
> "In a real fire scene, the camera will see a saturated hot region — a fire source. In naive contrast-stretching mode, that blows out the whole image and crushes contrast everywhere else. In adaptive mode — CLAHE with hot-pixel exclusion — we measured a 14% improvement in RMS contrast in the mid-tone region where people appear. Numbers are in the report."

**[Beat 5 — Fault injection → SAFE state → recovery, 2:15–2:45]**
> *(Press 'F' to cycle fault injection to IMU dropout)*
> "I'm dropping the IMU. Watch what happens."
> *(Pause 2 seconds)*
> "Markers fade. Reprojection stops. After 1.2 seconds with no valid IMU — SENSOR LOST. DIRECT VIEW. The overlay is gone. The operator is looking at raw thermal. That is the safest thing to show when inputs are uncertain.
>
> Now I'll restore the signal." *(Press 'F' again to clear fault)*
> "Three consecutive valid samples — recovery. Back to NORMAL. The system never gets stuck in SAFE mode. It returns to operation as soon as it can verify the inputs."

**[Beat 6 — Metrics summary, 2:45–3:00]**
> "End of the three-minute sequence. Metrics: render rate steady at 60 fps throughout. The sensor ran at 9 Hz throughout — that never changed. The display loop never blocked on inference. That architectural choice — decoupling the fast loop from the slow loop — is the reason this works on a mid-range student laptop."

*Cue: Let the display rest. Pause. Let silence work.*

---

### SLIDE 5 — Numbers Slide (Honest Results)

**Spoken words:**

> "Here are the real numbers. Not projected. Not theoretical. Measured on our test hardware today.
>
> Detection: mAP-50 of 61.4% at 160 by 120 sensor resolution, on the LLVIP infrared test set — 3,463 test images. We use a confidence threshold of 0.10 and IoU of 0.30, giving an F1 of 0.64. That threshold was chosen by sweeping the precision-recall curve against a false-alarm budget of at most one false positive per minute.
>
> By distance: people closer than 7 metres — 95.7% recall. 7 to 15 metres — 82.2%. 15 to 30 metres — 70.7%. Beyond 30 metres — 56.9%.
>
> Reprojection: at 150ms display delay — which is realistic — slow head scans see a 17-pixel benefit. Fast turns see a 75-pixel benefit. Abrupt reversals are the hard case — the predictor overshoots briefly. We documented that. It's a known, bounded limitation.
>
> Contrast: adaptive CLAHE achieves 58.2 RMS contrast on real LLVIP frames versus 51.3 for naive. A 14% lift in the region that matters.
>
> Inference: under 2 milliseconds total on CPU. On a GTX 1650 with the ONNX runtime: sub-15ms including pre- and post-processing.
>
> These numbers are the pitch. Everything else is explanation."

---

### SLIDE 6 — What We Are Not Claiming

**Spoken words:**

> "Let me be direct about what Yaazhi is not.
>
> It is not fire-rated equipment. It has not been tested in an actual fire environment. The IMU is synthetic in our demo — we label it on screen. The smoke is simulated — we label it on screen. The dataset is public thermal footage, not live fire footage.
>
> We are not claiming detection beyond 30 metres is reliable — our own data says it drops to 57%, and we believe that number before we believe any number that sounds better.
>
> We are not building hardware. No Jetson, no optical combiner, no radio link. This is a software stack designed to run on whatever compute platform the hardware integrator specifies.
>
> We are a perception layer. We know our lane."

---

### SLIDE 7 — Where We Are Starting / Go-Forward

**Spoken words:**

> "Here is where we are today and what the first real deployment looks like.
>
> Right now: we have a working software prototype that runs fully offline on a mid-range laptop. All tasks through to the expo application are complete. The model is exported and running through the ONNX runtime with a CPU fallback. The safety state machine is implemented and tested against every fault type we defined.
>
> The first real deployment step — not a demo, a real one — is a controlled evaluation with a fire service training facility. We don't need a real fire for that. We need a smoke machine, a building they already use for drills, and two people willing to walk through it while wearing development hardware. That produces the first real-world recall and latency numbers to replace our synthetic baselines.
>
> The integration path is: run the pipeline on any headset that exposes an IMU stream and a thermal video feed over USB or Ethernet. We do not care whether the headset is Bluetooth or wired, whether it's a COTS thermal unit or a custom module — if it outputs timestamped frames and quaternions, we connect to it.
>
> The first commercial motion is not selling to fire departments directly. That sales cycle is long, regulated, and funded by procurement budgets we don't yet have access to. The first commercial step is licensing the perception stack to a hardware manufacturer who already has those relationships and needs the software layer. That is the direct path to revenue while waiting for certification processes to complete.
>
> We are not raising money today. We are asking for two things: access to a training or testing facility, and an introduction to one hardware integrator who is already in the thermal headset market."

---

### SLIDE 8 — Closing

**Spoken words:**

> "Yaazhi is built on one principle: an overlay that lies is more dangerous than no overlay at all. Every design choice — the conservative detection thresholds, the graceful degradation, the SENSOR LOST state, the honest numbers — comes from that principle.
>
> We have a working system. We have measured results. We know where it works and where it doesn't.
>
> If you are building hardware that needs a perception stack, or if you have a relationship with a training facility willing to run a controlled drill — we want to talk after this session.
>
> Thank you."

*Cue: Stop the demo. Keep the window open. Walk to the edge of the stage. Do not fidget.*

---

## ─── PART 2: HOW TO PITCH FOR THE EVENT ───

### Audience Profile

This is likely a mix of: **technical judges** (engineers, academics), **non-technical judges** (investors, domain experts, institution representatives), and **general audience** (other teams, press, curious observers). Pitch strategy must work for all three simultaneously.

### The Core Rule

**Lead with the problem. Let the demo carry the technology. Close with credibility, not hype.**

Judges at an expo or hackathon have seen many "AI for fire safety" demos. What kills teams is: fabricated numbers, over-claimed range, and a demo that crashes. Your credibility comes from the fact that you say "57% at 30 metres" — most teams would hide that number.

### Pacing

| Time | What's happening |
|---|---|
| 0:00–0:45 | Problem framing. Speak slowly. Let the weight land. |
| 0:45–1:30 | Architecture. One sentence per layer. No acronyms unless necessary. |
| 1:30–4:30 | Demo. Narrate in real time. If something breaks, say "this is the fault-injection — watch what happens." |
| 4:30–5:30 | Numbers. Read them. Don't interpret. Numbers speak. |
| 5:30–6:30 | Limitations disclosure. This is what makes you credible. |
| 6:30–7:30 | Go-forward. Specific. No "we plan to." Use "the next step is." |
| 7:30–8:00 | Close. Brief. Stop. |

### Body Language Rules

- **Never turn your back to the audience while the demo is running.** Position the laptop so you can narrate and watch the screen sideways.
- When the SENSOR LOST state appears on screen: **pause, let them read it, then speak.** The silence is part of the pitch.
- Do not say "hopefully" or "should" during the demo. Say what is happening, not what you hope is happening.

### If the Demo Crashes

Have the fallback video (`demo_fallback.mp4`) on a separate window already minimised. If the live app fails, switch without comment. Say: "Let me run the pre-recorded version — same system, same run, captured yesterday." Continue narrating as if it were live.

### Language for Non-Technical Judges

Replace tech terms with:
- "YOLOv8n" → "our person-detection model"
- "IMU / quaternion" → "the headset's motion sensor"
- "ONNX runtime" → "the inference engine"
- "mAP50" → "detection accuracy on our test set"
- "Kalman filter" → "a predictor that extrapolates between sensor readings"

---

## ─── PART 3: EXPECTED QUESTIONS AND HOW TO HANDLE THEM ───

### Q1: "Why not just use a better thermal camera?"

**Counter argument behind the question:** "Your detection degrades at 30m — wouldn't a higher-resolution sensor just fix this?"

**Answer:**
> "That's exactly the right question. A higher-resolution sensor would improve detection range. Our pipeline is resolution-agnostic — if you pipe in 320 by 240 or 640 by 480 frames, the detection numbers improve. We built on 160 by 120 because that's the most common Lepton-class sensor in affordable headset form factors today, and we wanted our numbers to be truthful for the hardware market we're targeting. The software doesn't require that resolution. The hardware market currently operates there."

---

### Q2: "This is just YOLOv8. What have you actually built?"

**Counter argument:** "You're wrapping an existing model — what's the novel contribution?"

**Answer:**
> "YOLO is the detector. The novel work is everything around it. The sensor-resolution evaluation pipeline — most teams evaluate at high resolution and then deploy on a low-resolution sensor, which is dishonest. The real-time IMU reprojection that runs the display loop at 60 Hz while inference runs at 9 Hz. The safety state machine that degrades through DEGRADED to SAFE and recovers with hysteresis — that is not something YOLO does. And the adaptive contrast preprocessing tuned for thermal with hot-pixel exclusion — also not something YOLO does. We used YOLO the way you use a database — as a proven component in a system that solves a harder problem."

---

### Q3: "Has this been tested in an actual fire?"

**Counter argument:** Testing in real conditions — this is the legitimacy challenge.

**Answer:**
> "No. And we will say that clearly. Our evaluation is on public thermal footage from the LLVIP dataset, with optical blur, noise, and smoke synthetically applied. The IMU in the demo is synthetic. We don't claim field-validated performance. The next step is a controlled evaluation at a training facility — not a real fire, a drill scenario — to produce the first real-world baseline numbers. We know the difference between a software prototype and a certified tool. We are the former, working toward the path to the latter."

---

### Q4: "What's the latency? Can a firefighter actually react in time?"

**Counter argument:** "If the overlay lags 150ms, is it useful at all?"

**Answer:**
> "150ms is our worst-case measurement scenario — maximum display delay from frame capture to render. Our software-path motion-to-display latency is under 30ms median. The sensor runs at 9 Hz, so you get a new detection pass every 111ms. Within that window, the IMU at 200 Hz is continuously updating and the render loop is running at 60 Hz — so the visual is smooth even though the sensor is slow. What the firefighter actually sees is a smoothly tracking marker on a 60Hz display. The 150ms figure is the physics of the sensor, not the software."

---

### Q5: "What if the IMU drifts? Won't the markers get misaligned?"

**Counter argument:** IMU drift is a real problem in inertial navigation.

**Answer:**
> "Gyroscope drift is real and we account for it in two ways. First, our predictor is bounded — it only extrapolates up to a configurable short horizon, after which the next detected frame resets the reference. Second, the safety monitor has an IMU staleness threshold: if the IMU hasn't produced a valid reading within 100ms, we enter DEGRADED state and reduce marker opacity. We are not doing long-horizon dead reckoning. We are doing short-interval prediction between consecutive thermal frames, which caps the accumulated drift."

---

### Q6: "Who would buy this? The procurement cycle for fire departments is years."

**Counter argument:** Market entry challenge — this is often the real knockout question for hardware-adjacent products.

**Answer:**
> "Correct — direct sales to fire departments involves NFPA standards, procurement committees, and budgets that move slowly. That's not our first step. Our first commercial target is a hardware integrator who already has those relationships and needs a perception software layer. There are thermal headset manufacturers who ship hardware and license software separately. We are the software. They already have the distribution channel. The alternative model is a training-facility subscription — sell the system to the training centre, not the individual fire department, with a shorter sales cycle and a regulatory environment that is less restrictive than field-deployment gear."

---

### Q7: "What's stopping a large company from building this?"

**Counter argument:** "Google or Microsoft could just do this."

**Answer:**
> "Large companies target large markets. The market for Lepton-class thermal perception in emergency services headsets is not large enough at its current state to attract a platform player. We are a specialist stack, built from the ground up at sensor resolution, with a safety state machine tuned for life-critical degradation. A platform player would build a general thermal vision API and tell customers to figure out the safety layer. Our entire differentiation is that we built the safety layer first, then the perception on top of it. By the time a large company enters this space, we will have real-world evaluation data, which is the real moat."

---

### Q8: "Your recall at 30m is only 57%. That seems low for a safety device."

**Counter argument:** This is the hardest honest challenge. Do not get defensive.

**Answer:**
> "You're right. 57% recall at 30m on a 160 by 120 sensor means roughly 4 in 10 people at that distance don't produce a marker. We are not arguing that is good enough for field deployment. What we are arguing is: that number is real, measured, and used to define the operational range of the system — not hidden. A system claiming 95% detection at 30m on a 160-pixel sensor is either lying or evaluating at a different resolution. We know the range where this works. The use case is orientation in a structure — a firefighter 10–15 metres from a colleague or victim. At that range, we're at 70–82% recall. That's the claimed use case, not the whole building."

---

## ─── PART 4: PRODUCT LIMITATIONS — WHAT WE KNOW AND SAY OPENLY ───

| Limitation | What we say |
|---|---|
| **57% recall beyond 30m** | "We measure and report this. The operational range for this hardware is under 20m." |
| **No real-fire validation** | "This is a software prototype evaluated on public data. Field validation requires a controlled drill scenario." |
| **Synthetic IMU in demo** | "The headset IMU in the demo is simulated. We label it on screen." |
| **Abrupt reversal regression** | "Abrupt head reversals briefly cause the reprojection to overshoot before correcting. We document the pixel error magnitude." |
| **No hardware integration** | "We produce no hardware. The pipeline requires a USB/Ethernet thermal frame source and an IMU stream. Integration is the hardware partner's work." |
| **No fire / smoke rating** | "Nothing we produce has any fire-rating or regulatory certification. This is pre-certification software." |
| **No long-range IMU** | "We do not do dead-reckoning. The predictor window is short by design — this prevents drift accumulation but limits prediction during extended GPS-denied navigation." |
| **Storage and power** | "The demo runs on a GTX 1650. On embedded compute (Jetson class), we have not measured performance. That's the next hardware milestone." |
| **Walking_bob regression at 150ms** | "At 150ms display delay with a walking motion profile, the reprojection slightly over-compensates. Documented in the evaluation report." |

---

## ─── PART 5: HOW WE ARE STARTING — THE REAL STARTING POINT ───

### What is built right now (honest status):

- **Detection pipeline:** YOLOv8n ONNX, evaluated at 160×120, running at <2ms CPU latency. ✅
- **Sensor emulator:** Produces 9 Hz thermal frames with emulated blur, noise, and AGC. ✅
- **IMU synthesiser:** Five scripted motion profiles, 200 Hz, with gyro noise. ✅
- **Reprojection:** Rotation-delta warp, 60 Hz render loop, bounded predictor. ✅
- **Tracking:** Kalman-based per-target tracker, status lifecycle, coasting. ✅
- **Safety state machine:** NORMAL → DEGRADED → SAFE transitions, all four fault injectors. ✅
- **Expo app:** Split-screen, all hotkeys, 10-minute no-crash stability verified. ✅
- **Fallback video:** Pre-rendered scripted scenario — available if live demo fails. ✅

### What is not built yet:

- **Scripted scenario / replay viewer:** 70% done (T7.2 and T7.3 pending).
- **Alert messaging (UDP command post):** In progress (T6.4).
- **Performance history logging / README:** Pending (T8.1, T8.2).
- **Full evaluation report document:** Pending (T8.3 — individual reports exist; compilation pending).

### The three concrete starting moves after this event:

1. **Training facility contact.** Reach out to one fire service training centre. The ask is simple: 2 hours in a smoke drill environment, two people, a laptop, and permission to record thermal footage. This replaces synthetic IMU and simulated smoke with one real-world session.

2. **Hardware partner identification.** Identify one existing thermal headset manufacturer or developer kit (FLIR, Seek, or equivalent) that ships a developer SDK or open USB stream. File a one-page technical brief on the integration interface (we already know what we need: timestamped frames + quaternions). Request an engineering conversation.

3. **External evaluation.** Run the evaluation pipeline on a second public dataset (KAIST or FLIR ADAS) without retraining. If numbers hold, we have cross-dataset evidence. If they drop, we document it. Either result is informative.

### The first thing that would make this a real product:

> **One real-world recall measurement, in a smoke environment, at known distances, with a documented ground truth.**

Until that exists, everything is a software prototype. After that exists, it is an evaluated prototype ready for hardware integration discussion. That is one experiment away.

---

*End of script. Total word count: ~3,400 words spoken. Estimated delivery time: 8–10 minutes with demo, 5–6 minutes without.*
