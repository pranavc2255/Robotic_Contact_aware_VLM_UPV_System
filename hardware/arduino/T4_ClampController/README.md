# T4 Clamp Controller Firmware

This firmware controls the DM542T clamp axis through:

- `DIR_PIN = 8`
- `PUL_PIN = 9`
- `BAUD = 9600`

It keeps the old manual jog convention:

- `A`: open/release jog
- `D`: close/clamp jog
- `SPACE` or `STOP`: stop

T4 automation uses explicit serial commands, not continuous jog:

- `PING`
- `STATUS`
- `STOP`
- `ZERO_OPEN`
- `OPEN_FULL`
- `CLOSE_STEPS <steps>`
- `OPEN_STEPS <steps>`
- `MOVE_TO_SPACING_MM <spacing_mm>`
- `CLAMP_TRAVEL_MM <total_closing_mm>`
- `CLAMP_ONE_SIDE_MM <one_side_mm>`
- `FORCE_OPEN_STEPS <steps>`
- `JOG_OPEN_MS <milliseconds>`
- `JOG_CLOSE_MS <milliseconds>`
- `HOLD_MS <milliseconds>`
- `SET_STEPS_PER_MM <value>`
- `SET_STEP_DELAY_US <value>`

Physical range:

- Fully open probe spacing: `245.0 mm`
- Fully closed probe spacing: `40.0 mm`

`DEFAULT_STEPS_PER_MM = 100.0` is the current calibrated value for one-side actuator motion. Calibrate the real clamp and send `SET_STEPS_PER_MM` from the T4 Python config before relying on finite clamp distances.

## Clamp Math

The firmware treats probe spacing as total probe-to-probe distance. The actuator calibration is one-side motion:

```text
total_closing_mm = 245.0 - target_probe_spacing_mm
one_side_motion_mm = total_closing_mm / 2.0
steps = one_side_motion_mm * steps_per_mm
```

Example:

```text
MOVE_TO_SPACING_MM 76.86
total_closing_mm = 245.0 - 76.86 = 168.14
one_side_motion_mm = 84.07
steps = 8407 when steps_per_mm = 100
```

`MOVE_TO_SPACING_MM` takes the desired final probe spacing, not a travel distance.

`CLAMP_TRAVEL_MM` takes total probe-to-probe closing travel. The firmware divides that by two before converting to one-side actuator steps.

`CLAMP_ONE_SIDE_MM` is available only for direct calibration/debug moves.

`OPEN_FULL` moves to `position_steps = 0`, which corresponds to `spacing_mm = 245.0`.

Finite moves reject out-of-range spacing, negative steps, and negative travel. Long moves periodically check serial input so `STOP` or `SPACE` can interrupt motion.
