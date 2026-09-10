# Hardware References: Disabled in Public Workflows

Arduino T4 clamp firmware and robot/control implementation references are retained
for traceability. Nothing here is invoked by `upv-reproduce`. No hardware CLI is
exposed by the installed package. Additional historical modules under `reference`
may contain hardware commands and must not be used without independent review.

Do not flash firmware, connect RTDE, start a camera, or actuate a transducer as
part of reproduction. Calibration, observation poses, motion limits, emergency
stops and explicit operator confirmation are site-specific. Private robot hosts
were replaced with `ROBOT_HOST_NOT_CONFIGURED` in copied text. This does not make
historical hardware code a validated or safe deployment package.
