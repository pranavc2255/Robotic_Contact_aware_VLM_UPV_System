const int DIR_PIN = 8;
const int PUL_PIN = 9;
const long BAUD = 9600;

const float FULLY_OPEN_SPACING_MM = 245.0;
const float FULLY_CLOSED_SPACING_MM = 40.0;
const float DEFAULT_STEPS_PER_MM = 100.0;
const unsigned long DEFAULT_STEP_DELAY_US = 600;
const unsigned long MIN_STEP_DELAY_US = 200;
const unsigned long MAX_STEP_DELAY_US = 2000;

float stepsPerMm = DEFAULT_STEPS_PER_MM;
unsigned long stepDelayUs = DEFAULT_STEP_DELAY_US;
long positionSteps = 0;
bool openZeroSet = false;
bool moving = false;
int jogDirection = 0;  // -1 open, +1 close, 0 stopped

String readLineTrimmed() {
  String line = Serial.readStringUntil('\n');
  line.trim();
  return line;
}

float oneSideMotionFromSteps(long steps) {
  return float(steps) / stepsPerMm;
}

float spacingFromSteps(long steps) {
  return FULLY_OPEN_SPACING_MM - (2.0 * oneSideMotionFromSteps(steps));
}

float totalClosingFromSpacing(float spacingMm) {
  return FULLY_OPEN_SPACING_MM - spacingMm;
}

float oneSideMotionFromSpacing(float spacingMm) {
  return totalClosingFromSpacing(spacingMm) / 2.0;
}

long stepsFromSpacing(float spacingMm) {
  return lround(oneSideMotionFromSpacing(spacingMm) * stepsPerMm);
}

void printStatus() {
  Serial.print("OK STATUS position_steps=");
  Serial.print(positionSteps);
  Serial.print(" one_side_motion_mm=");
  Serial.print(oneSideMotionFromSteps(positionSteps), 3);
  Serial.print(" spacing_mm=");
  Serial.print(spacingFromSteps(positionSteps), 3);
  Serial.print(" moving=");
  Serial.print(moving ? 1 : 0);
  Serial.print(" homed=");
  Serial.print(openZeroSet ? 1 : 0);
  Serial.print(" steps_per_mm=");
  Serial.println(stepsPerMm, 6);
}

void setDirection(bool closeDirection) {
  digitalWrite(DIR_PIN, closeDirection ? HIGH : LOW);
}

bool checkStopDuringMove() {
  if (!Serial.available()) {
    return false;
  }
  if (Serial.peek() == ' ') {
    Serial.read();
    jogDirection = 0;
    moving = false;
    Serial.print("OK MOVE_INTERRUPTED position_steps=");
    Serial.print(positionSteps);
    Serial.print(" spacing_mm=");
    Serial.println(spacingFromSteps(positionSteps), 3);
    return true;
  }
  String cmd = readLineTrimmed();
  cmd.toUpperCase();
  if (cmd == "STOP" || cmd == "SPACE") {
    jogDirection = 0;
    moving = false;
    Serial.print("OK MOVE_INTERRUPTED position_steps=");
    Serial.print(positionSteps);
    Serial.print(" spacing_mm=");
    Serial.println(spacingFromSteps(positionSteps), 3);
    return true;
  }
  Serial.print("ERR BUSY_IGNORED command=");
  Serial.println(cmd);
  return false;
}

bool stepOnce(bool closeDirection) {
  setDirection(closeDirection);
  digitalWrite(PUL_PIN, HIGH);
  delayMicroseconds(stepDelayUs);
  digitalWrite(PUL_PIN, LOW);
  delayMicroseconds(stepDelayUs);
  positionSteps += closeDirection ? 1 : -1;
  if (positionSteps < 0) {
    positionSteps = 0;
  }
  return true;
}

bool finiteMoveSteps(bool closeDirection, long steps) {
  if (steps < 0) {
    Serial.println("ERR NEGATIVE_STEPS");
    return false;
  }
  moving = true;
  for (long i = 0; i < steps; i++) {
    if (checkStopDuringMove()) {
      moving = false;
      return false;
    }
    stepOnce(closeDirection);
  }
  moving = false;
  return true;
}

bool timedJog(bool closeDirection, unsigned long durationMs) {
  moving = true;
  unsigned long start = millis();
  while (millis() - start < durationMs) {
    if (checkStopDuringMove()) {
      moving = false;
      return false;
    }
    stepOnce(closeDirection);
  }
  moving = false;
  return true;
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) {
    return;
  }
  String upper = cmd;
  upper.toUpperCase();

  if (upper == "PING") {
    Serial.println("OK PONG");
  } else if (upper == "STATUS") {
    printStatus();
  } else if (upper == "STOP" || upper == "SPACE") {
    jogDirection = 0;
    moving = false;
    Serial.println("OK STOPPED");
  } else if (upper == "ZERO_OPEN") {
    positionSteps = 0;
    openZeroSet = true;
    Serial.println("OK ZERO_OPEN");
  } else if (upper == "OPEN_FULL") {
    long steps = positionSteps;
    Serial.println("OK OPEN_FULL_START");
    if (finiteMoveSteps(false, steps)) {
      positionSteps = 0;
      Serial.print("OK OPEN_FULL_DONE position_steps=");
      Serial.print(positionSteps);
      Serial.print(" spacing_mm=");
      Serial.println(spacingFromSteps(positionSteps), 3);
    }
  } else if (upper.startsWith("CLOSE_STEPS ")) {
    long steps = upper.substring(12).toInt();
    if (steps < 0) {
      Serial.println("ERR INVALID_STEPS");
      return;
    }
    Serial.print("OK CLOSE_STEPS_START steps=");
    Serial.println(steps);
    if (finiteMoveSteps(true, steps)) {
      Serial.print("OK CLOSE_STEPS_DONE position_steps=");
      Serial.println(positionSteps);
    }
  } else if (upper.startsWith("OPEN_STEPS ")) {
    long steps = upper.substring(11).toInt();
    if (steps < 0) {
      Serial.println("ERR INVALID_STEPS");
      return;
    }
    Serial.print("OK OPEN_STEPS_START steps=");
    Serial.println(steps);
    if (finiteMoveSteps(false, min(steps, positionSteps))) {
      Serial.print("OK OPEN_STEPS_DONE position_steps=");
      Serial.println(positionSteps);
    }
  } else if (upper.startsWith("MOVE_TO_SPACING_MM ")) {
    float spacing = upper.substring(19).toFloat();
    if (spacing < FULLY_CLOSED_SPACING_MM || spacing > FULLY_OPEN_SPACING_MM) {
      Serial.println("ERR SPACING_OUT_OF_RANGE");
      return;
    }
    float totalClosing = totalClosingFromSpacing(spacing);
    float oneSideMotion = oneSideMotionFromSpacing(spacing);
    long targetSteps = stepsFromSpacing(spacing);
    long delta = targetSteps - positionSteps;
    Serial.print("OK MOVE_TO_SPACING_START spacing_mm=");
    Serial.print(spacing, 3);
    Serial.print(" total_closing_mm=");
    Serial.print(totalClosing, 3);
    Serial.print(" one_side_motion_mm=");
    Serial.print(oneSideMotion, 3);
    Serial.print(" target_steps=");
    Serial.println(targetSteps);
    if (finiteMoveSteps(delta >= 0, labs(delta))) {
      positionSteps = targetSteps;
      Serial.print("OK MOVE_TO_SPACING_DONE spacing_mm=");
      Serial.print(spacingFromSteps(positionSteps), 3);
      Serial.print(" total_closing_mm=");
      Serial.print(totalClosing, 3);
      Serial.print(" one_side_motion_mm=");
      Serial.print(oneSideMotion, 3);
      Serial.print(" position_steps=");
      Serial.println(positionSteps);
    }
  } else if (upper.startsWith("CLAMP_TRAVEL_MM ")) {
    float totalClosing = upper.substring(16).toFloat();
    if (totalClosing < 0.0) {
      Serial.println("ERR NEGATIVE_TRAVEL");
      return;
    }
    if (totalClosing > (FULLY_OPEN_SPACING_MM - FULLY_CLOSED_SPACING_MM)) {
      Serial.println("ERR TRAVEL_OUT_OF_RANGE");
      return;
    }
    float oneSideMotion = totalClosing / 2.0;
    long steps = lround(oneSideMotion * stepsPerMm);
    Serial.print("OK CLAMP_TRAVEL_START total_closing_mm=");
    Serial.print(totalClosing, 3);
    Serial.print(" one_side_motion_mm=");
    Serial.print(oneSideMotion, 3);
    Serial.print(" steps=");
    Serial.println(steps);
    if (finiteMoveSteps(true, steps)) {
      Serial.print("OK CLAMP_TRAVEL_DONE position_steps=");
      Serial.print(positionSteps);
      Serial.print(" total_closing_mm=");
      Serial.print(totalClosing, 3);
      Serial.print(" one_side_motion_mm=");
      Serial.print(oneSideMotion, 3);
      Serial.print(" spacing_mm=");
      Serial.println(spacingFromSteps(positionSteps), 3);
    }
  } else if (upper.startsWith("CLAMP_ONE_SIDE_MM ")) {
    float oneSideMotion = upper.substring(18).toFloat();
    if (oneSideMotion < 0.0) {
      Serial.println("ERR NEGATIVE_ONE_SIDE_MOTION");
      return;
    }
    long steps = lround(oneSideMotion * stepsPerMm);
    Serial.print("OK CLAMP_ONE_SIDE_START one_side_motion_mm=");
    Serial.print(oneSideMotion, 3);
    Serial.print(" steps=");
    Serial.println(steps);
    if (finiteMoveSteps(true, steps)) {
      Serial.print("OK CLAMP_ONE_SIDE_DONE position_steps=");
      Serial.print(positionSteps);
      Serial.print(" spacing_mm=");
      Serial.println(spacingFromSteps(positionSteps), 3);
    }
  } else if (upper.startsWith("FORCE_OPEN_STEPS ")) {
    long steps = upper.substring(17).toInt();
    if (steps < 0) {
      Serial.println("ERR INVALID_STEPS");
      return;
    }
    Serial.print("OK FORCE_OPEN_STEPS_START steps=");
    Serial.println(steps);
    if (finiteMoveSteps(false, steps)) {
      Serial.print("OK FORCE_OPEN_STEPS_DONE position_steps=");
      Serial.print(positionSteps);
      Serial.print(" spacing_mm=");
      Serial.println(spacingFromSteps(positionSteps), 3);
    }
  } else if (upper.startsWith("JOG_OPEN_MS ")) {
    unsigned long ms = upper.substring(12).toInt();
    Serial.print("OK JOG_OPEN_MS_START ms=");
    Serial.println(ms);
    if (timedJog(false, ms)) {
      Serial.print("OK JOG_OPEN_MS_DONE position_steps=");
      Serial.print(positionSteps);
      Serial.print(" spacing_mm=");
      Serial.println(spacingFromSteps(positionSteps), 3);
    }
  } else if (upper.startsWith("JOG_CLOSE_MS ")) {
    unsigned long ms = upper.substring(13).toInt();
    Serial.print("OK JOG_CLOSE_MS_START ms=");
    Serial.println(ms);
    if (timedJog(true, ms)) {
      Serial.print("OK JOG_CLOSE_MS_DONE position_steps=");
      Serial.print(positionSteps);
      Serial.print(" spacing_mm=");
      Serial.println(spacingFromSteps(positionSteps), 3);
    }
  } else if (upper.startsWith("HOLD_MS ")) {
    unsigned long ms = upper.substring(8).toInt();
    Serial.print("OK HOLD_START ms=");
    Serial.println(ms);
    unsigned long start = millis();
    while (millis() - start < ms) {
      if (checkStopDuringMove()) {
        return;
      }
      delay(5);
    }
    Serial.println("OK HOLD_DONE");
  } else if (upper.startsWith("SET_STEPS_PER_MM ")) {
    float value = upper.substring(17).toFloat();
    if (value <= 0.0) {
      Serial.println("ERR INVALID_STEPS_PER_MM");
      return;
    }
    stepsPerMm = value;
    Serial.print("OK SET_STEPS_PER_MM value=");
    Serial.println(stepsPerMm, 6);
  } else if (upper.startsWith("SET_STEP_DELAY_US ")) {
    unsigned long value = upper.substring(18).toInt();
    if (value < MIN_STEP_DELAY_US || value > MAX_STEP_DELAY_US) {
      Serial.println("ERR STEP_DELAY_OUT_OF_RANGE");
      return;
    }
    stepDelayUs = value;
    Serial.print("OK SET_STEP_DELAY_US value=");
    Serial.println(stepDelayUs);
  } else if (upper == "A") {
    jogDirection = -1;
    Serial.println("OK JOG_OPEN");
  } else if (upper == "D") {
    jogDirection = 1;
    Serial.println("OK JOG_CLOSE");
  } else {
    Serial.print("ERR UNKNOWN_COMMAND ");
    Serial.println(cmd);
  }
}

void setup() {
  pinMode(DIR_PIN, OUTPUT);
  pinMode(PUL_PIN, OUTPUT);
  digitalWrite(PUL_PIN, LOW);
  Serial.begin(BAUD);
  Serial.setTimeout(20);
  Serial.println("OK T4_CLAMP_CONTROLLER_READY");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.peek();
    if (c == ' ') {
      Serial.read();
      handleCommand("SPACE");
    } else {
      handleCommand(readLineTrimmed());
    }
  }
  if (jogDirection != 0) {
    moving = true;
    stepOnce(jogDirection > 0);
  } else {
    moving = false;
    delay(2);
  }
}
