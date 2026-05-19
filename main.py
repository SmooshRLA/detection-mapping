import math
import cv2 as cv
import mediapipe as mp
import kyinput
import pyautogui

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Get screen size once for cursor mapping
SCREEN_W, SCREEN_H = pyautogui.size()


class HandDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def getDistance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def indexThumbDistance(self, hand_landmarks):
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        return self.getDistance(
            (thumb.x, thumb.y),
            (index.x, index.y)
        )

    def pinkyThumbDistance(self, hand_landmarks):
        pinky = hand_landmarks.landmark[20]
        thumb = hand_landmarks.landmark[4]
        return self.getDistance(
            (pinky.x, pinky.y),
            (thumb.x, thumb.y)
        )

    def openPalm(self, hand_landmarks):
        thumb  = hand_landmarks.landmark[4]
        index  = hand_landmarks.landmark[8]
        middle = hand_landmarks.landmark[12]
        return (
            self.getDistance((thumb.x, thumb.y),  (index.x,  index.y))  > 0.15 and
            self.getDistance((middle.x, middle.y), (index.x,  index.y)) > 0.15
        )

    def getIndexTip(self, hand_landmarks):
        """Returns normalised (x, y) of index fingertip (landmark 8)."""
        tip = hand_landmarks.landmark[8]
        return tip.x, tip.y


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        return

    detector = HandDetector()

    # --- keyboard hold states ---
    wKeyDown = False
    sKeyDown = False

    # --- mouse button hold states ---
    leftMouseDown  = False
    rightMouseDown = False

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        rgb   = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        results = detector.hands.process(rgb)

        # intentions this frame
        moveF       = False
        moveB       = False
        jumping     = False
        holdLeft    = False   # should left mouse be held?
        holdRight   = False   # should right mouse be held?

        if results.multi_hand_landmarks and results.multi_handedness:

            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # ---- pinch index + thumb ----
                if detector.indexThumbDistance(hand_landmarks) < 0.05:
                    if label == "Right":
                        moveF = True          # W key
                    else:
                        holdLeft = True       # hold left mouse button

                # ---- pinch pinky + thumb ----
                elif detector.pinkyThumbDistance(hand_landmarks) < 0.05:
                    if label == "Right":
                        moveB = True          # S key
                    else:
                        holdRight = True      # hold right mouse button

                # ---- open palm = jump ----
                if detector.openPalm(hand_landmarks):
                    jumping = True

                # ---- move cursor with LEFT hand index tip ----
                if label == "Left":
                    nx, ny = detector.getIndexTip(hand_landmarks)
                    # nx/ny are 0-1 normalised; map to screen pixels
                    screen_x = int(nx * SCREEN_W)
                    screen_y = int(ny * SCREEN_H)
                    kyinput.move_Cursor(screen_x, screen_y)

        # ==========================
        # KEYBOARD STATE MANAGEMENT
        # ==========================

        if moveF and not wKeyDown:
            kyinput.keyWForward()
            wKeyDown = True
        elif not moveF and wKeyDown:
            kyinput.stopWForward()
            wKeyDown = False

        if moveB and not sKeyDown:
            kyinput.keySBackward()
            sKeyDown = True
        elif not moveB and sKeyDown:
            kyinput.stopSBackward()
            sKeyDown = False

        if jumping:
            kyinput.jump()

        # ==========================
        # MOUSE STATE MANAGEMENT
        # ==========================

        # Left mouse button
        if holdLeft and not leftMouseDown:
            kyinput.pressLeftClick()
            leftMouseDown = True
        elif not holdLeft and leftMouseDown:
            kyinput.releaseLeftClick()
            leftMouseDown = False

        # Right mouse button
        if holdRight and not rightMouseDown:
            kyinput.pressRightClick()
            rightMouseDown = True
        elif not holdRight and rightMouseDown:
            kyinput.releaseRightClick()
            rightMouseDown = False

        # ==========================
        # SAFETY RESET (no hands)
        # ==========================

        if not results.multi_hand_landmarks:
            if wKeyDown:
                kyinput.stopWForward()
                wKeyDown = False
            if sKeyDown:
                kyinput.stopSBackward()
                sKeyDown = False
            if leftMouseDown:
                kyinput.releaseLeftClick()
                leftMouseDown = False
            if rightMouseDown:
                kyinput.releaseRightClick()
                rightMouseDown = False

        cv.imshow("Frame", image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # ---- cleanup on exit ----
    if wKeyDown:     kyinput.stopWForward()
    if sKeyDown:     kyinput.stopSBackward()
    if leftMouseDown:  kyinput.releaseLeftClick()
    if rightMouseDown: kyinput.releaseRightClick()

    cap.release()
    cv.destroyAllWindows()


main()