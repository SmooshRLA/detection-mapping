import keyboard
import mouse
import pyautogui

# ---------- Keyboard ----------

def keyWForward():
    keyboard.press('w')

def stopWForward():
    keyboard.release('w')

def keySBackward():
    keyboard.press('s')

def stopSBackward():
    keyboard.release('s')

def jump():
    keyboard.press_and_release('space')

# ---------- Mouse Buttons ----------

def pressLeftClick():
    mouse.press("left")

def releaseLeftClick():
    mouse.release("left")

def pressRightClick():
    mouse.press("right")

def releaseRightClick():
    mouse.release("right")

# ---------- Mouse Movement ----------

def move_Cursor(x, y):
    pyautogui.moveTo(x, y, _pause=False)  # _pause=False = no delay