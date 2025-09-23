import cv2 as cv
import numpy as np
from functions import all_def

MODES = [
    ("1", "Convert image color (RGB <> grey <> HSV)",all_def.color_demo),
    ("2", "Contrast and brightness",all_def.cb_demo),
    ("3", "Image histogram",all_def.hist_demo),
    ("4", "Gaussian filter with changable parameter",all_def.gaussian_demo),
    ("5", "Bilateral Filter with changable parameter",all_def.bilateral_demo),
    ("6", "Canny edge detection",all_def.canny_demo),
    ("7", "Line detection using Hough Transform",all_def.hough_demo),
    ("8", "Panorama",all_def.panorama_demo),
    ("9", "Image translation, rotation, and scale",all_def.transform_demo),
    ("+", "Calibrate the camera",all_def.calibrate_demo), 
    ("-", "Augmented Reality",all_def.ar_demo), ]

def put_overlay(img, text):
    out = img.copy()
    cv.rectangle(out, (0,0), (out.shape[1], 36), (0,0,0), -1)
    cv.putText(out, text, (10,26), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
    cv.putText(out, "1..7,8,9,+,-: switch mode / ESC: quit  ", (10, out.shape[0]-10),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    return out

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera"); return
    all_def.CAP = cap
    mode_idx = 0
    cv.namedWindow("CV-APP", cv.WINDOW_NORMAL)
    cv.resizeWindow("CV-APP", 1280, 720)

    while True:
        ok, frame = cap.read()
        if not ok: break

        key, title, fn = MODES[mode_idx]
        try:
            view = fn(frame)  
        except Exception as e:
            view = frame.copy()
            cv.putText(view, f"Error in mode '{title}': {e}", (10,60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        view = put_overlay(view, f"[{key}] {title}")
        cv.imshow("CV-APP", view)

        k = cv.waitKey(1) & 0xFF
        if k == 27:  
            break
        pressed = chr(k) if 32 <= k <= 126 else ""
        for i, (hotkey, _, _) in enumerate(MODES):
            if pressed == hotkey and i != mode_idx:
             try:
                if hasattr(all_def, "pano_reset"):
                    all_def.pano_reset()
             except Exception:
                pass
             
             cv.destroyAllWindows()

             cv.namedWindow("CV-APP", cv.WINDOW_NORMAL)
             cv.resizeWindow("CV-APP", 1280, 720)

             mode_idx = i
             break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
