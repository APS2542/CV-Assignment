import cv2 as cv
import numpy as np
from functions import all_def

CALIBRATION_HOTKEY = "+"

MODES = [
    ("0", "Mode: Normal", all_def.normal_demo),
    ("1", "Mode: Convert image color", all_def.color_demo),
    ("2", "Mode: Contrast and brightness", all_def.cb_demo),
    ("3", "Mode: Image histogram", all_def.hist_demo),
    ("4", "Mode: Gaussian filter", all_def.gaussian_demo),
    ("5", "Mode: Bilateral Filter", all_def.bilateral_demo),
    ("6", "Mode: Canny edge detection", all_def.canny_demo),
    ("7", "Mode: Line detection (Hough Transform)", all_def.hough_demo),
    ("8", "Mode: Panorama", all_def.panorama_demo),
    ("9", "Mode: Image transform", all_def.transform_demo),
    ("+", "Mode: Calibrate the camera", all_def.calibrate_demo),
    ("-", "Mode: Augmented Reality (T-rex)", all_def.ar_demo),]

def put_overlay(img, text):
    out = img.copy()
    cv.putText(out, text, (12, 28), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(out, text, (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(out, "0,1,2,...,9,+,-: switch mode / ESC: quit",
               (10, out.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)
    return out

def main():
    if hasattr(all_def, "load_calibration"):
        try:
            all_def.load_calibration("calibration.npz")
        except Exception:
            pass

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    all_def.CAP = cap

    mode_idx = 0
    cv.namedWindow("CV-APP", cv.WINDOW_NORMAL)
    cv.resizeWindow("CV-APP", 1280, 720)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        key, title, fn = MODES[mode_idx]

        try:
            view = fn(frame)
        except Exception as e:
            view = frame.copy()
            cv.putText(view, f"Error in mode '{title}': {e}", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)

        if view is None:
            view = frame.copy()

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
                try:
                    if hotkey == CALIBRATION_HOTKEY and hasattr(all_def, "calib_reset"):
                        all_def.calib_reset()
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

