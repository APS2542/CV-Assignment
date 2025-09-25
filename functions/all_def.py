import cv2 as cv
import numpy as np
import time
import os
import time
_CALRUN = {"board": (9, 6),   
    "square_mm": 25.0,   
    "shots_needed": 20,  
    "interval": 2.0,     
    "next_t": 0.0,
    "objpoints": [],
    "imgpoints": [],
    "last_corners": None,
    "rms": None}
ARUCO_DICT = cv.aruco.DICT_6X6_250
MARKER_LENGTH_M = 0.05     
if "_AROBJ" not in globals():
    _AROBJ = {"path": "data/trex_model.obj",   
        "scale": 3,                   
        "face_step": 1,                  
        "max_faces": 30000,              
        "verts": None, "faces": None, "loaded": False,}
_CALIB = {
    "loaded": False, "mtx": None, "dist": None,
    "newmtx": None, "roi": (0,0,0,0),
    "map1": None, "map2": None,
    "map_res": None, "alpha": 1.0,
    "fallback": False,}
CAP = None

#0 Normal mode
def normal_demo(img, state=None):
    return img 

#1 convert image color (RGB > Grey > HSV)
def color_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    hue_vis_bgr = cv.cvtColor(
        cv.merge([h, np.full_like(h, 255), np.full_like(h, 255)]),cv.COLOR_HSV2BGR)
    
    def _label(im, text):
        out = im.copy()
        cv.putText(out, text, (15, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2, cv.LINE_AA)
        return out

    left   = _label(img,        "BGR (camera)")
    middle = _label(gray_bgr,   "Gray")
    right  = _label(hue_vis_bgr,"HSV")

    return cv.hconcat([left, middle, right])

#2. contrast and brightness adjustment
def cb_demo(img):
    win = "Contrast & Brightness"
    try:
        exists = cv.getWindowProperty(win, 0) >= 0
    except:
        exists = False
    if not exists:
        cv.namedWindow(win, cv.WINDOW_NORMAL)
        cv.resizeWindow(win, 420, 100)
        cv.createTrackbar("alpha",    win, 100, 150, lambda x: None) 
        cv.createTrackbar("beta", win,  50, 100, lambda x: None)  

    a = cv.getTrackbarPos("alpha", win) / 100.0   
    b = cv.getTrackbarPos("beta", win) - 50      

    out = cv.convertScaleAbs(img, alpha=a, beta=b)
    return out

#3. plot histogram
def hist_demo(img):
    bins, h_h, h_w = 256, 200, 256
    colors = [(255,0,0), (0,255,0), (0,0,255)]  # B,G,R
    hists = []
    for ch in range(3):
        hist = cv.calcHist([img],[ch],None,[bins],[0,256]).ravel()
        hists.append(hist)
    max_h = max(float(h.max()) for h in hists) or 1.0

    canvas = np.zeros((h_h, h_w, 3), np.uint8)
    for ch, col in enumerate(colors):
        scaled = (hists[ch] / max_h) * (h_h - 1)
        for x in range(bins):
            y = int(scaled[x])
            cv.line(canvas, (x, h_h-1), (x, h_h-1-y), col, 1)

    hist_vis = cv.resize(canvas, (img.shape[1], h_h), cv.INTER_AREA)
    return cv.vconcat([img, hist_vis])

#4. Gaussian filter with changeable parameter
def gaussian_demo(img):
    win = "Gaussian filter"
    try:
        exists = cv.getWindowProperty(win, 0) >= 0
    except:
        exists = False
    if not exists:
        cv.namedWindow(win, cv.WINDOW_NORMAL)
        cv.resizeWindow(win, 420, 90)
        cv.createTrackbar("kernel_sixe",  win, 5, 31,  lambda x: None)
        cv.createTrackbar("sigma",   win, 10, 100, lambda x: None)
        

    k = cv.getTrackbarPos("kernel_sixe", win)
    k = max(3, k | 1)
    sigma = cv.getTrackbarPos("sigma", win) / 10.0
    out = cv.GaussianBlur(img, (k, k), sigma if sigma > 0 else 0)
    return out


#5. Bilateral filter with changeable parameters
def bilateral_demo(img):
    win = "Bilateral filter"
    try:
        exists = cv.getWindowProperty(win, 0) >= 0
    except:
        exists = False
    if not exists:
        cv.namedWindow(win, cv.WINDOW_NORMAL)
        cv.resizeWindow(win, 420, 120)
        cv.createTrackbar("d (px)",      win,  9,  25,  lambda x: None)   
        cv.createTrackbar("sigmaColor",  win, 75, 150,  lambda x: None)   
        cv.createTrackbar("sigmaSpace",  win, 75, 150,  lambda x: None)   

    d  = max(1, cv.getTrackbarPos("d (px)", win))
    sc = float(cv.getTrackbarPos("sigmaColor", win))
    ss = float(cv.getTrackbarPos("sigmaSpace", win))

    out = cv.bilateralFilter(img, d, sc, ss)
    return out

#6. Canny edge detection
def canny_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    win = "Canny edge detection"
    try:
        exists = cv.getWindowProperty(win, 0) >= 0
    except cv.error:
        exists = False

    if not exists:
        cv.namedWindow(win, cv.WINDOW_NORMAL)
        cv.resizeWindow(win, 400, 80)
        cv.createTrackbar("threshold1", win, 100, 500, lambda x: None)
        cv.createTrackbar("threshold2", win, 200, 500, lambda x: None)

    t1 = cv.getTrackbarPos("threshold1", win)
    t2 = cv.getTrackbarPos("threshold2", win)

    low, high = min(t1, t2), max(t1, t2)

    edges = cv.Canny(gray, low, high)
    return cv.cvtColor(edges, cv.COLOR_GRAY2BGR)


#7. Line detection using Hough Transform
def hough_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200)
    Red_line = img.copy()
    lines = cv.HoughLines(edges, 1, np.pi/180, 150)
    if lines is not None:
        for rho, theta in lines[:, 0]:
        # Convert polar coords (rho, theta) to Cartesian direction
          a = np.cos(theta)  # x-component of the unit direction vector
          b = np.sin(theta)  # y-component of the unit direction vector
          x0 = a * rho
          y0 = b * rho
          x1 = int(x0 + 1000 * (-b))
          y1 = int(y0 + 1000 * (a))
          x2 = int(x0 - 1000 * (-b))
          y2 = int(y0 - 1000 * (a))
          cv.line(Red_line, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return Red_line

#8. Panorama
def stitch_images(images, ratio=0.75, min_matches=10, ransac_reproj=4.0):
    pano = images[0].copy()
    for i in range(1, len(images)):
        orb = cv.ORB_create(4000)
        k1, d1 = orb.detectAndCompute(pano, None)
        k2, d2 = orb.detectAndCompute(images[i], None)
        if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
            print("[Panorama] Not enough keypoints/descriptors.")
            return pano

        bf = cv.BFMatcher(cv.NORM_HAMMING)
        raw = bf.knnMatch(d1, d2, k=2)
        good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
        if len(good) < min_matches:
            print(f"[Panorama] Not enough matches: {len(good)} < {min_matches}")
            return pano

        pts1 = np.float32([k1[m.queryIdx].pt for m in good])
        pts2 = np.float32([k2[m.trainIdx].pt for m in good])

        H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, ransac_reproj)
        if H is None or mask is None or mask.sum() < min_matches:
            print("[Panorama] Homography failed.")
            return pano

        h1, w1 = pano.shape[:2]
        h2, w2 = images[i].shape[:2]

        corners_img  = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
        corners_base = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
        warped_corners = cv.perspectiveTransform(corners_img, H)
        all_corners = np.vstack((warped_corners, corners_base))

        xmin, ymin = np.floor(all_corners.min(axis=0).ravel()).astype(int)
        xmax, ymax = np.ceil (all_corners.max(axis=0).ravel()).astype(int)
        tx, ty = -xmin, -ymin

        T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float32)
        canvas_size = (xmax - xmin, ymax - ymin)

        warped = cv.warpPerspective(images[i], T @ H, canvas_size)

        canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=pano.dtype)
        canvas[ty:ty+h1, tx:tx+w1] = pano

       
        nonzero = warped.sum(axis=2) > 0
        canvas[nonzero] = warped[nonzero]

        pano = canvas
    return pano

def capture_images(num_frames=3, cam_index=0, win_name="CV-APP"):
    global CAP
    own_cap = False
    cap = CAP
    if cap is None:
        cap = cv.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError("Failed to open the webcam.")
        own_cap = True

    images = []
    print(f"Press SPACE to capture {num_frames} images, ESC to cancel.")
    while True:
        ok, frame = cap.read()
        if not ok: break

        preview = frame.copy()
        cv.putText(preview, f"Panorama: {len(images)}/{num_frames}  (SPACE=capture, ESC=cancel)",
                   (20,40), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
        x, y, margin, tw = 20, preview.shape[0]-20, 8, 160
        for img in images[-3:][::-1]:  
            th = int(img.shape[0] * (tw / img.shape[1]))
            thumb = cv.resize(img, (tw, th))
            y -= th
            preview[y:y+th, x:x+tw] = thumb
            y -= margin

        cv.imshow(win_name, preview)
        k = cv.waitKey(1) & 0xFF
        if k == 27:          
            images = []
            break
        elif k == 32:       
            images.append(frame.copy())
            print(f"Captured: {len(images)}/{num_frames}")
            if len(images) >= num_frames:
                break

    if own_cap:
        cap.release()
    return images

def panorama_demo(img, state=None):
    if getattr(panorama_demo, "_cooldown", 0) > 0 and hasattr(panorama_demo, "_last"):
        panorama_demo._cooldown -= 1
        return panorama_demo._last

    imgs = capture_images(num_frames=3, win_name="CV-APP")
    if len(imgs) >= 2:
        pano = stitch_images(imgs)    
        panorama_demo._last = pano
        panorama_demo._cooldown = 360  
        return pano
    else:
        return img

#9. Image translation, rotation, and scale
def transform_demo(img):
    win = "Image transform"
    try:
        exists = cv.getWindowProperty(win, 0) >= 0
    except:
        exists = False
    if not exists:
        cv.namedWindow(win, cv.WINDOW_NORMAL)
        cv.resizeWindow(win, 420, 160)
        cv.createTrackbar("Angle",       win, 180, 360, lambda x: None)
        cv.createTrackbar("Translate X", win, 150, 300, lambda x: None)
        cv.createTrackbar("Translate Y", win, 100, 200, lambda x: None)
        cv.createTrackbar("Scale", win, 100, 200, lambda x: None)

    rows, cols = img.shape[:2]
    center = (cols/2, rows/2)

    angle = cv.getTrackbarPos("Angle", win) - 180
    tx    = cv.getTrackbarPos("Translate X", win) - 150
    ty    = cv.getTrackbarPos("Translate Y", win) - 100
    scale = cv.getTrackbarPos("Scale", win) / 100.0

    M = cv.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    out = cv.warpAffine(
        img, M, (cols, rows),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT, borderValue=(47,53,66))
    return out

#10. Calibrate the camera
BGCOLOR = (47, 53, 66)
def _draw_text(img, text, org=(20, 40), scale=0.8, color=(255,255,255), thickness=2):
    cv.putText(img, str(text), org, cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv.LINE_AA)

def _mk_objp(board, square_mm):
    w, h = board
    objp = np.zeros((w*h, 3), np.float32)
    grid = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objp[:, :2] = grid * square_mm
    return objp

def _try_find_corners(gray, board):
    pattern_size = (board[0], board[1])
    flags = (cv.CALIB_CB_ADAPTIVE_THRESH |
             cv.CALIB_CB_NORMALIZE_IMAGE |
             cv.CALIB_CB_FAST_CHECK)
    found, corners = cv.findChessboardCorners(gray, pattern_size, flags)
    if not found and hasattr(cv, "findChessboardCornersSB"):
        try:
            found, corners = cv.findChessboardCornersSB(gray, pattern_size)
        except Exception:
            pass
    if found and corners is not None:
        h, w = gray.shape[:2]
        win = max(5, int(min(w, h) * 0.01)) 
        if win % 2 == 0:
            win += 1
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv.cornerSubPix(gray, corners, (win, win), (-1,-1), term)
    return found, corners

def _mean_corner_shift(c1, c2):
    if c1 is None or c2 is None:
        return 1e9
    a = c1.reshape(-1,2).astype(np.float32)
    b = c2.reshape(-1,2).astype(np.float32)
    n = min(len(a), len(b))
    if n == 0:
        return 1e9
    return float(np.linalg.norm(a[:n] - b[:n], axis=1).mean())

def _enough_motion(prev_corners, now_corners, frame_size, min_px=2.0, min_pct=0.005):
    if prev_corners is None or now_corners is None:
        return True
    w, h = frame_size
    diag = (w*w + h*h) ** 0.5
    thr = max(min_px, min_pct * diag)
    return _mean_corner_shift(prev_corners, now_corners) > thr

def _save_calib(mtx, dist, path="calibration.npz"):
    np.savez(path, mtx=mtx, dist=dist)

def _ensure_maps_for_size(img_size, alpha=1.0):
    w, h = img_size
    K, dist = _CALIB["mtx"], _CALIB["dist"]
    need_new = (
        _CALIB.get("map_res") != (w, h) or
        _CALIB.get("alpha") != float(alpha) or
        _CALIB.get("map1") is None or _CALIB.get("map2") is None)
    if need_new:
        newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
        map1, map2 = cv.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv.CV_16SC2)
        _CALIB.update({
            "newmtx": newK, "roi": roi,
            "map1": map1, "map2": map2,
            "map_res": (w, h), "alpha": float(alpha)})

def _undistort_fullsize(img, alpha=1.0):
    h, w = img.shape[:2]
    _ensure_maps_for_size((w, h), alpha)
    undist = cv.remap(img, _CALIB["map1"], _CALIB["map2"], interpolation=cv.INTER_LINEAR)
    x, y, w2, h2 = _CALIB["roi"]

    if w2 > 0 and h2 > 0:
        und = undist[y:y+h2, x:x+w2]
        top, left = y, x
        bottom = max(0, h - (top + und.shape[0]))
        right  = max(0, w - (left + und.shape[1]))
        und = cv.copyMakeBorder(und, top, bottom, left, right, cv.BORDER_CONSTANT, value=BGCOLOR)
        und = und[:h, :w]
    else:
        und = undist
    if und.shape[:2] != (h, w):
        und = cv.resize(und, (w, h), interpolation=cv.INTER_LINEAR)
    return und

def calibrate_demo(img, state=None, alpha_preview=1.0):
    out = img.copy()
    h, w = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if _CALIB.get("loaded", False) and _CALRUN.get("done", False) and len(_CALRUN.get("objpoints", [])) == 0:
        _draw_text(out, "Calibration successful!", (20, 60), 0.75, (0, 255, 0))
        if _CALRUN.get("rms") is not None:
            _draw_text(out, f"RMS reprojection error: {_CALRUN['rms']:.3f}", (20, 100), 0.5, (255, 255, 255))
            _draw_text(out, "Camera matrix and distortion coefficients will display in your terminal", (20, 140), 0.5, (255, 255, 255))
        return out 
    found, corners = _try_find_corners(gray, _CALRUN["board"])
    if found:
        cv.drawChessboardCorners(out, _CALRUN["board"], corners, found)
        now = time.time()
    if (found and now >= _CALRUN["next_t"] and
        len(_CALRUN["objpoints"]) < _CALRUN["shots_needed"] and
        _enough_motion(_CALRUN["last_corners"], corners, (w, h))):

        _CALRUN["objpoints"].append(_mk_objp(_CALRUN["board"], _CALRUN["square_mm"]))
        _CALRUN["imgpoints"].append(corners.reshape(-1,1,2))
        _CALRUN["last_corners"] = corners.copy()
        _CALRUN["next_t"] = now + _CALRUN["interval"]

    if len(_CALRUN["objpoints"]) >= _CALRUN["shots_needed"] and not _CALRUN.get("done", False):
        img_size = (w, h)
        rms, K, dist, rvecs, tvecs = cv.calibrateCamera(
            _CALRUN["objpoints"], _CALRUN["imgpoints"], img_size, None, None
        )
        _CALRUN["rms"] = float(rms)
        _CALRUN["done"] = True

        _CALIB.update({
            "loaded": True,
            "mtx": K.astype(np.float32),
            "dist": dist.astype(np.float32),
            "fallback": False})
        _ensure_maps_for_size(img_size, alpha_preview)
        _save_calib(K, dist)

        print("\nCalibration successful!")
        print("Camera Matrix (mtx):")
        print(K)
        print("\nDistortion Coefficients (dist):")
        print(dist)
        print(f"\nRMS reprojection error: {_CALRUN['rms']:.6f}")
        print(f"Image size used: {w} x {h}\n")

        _CALRUN["objpoints"].clear()
        _CALRUN["imgpoints"].clear()

        _draw_text(out, "Calibration successful!", (20, 60), 0.5, (0, 0, 0))
        _draw_text(out, f"RMS error: {_CALRUN['rms']:.3f}", (20, 100), 0.5, (0, 0, 0))
        return out 

    _draw_text(out, "Show a 9x6 chessboard.", (20, 60), 0.75, (0,0,0))
    _draw_text(out, f"Captured: {len(_CALRUN['objpoints'])}/{_CALRUN['shots_needed']}", (20, 100), 0.75, (0,0,0))
    return out

def calib_reset():
    _CALRUN.update({"objpoints": [], "imgpoints": [],"last_corners": None, "next_t": 0.0,"rms": None, "done": False})
    
#11. Augmented Reality     
def _load_obj(path):
    verts, faces = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if ln.startswith("v "):
                _, x, y, z = ln.split()[:4]
                verts.append([float(x), float(y), float(z)])
            elif ln.startswith("f "):
                parts = ln.split()[1:]
                idx = [int(p.split("/")[0]) - 1 for p in parts] 
                if len(idx) >= 3:
                    faces.append(idx)
    return np.asarray(verts, np.float32), faces

def _ensure_obj_loaded():
    if _AROBJ.get("loaded", False):
        return
    path = _AROBJ["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"OBJ not found: {path}")

    verts, faces = _load_obj(path)
    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("OBJ has no vertices or faces")
    verts = verts - verts.mean(axis=0, keepdims=True)
    max_len = float(np.linalg.norm(verts, axis=1).max())
    if max_len > 0:
        verts = verts / max_len

    Rx = np.array([[1,0,0],
               [0,0,-1],
               [0,1,0]], np.float32)   
    verts = verts @ Rx.T
    scale = float(_AROBJ.get("scale", 0.2))
    verts = verts * scale
    max_faces = int(_AROBJ.get("max_faces", 5000))
    face_step = int(_AROBJ.get("face_step", 4))
    if len(faces) > max_faces:
        step = int(np.ceil(len(faces) / max_faces))
        faces = faces[::step]
        face_step = max(face_step, step)
        print(f"[AR] OBJ decimated: {len(faces)} faces (step={step})")

    _AROBJ.update({
        "verts": verts.astype(np.float32),
        "faces": faces,
        "loaded": True,
        "face_step": face_step,})
    
def _draw_obj_model(img, proj_pts, faces):
    for f in faces:
        if max(f) < len(proj_pts):
            pts = proj_pts[f].reshape(-1, 2)
            if len(pts) >= 3:
                cv.fillConvexPoly(img, pts, (60, 200, 60), lineType=cv.LINE_AA)
                cv.polylines(img, [pts], True, (0, 80, 0), 1, cv.LINE_AA)      
    return img

def _pick_largest_marker(corners):
    if not corners:
        return 0
    perims = [cv.arcLength(np.int32(c[0]), True) for c in corners]
    return int(np.argmax(perims))

def _aruco_detect(gray, dict_id):
    aruco = cv.aruco
    dictionary = aruco.getPredefinedDictionary(dict_id)
    if hasattr(aruco, "ArucoDetector") and hasattr(aruco, "DetectorParameters"):
        params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, params)
        return detector.detectMarkers(gray) 
    params = None
    if hasattr(aruco, "DetectorParameters_create"):
        params = aruco.DetectorParameters_create()
    elif hasattr(aruco, "DetectorParameters"):
        params = aruco.DetectorParameters()
    return aruco.detectMarkers(gray, dictionary, parameters=params)

def _pose_from_marker(corners, idx, marker_len_m, K, dist):
    aruco = cv.aruco
    if hasattr(aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_len_m, K, dist)
        return rvecs[idx], tvecs[idx]
    half = marker_len_m / 2.0
    objp = np.array([[-half,  half, 0],
                     [ half,  half, 0],
                     [ half, -half, 0],
                     [-half, -half, 0]], dtype=np.float32)
    imgp = corners[idx][0].astype(np.float32) 
    ok, rvec, tvec = cv.solvePnP(objp, imgp, K, dist, flags=cv.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        ok, rvec, tvec = cv.solvePnP(objp, imgp, K, dist)
        if not ok:
            raise cv.error("solvePnP failed")
    return rvec, tvec

def ar_demo(img, state=None):
    try:
        _ensure_obj_loaded()
    except Exception as e:
        cv.putText(img, f"OBJ load error: {e}", (20, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv.LINE_AA)
        return img

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if "_CALIB" in globals() and isinstance(_CALIB, dict) and _CALIB.get("loaded", False):
        K, dist = _CALIB["mtx"], _CALIB["dist"]
    else:
        if not os.path.exists("calibration.npz"):
            cv.putText(img, "No calibration.npz — calibrate first (+).",
                       (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv.LINE_AA)
            return img
        with np.load("calibration.npz") as X:
            K = X["mtx"].astype(np.float32)
            dist = X["dist"].astype(np.float32)

    if not hasattr(cv, "aruco"):
        cv.putText(img, "cv2.aruco missing (install opencv-contrib-python).",
                   (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv.LINE_AA)
        return img

    try:
        corners, ids, _ = _aruco_detect(gray, ARUCO_DICT)
    except AttributeError as e:
        cv.putText(img, f"aruco API missing: {str(e)}", (20, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv.LINE_AA)
        return img

    if ids is None or len(ids) == 0:
        cv.putText(img, "Show ArUco marker",
                   (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv.LINE_AA)
        return img

    cv.aruco.drawDetectedMarkers(img, corners, ids)

    idx = _pick_largest_marker(corners)
    try:
        rvec, tvec = _pose_from_marker(corners, idx, MARKER_LENGTH_M, K, dist)
    except cv.error as e:
        cv.putText(img, f"Pose failed: {e}", (20, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv.LINE_AA)
        return img

    verts_m = (_AROBJ["verts"] / (1.0 / MARKER_LENGTH_M)).astype(np.float32)
    verts_m += np.array([0, 0, -MARKER_LENGTH_M * 0.5], np.float32)

    try:
        R, _ = cv.Rodrigues(rvec)
        verts_cam = (R @ verts_m.T).T + tvec.reshape(1, 3)
        depth = verts_cam[:, 2]
        faces = _AROBJ["faces"]
        order = np.argsort([float(np.mean(depth[f])) for f in faces])[::-1]
        sorted_faces = [faces[i] for i in order]

        imgpts, _ = cv.projectPoints(verts_m, rvec, tvec, K, dist)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        _draw_obj_model(img, imgpts, sorted_faces)

    except cv.error as e:
        cv.putText(img, f"Project err: {e}", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv.LINE_AA)
        return img

    cv.putText(img, "AR: T-REX", (20, 80),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv.LINE_AA)
    return img



