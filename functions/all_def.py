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
_AR = {"inited": False, "dict": None, "params": None, "detector": None}
CAP = None 
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
        cv.putText(out, text, (15, 35), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv.LINE_AA)
        return out

    left   = _label(img,        "BGR (camera)")
    middle = _label(gray_bgr,   "Gray")
    right  = _label(hue_vis_bgr,"HSV (Hue visualization)")

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
def hough_demo(img, threshold=80, minLineLength=50, maxLineGap=10,thickness=5):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200)
    c = img.copy()
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold,minLineLength, maxLineGap)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv.line(c, (x1, y1), (x2, y2), (0, 0, 255),thickness)
    return c

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
        cv.putText(preview, f"Panorama: {len(images)}/{num_frames}  (SPACE=shoot, ESC=cancel)",
                   (20,40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
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
        cv.createTrackbar("Scale x0.01", win, 100, 200, lambda x: None)

    rows, cols = img.shape[:2]
    center = (cols/2, rows/2)

    angle = cv.getTrackbarPos("Angle", win) - 180
    tx    = cv.getTrackbarPos("Translate X", win) - 150
    ty    = cv.getTrackbarPos("Translate Y", win) - 100
    scale = cv.getTrackbarPos("Scale x0.01", win) / 100.0

    M = cv.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty


    out = cv.warpAffine(
        img, M, (cols, rows),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT, borderValue=(47,53,66))


    cv.putText(out, f"Angle={angle}, tx={tx}, ty={ty}, scale={scale:.2f}",
                (20,40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)

    return out

#10. Calibrate the camera

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
        except:
            pass
    if found:
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), term)
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

def _save_calib(mtx, dist, path="calibration.npz"):
    np.savez(path, mtx=mtx, dist=dist)

def _ensure_maps_for_size(img_size, alpha=1.0):
    w, h = img_size
    K, dist = _CALIB["mtx"], _CALIB["dist"]
    need_new = (
        _CALIB.get("map_res") != (w, h) or
        _CALIB.get("alpha") != float(alpha) or
        _CALIB.get("map1") is None or _CALIB.get("map2") is None
    )
    if need_new:
        newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
        map1, map2 = cv.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv.CV_16SC2)
        _CALIB.update({
            "newmtx": newK, "roi": roi,
            "map1": map1, "map2": map2,
            "map_res": (w, h), "alpha": float(alpha)
        })

def calibrate_demo(img, state=None, alpha_preview=1.0):

    out = img.copy()
    h, w = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if _CALIB.get("loaded", False):
        _ensure_maps_for_size((w, h), alpha_preview)
        undist = cv.remap(img, _CALIB["map1"], _CALIB["map2"], interpolation=cv.INTER_LINEAR)
        x, y, w2, h2 = _CALIB["roi"]
        if w2 > 0 and h2 > 0:
            und = undist[y:y+h2, x:x+w2]
            und = cv.copyMakeBorder(und, 0, h - und.shape[0], 0, w - und.shape[1],
                                    cv.BORDER_CONSTANT, value=(47,53,66))
        else:
            und = undist
        out = cv.hconcat([img, und])
        _draw_text(out, "Calibrated view (Before | After)", (20, 40), 0.9, (0,255,0))
        if _CALRUN["rms"] is not None:
            _draw_text(out, f"RMS reprojection error: {_CALRUN['rms']:.3f}", (20, 80), 0.8, (0,255,255))
        return out

    
    found, corners = _try_find_corners(gray, _CALRUN["board"])
    if found:
        cv.drawChessboardCorners(out, _CALRUN["board"], corners, found)

    now = time.time()
    if found and now >= _CALRUN["next_t"] and len(_CALRUN["objpoints"]) < _CALRUN["shots_needed"]:
        shift = _mean_corner_shift(_CALRUN["last_corners"], corners)
        if shift > 2.5:  # require some motion to avoid duplicates
            _CALRUN["objpoints"].append(_mk_objp(_CALRUN["board"], _CALRUN["square_mm"]))
            _CALRUN["imgpoints"].append(corners.reshape(-1,1,2))
            _CALRUN["last_corners"] = corners.copy()
            _CALRUN["next_t"] = now + _CALRUN["interval"]
            _draw_text(out, f"Captured {_CALRUN['objpoints'].__len__()}/{_CALRUN['shots_needed']}",
                       (20, 120), 0.9, (0,255,255))

    if len(_CALRUN["objpoints"]) >= _CALRUN["shots_needed"]:
        img_size = (w, h)
        rms, K, dist, rvecs, tvecs = cv.calibrateCamera(
            _CALRUN["objpoints"], _CALRUN["imgpoints"], img_size, None, None)
        _CALRUN["rms"] = float(rms)
        _CALIB.update({
            "loaded": True,
            "mtx": K.astype(np.float32),
            "dist": dist.astype(np.float32),
            "fallback": False
        })
        _ensure_maps_for_size(img_size, alpha_preview)
        _save_calib(K, dist)
        _draw_text(out, f"Calibrated! RMS={_CALRUN['rms']:.3f}  saved: calibration.npz",
                   (20, 80), 0.8, (0,255,0))

        undist = cv.remap(img, _CALIB["map1"], _CALIB["map2"], interpolation=cv.INTER_LINEAR)
        x, y, w2, h2 = _CALIB["roi"]
        if w2 > 0 and h2 > 0:
            und = undist[y:y+h2, x:x+w2]
            und = cv.copyMakeBorder(und, 0, h - und.shape[0], 0, w - und.shape[1],
                                    cv.BORDER_CONSTANT, value=(47,53,66))
        else:
            und = undist
        out = cv.hconcat([img, und])
        return out

    _draw_text(out, "Show a 9x6 chessboard. Move/tilt; it will auto-capture.",
               (20, 40), 0.9, (0,255,255))
    _draw_text(out, f"Captured: {len(_CALRUN['objpoints'])}/{_CALRUN['shots_needed']}",
               (20, 80), 0.9, (255,255,0))
    _draw_text(out, "Tip: vary distance and angle for better accuracy.",
               (20, 120), 0.8, (200,200,200))
    return out

def calib_reset():
    """Call when switching modes to reset the capture state (keeps existing _CALIB)."""
    _CALRUN.update({
        "objpoints": [], "imgpoints": [],
        "last_corners": None, "next_t": 0.0, "rms": None
    })

#11. Augmented Reality
def ar_demo(img, state=None, calib_path="calibration.npz", marker_length_m=0.05):
   
    K, dist, used_fallback = _ensure_calib(calib_path)
    detector = _ensure_aruco()

    out = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        _draw_text(out, "AR mode: No ArUco markers detected", (20, 40), 0.9, (0,0,255))
        if used_fallback:
            _draw_text(out, "WARNING: calibration.npz not found (using fallback intrinsics)", (20, 80), 0.7, (0,255,255))
        return out

    cv.aruco.drawDetectedMarkers(out, corners, ids)

    rvecs, tvecs, _objPoints = cv.aruco.estimatePoseSingleMarkers(corners, marker_length_m, K, dist)

    L = float(marker_length_m)
    obj_pts = np.float32([
        [0,0,0], [L,0,0], [L,L,0], [0,L,0],       # bottom square
        [0,0,-L],[L,0,-L],[L,L,-L],[0,L,-L]       # top square (negative z upward from board)
    ])

    rvec, tvec = rvecs[0], tvecs[0]
    img_pts, _ = cv.projectPoints(obj_pts, rvec, tvec, K, dist)
    img_pts = np.int32(img_pts).reshape(-1, 2)

    faces = [
        [0,1,2,3],  # bottom
        [4,5,6,7],  # top
        [0,1,5,4],  # side 1
        [1,2,6,5],  # side 2
        [2,3,7,6],  # side 3
        [3,0,4,7],  # side 4
    ]
    face_colors = [
        (255, 0, 0),    # blue
        (0, 255, 0),    # green
        (0, 0, 255),    # red
        (255, 255, 0),  # cyan
        (255, 0, 255),  # magenta
        (0, 255, 255),  # yellow
    ]
    for idx, face in enumerate(faces):
        cv.fillConvexPoly(out, img_pts[face], face_colors[idx], lineType=cv.LINE_AA)
    for face in faces:
        cv.polylines(out, [img_pts[face]], True, (0,0,0), 2, cv.LINE_AA)

    _draw_text(out, f"AR: id={int(ids[0])}, marker={marker_length_m*1000:.0f} mm", (20, 40), 0.9)
    if used_fallback:
        _draw_text(out, "WARNING: fallback intrinsics (add calibration.npz for accuracy)", (20, 80), 0.7, (0,255,255))

    return out