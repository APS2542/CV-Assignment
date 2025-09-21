import cv2 as cv
from functions import convert_colors, enhance, filters, edges, lines, panorama, geom, calibrate, ar




def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
            print("Cannot open camera")
            return
    cv.namedWindow("CV-APP", cv.WINDOW_NORMAL)
    cv.resizeWindow("CV-APP", 1280, 480)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        view = convert_colors.demo(frame)

        cv.imshow("CV-APP", view)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
