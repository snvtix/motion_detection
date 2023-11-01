import cv2

cap = cv2.VideoCapture(0)

# background subtractor object
mog = cv2.createBackgroundSubtractorMOG2()

while True:
    # pobieranie klatek
    ret, frame = cap.read()

    # grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # background subtraction
    fgmask = mog.apply(gray)

    # reducing noise and filling gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    # detekcja obrazu
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # ignore small contours
        if cv2.contourArea(contour) < 1000:
            continue

        # draw bounding box around contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # wyswietlanie klatek
    cv2.imshow('Motion detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()