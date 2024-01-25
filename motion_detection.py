import cv2

mog2 = cv2.createBackgroundSubtractorMOG2()
gmg = cv2.bgsegm.createBackgroundSubtractorGMG()
mog = cv2.bgsegm.createBackgroundSubtractorMOG()
cnt = cv2.bgsegm.createBackgroundSubtractorCNT()
gsoc = cv2.bgsegm.createBackgroundSubtractorGSOC()
lsbp = cv2.bgsegm.createBackgroundSubtractorLSBP()
knn = cv2.createBackgroundSubtractorKNN()

def backgroundSubtraction(bg_sub, video, window_name):
    while True:
        # pobieranie klatek
        # ret is a boolean indicating whether the frame was read successfully
        ret, frame = video.read()

        if not ret:
            break

        motion_detected = False

        # skala szarości
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # background subtraction
        fgmask = bg_sub.apply(gray)

        # redukcja szumów i ubytków
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=1)

        # detekcja obrazu
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            motion_detected = True

        if motion_detected:
            cv2.putText(frame, "Wykryto Ruch!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)

        # wyswietlanie klatek
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) == ord('q'):
            break

def foregroundExtraction(video, window_name):
    while True:
        # Pobieranie dwóch kolejnych klatek
        ret, frame1 = video.read()
        ret, frame2 = video.read()

        if not ret:
            break

        motion_detected = False

        # Konwersja klatek na odcienie szarości
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Różnica klatek
        fgmask = cv2.absdiff(gray1, gray2)

        # Thresholding
        _, fgmask = cv2.threshold(fgmask, 30, 255, cv2.THRESH_BINARY)

        # Redukcja szumów
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=1)

        # Detekcja konturów
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Ignorowanie małych konturów
            if cv2.contourArea(contour) < 100:
                continue

            # Rysowanie prostokąta wokół konturu
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 1)

            motion_detected = True

        if motion_detected:
            cv2.putText(frame, "Wykryto Ruch!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)

        # Wyświetlanie klatek
        cv2.imshow(window_name, frame1)

        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    print("Dostępne metody wykrywania ruchu z wykorzystaniem: ")
    print("1. background subtractor MOG2")
    print("2. background subtractor GMG")
    print("3. background subtractor MOG")
    print("4. background subtractor CNT")
    print("5. background subtractor GSOC")
    print("6. background subtractor LSBP")
    print("7. background subtractor KNN")
    print("8. foreground extraction")

    selected_method = input("Enter the number corresponding to the desired method: ")

    names = ["kalk.MOV", "ball.MOV", "wave.MOV"]

    if selected_method.isdigit() and 1 <= int(selected_method) <= 8:
        if selected_method == "1":
            for name in names:
                cap = cv2.VideoCapture(name)
                backgroundSubtraction(mog2,cap,"Background Subtractor: MOG2")
                cap.release()
                cv2.destroyAllWindows()
        if selected_method == "2":
            for name in names:
                cap = cv2.VideoCapture(name)
                backgroundSubtraction(gmg,cap,"Background Subtractor: GMG")
                cap.release()
                cv2.destroyAllWindows()
        if selected_method == "3":
            for name in names:
                cap = cv2.VideoCapture(name)
                backgroundSubtraction(mog,cap,"Background Subtractor: MOG")
                cap.release()
                cv2.destroyAllWindows()
        if selected_method == "4":
            for name in names:
                cap = cv2.VideoCapture(name)
                backgroundSubtraction(cnt,cap,"Background Subtractor: CNT")
                cap.release()
                cv2.destroyAllWindows()
        if selected_method == "5":
            for name in names:
                cap = cv2.VideoCapture(name)
                backgroundSubtraction(gsoc,cap,"Background Subtractor: GSOC")
                cap.release()
                cv2.destroyAllWindows()
        if selected_method == "6":
            for name in names:
                cap = cv2.VideoCapture(name)
                backgroundSubtraction(lsbp,cap,"Background Subtractor: LSBP")
                cap.release()
                cv2.destroyAllWindows()
        if selected_method == "7":
            for name in names:
                cap = cv2.VideoCapture(name)
                backgroundSubtraction(knn,cap,"Background Subtractor: KNN")
                cap.release()
                cv2.destroyAllWindows()
        if selected_method == "8":
            for name in names:
                cap = cv2.VideoCapture(name)
                foregroundExtraction(cap,"Foreground Extraction")
                cap.release()
                cv2.destroyAllWindows()
    else:
        print("Invalid input.")

# xxx