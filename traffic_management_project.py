import cv2

# Video dosyalarının adları
video_files = ['intersection1.mp4', 'intersection2.mp4', 'intersection3.mp4', 'intersection4.mp4']

# VideoCapture nesnelerini oluştur
video_captures = [cv2.VideoCapture(file) for file in video_files]

# Pencere oluştur
cv2.namedWindow('Traffic Monitoring')

# Pre-trained car and pedestrian classifiers
car_trackerfile = 'car_detect.xml'
pedestrian_trackerfile = "pedestrian_detect.xml"

# create car and pedestrian classifiers
car_tracker = cv2.CascadeClassifier(car_trackerfile)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_trackerfile)

# Variables for motion analysis
fgbg = cv2.createBackgroundSubtractorMOG2()

# Döngüyü başlat
while True:
    car_counts = [0, 0, 0, 0]
    pedestrian_counts = [0, 0, 0, 0]
    combined_frame_top = None
    combined_frame_bottom = None

    # Her bir videoyu oku ve araçları/insanları tespit et
    for i, video_capture in enumerate(video_captures):
        ret, frame = video_capture.read()

        # Eğer video tamamlanmışsa, döngüyü sonlandır
        if not ret:
            break

        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fgmask = fgbg.apply(grayscaled_frame)
        fgmask = cv2.GaussianBlur(fgmask, (21, 21), 0)

        # Detect cars in the frame and draw rectangles
        cars = car_tracker.detectMultiScale(grayscaled_frame)
        for (x, y, w, h) in cars:
            # Perform motion analysis for car detection
            roi = fgmask[y:y+h, x:x+w]
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    car_counts[i] += 1
        
        # Detect pedestrians in the frame and draw rectangles
        pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            pedestrian_counts[i] += 1

        # Videoları ekranda göstermek için boyutlandır
        frame = cv2.resize(frame, (320, 240))  # Örnek boyutlandırma (640x480)

        # Aracı ve yayayı sayıları ekrana yazdır
        cv2.putText(frame, "Cars: " + str(car_counts[i]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "Pedestrians: " + str(pedestrian_counts[i]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Videoları üstte ve altta olacak şekilde böl
        if i < 2:
            if combined_frame_top is None:
                combined_frame_top = frame
            else:
                combined_frame_top = cv2.hconcat([combined_frame_top, frame])
        else:
            if combined_frame_bottom is None:
                combined_frame_bottom = frame
            else:
                combined_frame_bottom = cv2.hconcat([combined_frame_bottom, frame])

    # Üst ve alt videoları birleştir
    combined_frame = cv2.vconcat([combined_frame_top, combined_frame_bottom])

    # Kombinasyonlu frame'i ekranda göster
    cv2.imshow('Traffic Monitoring', combined_frame)

    # Çıkış için 'q' tuşuna basılırsa döngüyü sonlandır
    if cv2.waitKey(1) == ord('q'):
        break

# Pencereyi kapat ve video dosyalarını serbest bırak
cv2.destroyAllWindows()
for video_capture in video_captures:
    video_capture.release()

