import cv2

# Our Image
img_file = 'car2.jfif'
video = cv2.VideoCapture('intersection.mp4')

# Our pre-trained car and pedestrian classifier
car_trackerfile = 'car_detect.xml'
pedestrian_trackerfile = "pedestrian_detect.xml"

# create car classifier
car_tracker = cv2.CascadeClassifier(car_trackerfile)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_trackerfile)

# Initialize car and pedestrian counts

# Variables for motion analysis
fgbg = cv2.createBackgroundSubtractorMOG2()

# Run forever until car stops or something
while True:
    car_count = 0
    pedestrian_count = 0
    # Read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    # Apply background subtraction
    fgmask = fgbg.apply(grayscaled_frame)
    fgmask = cv2.GaussianBlur(fgmask, (21, 21), 0)
    
    # detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    
    # Draw rectangles around the cars and update car count
    for (x, y, w, h) in cars:
        # Perform motion analysis for car detection
        roi = fgmask[y:y+h, x:x+w]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                car_count += 1
    
    # detect pedestrians
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # Draw rectangles around the pedestrians and update pedestrian count
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        pedestrian_count += 1
    
    # Display the car and pedestrian counts
    cv2.putText(frame, "Cars: " + str(car_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, "Pedestrians: " + str(pedestrian_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the image with the cars and pedestrians spotted
    cv2.imshow('car_video', frame)

    # Dont autoclose (Wait here in the code and listen for a key press)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
video.release()
cv2.destroyAllWindows()
# motiaon analysis ile car sayısı biraz daha gerçekçi oldu
# roi(region of interest ile belki yol tanımlanığ o alan içinde araba tespiti yapılabilir.)
# bide bu örnek video hareketli motor olduğundan motion analysis tam verimli çalışmıyor olabilir ama kamera sabit olacağından obje de sabit kalacaktır onu başka mobese gibi kamera ile dene
