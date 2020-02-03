import cv2

cap = cv2.VideoCapture(0)
skip = 0
img = 1
person = input("Enter your name")

while True:
    val , frame = cap.read()
    skip += 1
    cv2.imshow("video_feed" , frame)
    if skip%10 == 0:
        cv2.imwrite("data/" + str(person) + "/" + str(img) + ".jpg" , frame)
        img += 1
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

