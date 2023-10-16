import cv2
import imutils

video = cv2.VideoCapture("test3.avi")
# video = cv2.VideoCapture(0)

status, frame_inicial = video.read()
frame_inicial = imutils.resize(frame_inicial, width=500)
frame_inicial = cv2.cvtColor(frame_inicial, cv2.COLOR_BGR2GRAY)
frame_inicial = cv2.GaussianBlur(frame_inicial, (21, 21), 0)

while True:
	status, frame = video.read()
	if not status:
		break
	frame = imutils.resize(frame, width=500)

	frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_cinza = cv2.GaussianBlur(frame_cinza, (21, 21), 0)

	diferenca_frames = cv2.absdiff(frame_cinza, frame_inicial)

	thresh = cv2.threshold(diferenca_frames, 30, 255, cv2.THRESH_BINARY)[1]
	# thresh = cv2.dilate(thresh, None, iterations = 2)

	contornos, _ = cv2.findContours(
		thresh.copy(),
		cv2.RETR_EXTERNAL, 
		cv2.CHAIN_APPROX_SIMPLE
	)

	for contorno in contornos:
		if cv2.contourArea(contorno) < 10000:
			continue
		(x, y, w, h) = cv2.boundingRect(contorno)
		cv2.rectangle(
			frame,
			(x, y),
			(x + w, y + h),
			(0, 255, 0), 
			3
		)

	cv2.imshow("All Contours", frame)
	cv2.imshow("Threshold Video", thresh)
	cv2.imshow("Diff Video", diferenca_frames)
	cv2.imshow("Gray Video", frame_cinza)

	key = cv2.waitKey(1)
	if key == ord('q'):
		break

video.release()
cv2.destroyAllWindows()