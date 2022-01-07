import cv2
import numpy as np
import os
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol


def postprocess(frame, outs, file, threshold, classes, target_dir):
	frameHeight, frameWidth = frame.shape[:2]
	# cv2.imwrite("orig_"+file, frame)
	classIds = []
	confidences = []
	boxes = []

	for out in outs:
		for detection in out:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = scores[classId]
			if confidence > threshold:
				x, y, width, height = detection[:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
				left = int(x - width / 2)
				top = int(y - height / 2)
				classIds.append(classId)
				confidences.append(float(confidence))
				boxes.append([left, top, int(width), int(height)])

	indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, threshold - 0.1)
	print(indices)
	for i in indices:
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		cropped_image = frame[top:top + height + 20, left:left + width + 20]
		data = decode(cropped_image, symbols=[ZBarSymbol.QRCODE])
		print(data)
		cv2.imwrite(os.path.join(target_dir, file), cropped_image)

def execute(image_path, net, file, classes, target_dir):
	frame = cv2.imread(image_path)
	threshold = 0.6

	blob = cv2.dnn.blobFromImage(frame, 1 / 255, (800, 800), swapRB=True, crop=False)
	ln = net.getLayerNames()
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
	net.setInput(blob)
	outs = net.forward(ln)
	postprocess(frame, outs, file, threshold, classes, target_dir)


def main():
	classes = open('./yolo3-tiny-qr/qrcode.names').read().strip().split('\n')
	net = cv2.dnn.readNetFromDarknet('./yolo3-tiny-qr/qrcode-yolov3-tiny.cfg',
									 './yolo3-tiny-qr/qrcode-yolov3-tiny_last.weights')
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	source_dir = "./sample_docs"
	target_dir = "./output"
	for file in os.listdir(source_dir):
		print("Executing File: {}".format(file))
		execute(os.path.join(source_dir, file), net, file, classes, target_dir)


main()
