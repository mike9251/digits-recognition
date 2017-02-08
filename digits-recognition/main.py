import cv2
import matplotlib.pyplot as plt
import numpy as np
from train import get_model

def get_sample_image(filename='example1.jpg'):
    return cv2.imread(filename, 0)

def show(img):
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()

def binarize(img=get_sample_image()):
    thresh = cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 4)
    return thresh

def insert_into_center(resized_digits):
    results = []
    for img in resized_digits:
        i = np.zeros((28, 28))
        # calculate center of mass of the pixels
        M = cv2.moments(img['image'])
        try:
            xc = M['m10'] / M['m00']
            yc = M['m01'] / M['m00']
        except ZeroDivisionError:
            xc = 10
            yc = 10

        # translating the image so as to position
        # this point at the center of the 28x28 field.
        start_a = max(min(4 + (10 - int(yc)), 8), 0)
        start_b = max(min(4 + (10 - int(xc)), 8), 0)
        i[start_a:start_a+20, start_b:start_b+20] = img['image']

        results.append(i)
    return results

def find_digits(frame):

	thresh = binarize(frame)
	clear_img = cv2.medianBlur(thresh, 5)
	inv = cv2.bitwise_not(clear_img)

	kernel_d = np.ones((7,7),np.uint8)
	dilation_res = cv2.dilate(inv, kernel_d, iterations = 1)

	kernel = np.ones((5,5),np.uint8)
	kernel_d = np.ones((37,37),np.uint8)

	erosion = cv2.erode(dilation_res, kernel, iterations = 2)
	dilation = cv2.dilate(erosion, kernel_d, iterations = 11)

	(img2, contours, _) = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

	max_cnt = sorted(contours, key = cv2.contourArea, reverse = True)[0]

	[x, y, w, h] = cv2.boundingRect(max_cnt)

	digits_work = dilation_res[y : y + h, x : x + w].copy()

	digits_image = frame[y : y + h, x : x + w].copy()

	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=2)
	roi = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))
	cv2.imshow('frame',roi)
	cv2.waitKey(0)

	(_, contours_digits, _) = cv2.findContours(digits_work.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

	digits = []

	hlist = []

	xlist = []
	j = 0

	cnts = sorted(contours_digits, key = cv2.contourArea)
	for i in cnts:
		[_, _, _, h] = cv2.boundingRect(i)
		hlist.append(h)

		[x1, _, _, _] = cv2.boundingRect(contours_digits[j])
		xlist.append(x1)
		j = j + 1

	hlist.sort()
	max_h = hlist[-1:][0]

	ind_median_h=hlist.index(hlist[len(hlist)//2])

	x_ind = np.argsort(np.array(xlist))

	for i in x_ind:#cnt in contours_digits:
		cnt = contours_digits[i]
		area = cv2.contourArea(cnt)

		[x, y, w, h] = cv2.boundingRect(cnt)

		if max_h - h <= hlist[ind_median_h]:
			margin = 10
			x -= margin
			y -= margin
			w += margin*2
			h += margin*2

			cv2.rectangle(digits_image,(x,y),(x+w,y+h),(255,0,0),thickness=2)
			figure = digits_work[y: y + h, x: x + w]

			if figure.size > 0:
				digits.append({
					'image': figure,
					'x': x,
					'y': y,
					'w': w,
					'h': h,
					})

	roi = cv2.resize(digits_image, (digits_image.shape[1] // 2, digits_image.shape[0] // 1))
	print(roi.shape)
	cv2.imshow('digits_image', roi)
	cv2.waitKey(0)

	return digits, digits_image


def preprocess(digits):
	for digit in digits:
		digit['image'] = cv2.resize(digit['image'], (20,20))

	centered_img = insert_into_center(digits)

	return np.vstack([digit.reshape(1, 1, 28, 28).astype(np.float)/255 for digit in centered_img])


def video():
    model = get_model()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('OCR')
    last_seen = "Number: NaN"

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = binarize(gray)
        cv2.imshow('OCR', thresh)
        key = cv2.waitKey(1)
        if key == ord('p'):
        	digits, digits_image = find_digits(gray)
        	#draw_contours(frame, contours)
        	X = preprocess(digits)
        	prediction = np.argmax(model.predict(X), axis=1)

        	print('predictions: ', prediction)

        	for i in range(len(digits)):
        		cv2.putText(digits_image, str(prediction[i]), (digits[i]['x'], digits[i]['y']),
        			cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        	roi = cv2.resize(digits_image, (digits_image.shape[1], digits_image.shape[0]))
        	cv2.imshow("Result", roi)
        	cv2.waitKey(0)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

def static_image():
	frame = get_sample_image()
	roi = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))
	cv2.imshow('frame',roi)
	key = cv2.waitKey(0)

	digits, digits_image = find_digits(frame)

	X = preprocess(digits.copy())

	model = get_model()
	prediction = np.argmax(model.predict(X), axis=1)

	for i in range(len(digits)):
		cv2.putText(digits_image, str(prediction[i]), (digits[i]['x'], digits[i]['y']),
			cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
		roi = cv2.resize(digits_image, (digits_image.shape[1] // 3, digits_image.shape[0] // 3))
		cv2.imshow("Result", roi)
	cv2.waitKey(0)


#video()
static_image()
