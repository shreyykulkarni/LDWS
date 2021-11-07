import cv2
import numpy as np 


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    print(slope, intercept)
    y1 = image.shape[0]
    y2 = int(y1*(35/100)) 
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])



def averaged_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    # right_line = np.empty(4, dtype=object)
    # left_line = np.empty(4, dtype=object)
    # lines = []
    print(lines)
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        print(left_fit_average, 'left')
        left_line = make_coordinates(image, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        print(right_fit_average, 'right')
        right_line = make_coordinates(image, right_fit_average)
    # left_fit_average = np.average(left_fit, axis=0)
    # right_fit_average = np.average(right_fit, axis=0)
    # left_line = make_coordinates(image, left_fit_average)
    # right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])
    



def canny(lane_image):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray ,(5,5),0)
    cannyy = cv2.Canny(blur, 50, 150)
    return cannyy



def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None: 
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 10)
    return line_image 



def region_of_interest(image,L1,L2,H1,H2):
    L1 = int(L1)
    L2 = int(L2)
    H1 = int(H1)
    H2 = int(H2)
    height = image.shape[0]
    polygons = np.array([[(L1, height), (L2, height), (H1, H2)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image 



'''FOR IMAGE'''
def LDI(imge,L1,L2,H1,H2):   
    image = cv2.imread(imge)
    lane_image = np.copy(image)
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image,L1,L2,H1,H2)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = averaged_slope_intercept(lane_image, lines)
    line_image = display_lines(lane_image, averaged_lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    combo_image = cv2.resize(combo_image, (500,500))
    cropped_image = cv2.resize(cropped_image, (700,700))
    cv2.imshow("result",cropped_image)
    cv2.waitKey(0)




'''FOR VIDEO'''
def LDV(image,cap,L1,L2,H1,H2):
    while(cap.isOpened()):
        lane_image = np.copy(image)
        _, frame = cap.read()
        canny_image = canny(lane_image)
        cropped_image = region_of_interest(canny_image,L1,L2,H1,H2)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = averaged_slope_intercept(lane_image, lines)
        line_image = display_lines(lane_image, lines)
        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
        cv2.imshow("result", combo_image)
        if cv2.waitKey(1) == ord('f'):
            break
    cap.release()
    cv2.destroyAllWindows()




 
