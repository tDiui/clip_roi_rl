import cv2
import numpy as np

def apply_clahe_enhancement(frame):
    """
    CLAHE có kiểm soát - tránh phá feature của CLIP
    """

    # chỉ áp dụng nếu ảnh tối
    if frame.mean() > 90:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # giảm noise
    enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)

    return enhanced