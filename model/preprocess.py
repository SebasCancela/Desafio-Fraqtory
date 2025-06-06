import cv2
import numpy as np

def undistort_and_resize(img, kps, width, height):
        # Original distorted image size
        h, w = img.shape[:2]

        # Camera intrinsics and distortion coefficients for fisheye
        K = np.array([
            [w, 0, w / 2],
            [0, w, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        D = np.array([-1, 0.9, 1.05, 0], dtype=np.float64)

        # Estimate new camera matrix and undistortion maps
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=1.0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )

        # Undistort image
        img_undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Resize image to model input size (1280x720)
        img_resized = cv2.resize(img_undistorted, (width, height))

        if kps is not None:
            # Undistort and transform keypoints
            kps = np.array(kps, dtype=np.float32).reshape(-1, 1, 2)
            kps_undistorted = cv2.fisheye.undistortPoints(kps, K, D, P=new_K).reshape(-1, 2)

            scale_x = width / w
            scale_y = height / h
            kps_rescaled = kps_undistorted * np.array([scale_x, scale_y])

            return img_resized, kps_rescaled

        return img_resized