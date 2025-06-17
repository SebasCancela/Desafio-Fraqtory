import cv2
import numpy as np
import torch
from tracknet import BallTrackerNet
import torch.nn.functional as F
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps
import argparse
from preprocess import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--input_path', type=str, help='path to input image')
    parser.add_argument('--output_path', type=str, help='path to output image')
    parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing')
    parser.add_argument('--use_homography', action='store_true', help='whether to use homography postprocessing')
    args = parser.parse_args()

    model = BallTrackerNet(out_channels=9)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    OUTPUT_WIDTH = 640
    OUTPUT_HEIGHT = 360

    image = cv2.imread(args.input_path)
    image = undistort_and_resize(image, None, 1280, 720)
    img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    inp = (img.astype(np.float32) / 255.)
    inp = torch.tensor(np.rollaxis(inp, 2, 0))
    inp = inp.unsqueeze(0)

    out = model(inp.float().to(device))[0]
    pred = F.sigmoid(out).detach().cpu().numpy()

    points = []
    for kps_num in range(8):
        heatmap = (pred[kps_num]*255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap)
        if args.use_refine_kps and x_pred and y_pred:
            x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
        points.append((x_pred, y_pred))
        
    # === Retrieve the center keypoint from the 9th heatmap
    center_heatmap = (pred[8] * 255).astype(np.uint8)
    x_center, y_center = postprocess(center_heatmap)
    if args.use_refine_kps and x_center and y_center:
        x_center, y_center = refine_kps(image, int(y_center), int(x_center))

    if x_center and y_center:
        image = cv2.circle(image, (int(x_center), int(y_center)), radius=0, color=(0, 255, 0), thickness=10)


    if args.use_homography:
        matrix_trans = get_trans_matrix(points)
        if matrix_trans is not None:
            points = cv2.perspectiveTransform(refer_kps, matrix_trans)
            points = [np.squeeze(x) for x in points]

    for j in range(len(points)):
        if points[j][0] is not None:
            x = int(points[j][0])
            y = int(points[j][1])
            image = cv2.circle(image, (x, y),
                               radius=0, color=(0, 0, 255), thickness=10)
            cv2.putText(image, str(j+1), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(args.output_path, image)
