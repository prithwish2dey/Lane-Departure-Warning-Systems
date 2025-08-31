import os
import cv2
import numpy as np
import tempfile
import uuid
from flask import Flask, request, jsonify, render_template, url_for

app = Flask(__name__)
UPLOAD_FOLDER = 'static/videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- Lane Detection Functions -------------------

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def average_slope_intercept(img, lines):
    left, right = [], []
    if lines is None:
        return None

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:
                continue
            intercept = y1 - slope * x1
            if slope < 0:
                left.append((slope, intercept))
            else:
                right.append((slope, intercept))

    def make_line(points, y1, y2):
        if len(points) == 0:
            return None
        slope, intercept = np.mean(points, axis=0)
        if abs(slope) < 1e-6:
            return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [[x1, y1, x2, y2]]

    height = img.shape[0]
    y1, y2 = height, int(height * 0.6)

    left_line = make_line(left, y1, y2)
    right_line = make_line(right, y1, y2)

    return [left_line, right_line] if left_line and right_line else None

def draw_lane_overlay(frame, lines):
    lane_img = np.zeros_like(frame)
    if lines is None or len(lines) < 2:
        return lane_img, None

    left, right = lines
    if left is None or right is None:
        return lane_img, None

    lx1, ly1, lx2, ly2 = left[0]
    rx1, ry1, rx2, ry2 = right[0]

    pts = np.array([[(lx1, ly1), (lx2, ly2), (rx2, ry2), (rx1, ry1)]], dtype=np.int32)
    cv2.fillPoly(lane_img, pts, (0, 255, 0))

    pink = (255, 0, 255)
    cv2.line(lane_img, (lx1, ly1), (lx2, ly2), pink, 8)
    cv2.line(lane_img, (rx1, ry1), (rx2, ry2), pink, 8)

    lane_center = (rx1 + lx1) // 2
    return lane_img, lane_center

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    height, width = frame.shape[:2]
    roi_vertices = np.array([[
        (50, height),
        (width // 2, int(height * 0.6)),
        (width - 50, height)
    ]], dtype=np.int32)

    roi = region_of_interest(edges, roi_vertices)

    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=50,
                            minLineLength=100, maxLineGap=50)

    avg_lines = average_slope_intercept(frame, lines)
    lane_img, lane_center = draw_lane_overlay(frame, avg_lines)

    result = cv2.addWeighted(frame, 0.8, lane_img, 0.5, 0)

    if lane_center is not None:
        car_center = width // 2
        offset = abs(car_center - lane_center)
        if offset > width * 0.1:
            warning_overlay = np.zeros_like(frame)
            cv2.putText(warning_overlay, "LANE DEPARTURE!", (150, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            result = cv2.addWeighted(result, 0.7, warning_overlay, 0.6, 0)

    return result

# ------------------- Flask Routes -------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    video = request.files['video']
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video.save(temp_input.name)

    # Unique output file name
    output_filename = f"{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)

    cap = cv2.VideoCapture(temp_input.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output = process_frame(frame)
        out.write(output)

    cap.release()
    out.release()

    video_url = url_for('static', filename=f'videos/{output_filename}')
    return jsonify({'video_url': video_url})

if __name__ == '__main__':
    app.run(debug=True)
