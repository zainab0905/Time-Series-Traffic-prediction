from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# -----------------------
# Setup
# -----------------------
app = Flask(__name__)
app.secret_key = 'traffic123'

# Models & assets
traffic_model = load_model('traffic_model.h5')
scaler = joblib.load('scaler.save')
model = YOLO('yolov8n.pt')  # swap to yolov8m/x for better accuracy (slower)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Label Encoder for risk
label_encoder = LabelEncoder()
label_encoder.fit(["low", "medium", "high"])

risk_map = {
    'low': 'Low Risk ✅',
    'medium': 'Medium Risk ⚠️',
    'high': 'High Risk 🚨'
}

# State
video_path = None

# Global store for unique IDs during live
live_counts = {
    'car': set(),
    'motorcycle': set(),
    'bus': set(),
    'truck': set()
}

# Colors per class (BGR)
CLASS_COLORS = {
    'car': (0, 255, 0),          # green
    'motorcycle': (255, 0, 0),   # blue
    'bus': (0, 165, 255),        # orange
    'truck': (0, 0, 255)         # red
}
DEFAULT_COLOR = (0, 255, 0)

# -----------------------
# Helpers
# -----------------------
def draw_text_with_bg(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
                      font_scale=1.0, text_thickness=2,
                      text_color=(255, 255, 255), bg_color=(0, 0, 0),
                      pad_x=6, pad_y=6):
    """Draw text with filled background rectangle for readability."""
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    x, y = org
    top_left = (x - pad_x, y - th - pad_y)
    bottom_right = (x + tw + pad_x, y + baseline + pad_y // 2)
    cv2.rectangle(img, top_left, bottom_right, bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, text_thickness, cv2.LINE_AA)


# -----------------------
# Routes
# -----------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'csvfile' not in request.files:
        flash('No CSV file uploaded')
        return redirect(url_for('index'))

    file = request.files['csvfile']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    df = pd.read_csv(filepath)

    # Extract hour
    if 'Time' in df.columns:
        df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour.fillna(0).astype(int)
    else:
        df['Hour'] = 0

    # Total vehicle count
    df['Total'] = df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']].sum(axis=1)

    # Scale & reshape
    X = df[['Hour', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount']].values
    X_scaled = scaler.transform(X)
    X_input = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # LSTM expects (samples, timesteps, features)

    # Predict risk levels
    preds = traffic_model.predict(X_input)
    y_pred = np.argmax(preds, axis=1)
    labels = label_encoder.inverse_transform(y_pred)

    df['Predicted'] = labels
    df['Risk'] = df['Predicted'].map(risk_map)

    # Save CSV
    output_path = os.path.join(UPLOAD_FOLDER, 'processed_result.csv')
    df.to_csv(output_path, index=False)

    # Sample 5 random records from each level
    high_df = df[df['Predicted'] == 'high'].sample(n=min(5, len(df[df['Predicted'] == 'high'])), random_state=None)
    medium_df = df[df['Predicted'] == 'medium'].sample(n=min(5, len(df[df['Predicted'] == 'medium'])), random_state=None)
    low_df = df[df['Predicted'] == 'low'].sample(n=min(5, len(df[df['Predicted'] == 'low'])), random_state=None)

    return render_template(
        'csv_result.html',
        last_rows=df.tail(10).to_html(classes='data', index=False),
        high_rows=high_df.to_html(classes='high', index=False),
        medium_rows=medium_df.to_html(classes='medium', index=False),
        low_rows=low_df.to_html(classes='low', index=False),
        csv_path=output_path
    )

@app.route('/upload', methods=['POST'])
def upload():
    global video_path
    file = request.files['video']
    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)
        print("Uploaded video path:", video_path)
        return redirect(url_for('result'))
    return 'No file uploaded', 400

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    global video_path
    if not video_path or not os.path.exists(video_path):
        return "No valid video uploaded", 400
    return Response(generate_frames(video_path, live=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/live_feed')
def live_feed():
    return Response(generate_frames(0, live=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze_video')
def analyze_video():
    global video_path
    if not video_path or not os.path.exists(video_path):
        return "No valid video uploaded", 400

    cap = cv2.VideoCapture(video_path)
    unique_ids = { 'car': set(), 'motorcycle': set(), 'bus': set(), 'truck': set() }

    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            break

        results = model.track(source=frame, persist=True, conf=0.3, iou=0.5)[0]
        if results.boxes:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                if label in unique_ids:
                    obj_id = int(box.id[0]) if box.id is not None else None
                    if obj_id is not None:
                        unique_ids[label].add(obj_id)

    cap.release()
    total = sum(len(v) for v in unique_ids.values())

    if total <= 10:
        level = "Low"
        recommendation = "Safe to use any vehicle (Car, Bike, Bus, Truck)."
    elif total <= 30:
        level = "Medium"
        recommendation = "Better to use medium or small vehicles (Car, Bike)."
    else:
        level = "High"
        recommendation = "Traffic is high 🚨. Prefer small vehicles (Bike, Auto)."

    return render_template("analysis.html", total=total, level=level,recommendation=recommendation)

@app.route('/analyze_live')
def analyze_live():
    total = sum(len(v) for v in live_counts.values())

    if total <= 10:
        level = "Low"
        recommendation = "Safe to use any vehicle (Car, Bike, Bus, Truck)."
    elif total <= 20:
        level = "Medium"
        recommendation = "Better to use medium or small vehicles (Car, Bike)."
    else:
        level = "High"
        recommendation = "Traffic is high 🚨. Prefer small vehicles (Bike, Auto)."

    return render_template("analysis.html", total=total, level=level,recommendation=recommendation)


# -----------------------
# Video generator
# -----------------------
def generate_frames(source, live=True):
    cap = cv2.VideoCapture(source)
    vehicle_ids = { 'car': set(), 'motorcycle': set(), 'bus': set(), 'truck': set() }

    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            break

        # Run tracker
        results = model.track(source=frame, persist=True, conf=0.3, iou=0.5)[0]

        # Draw detections
        if results.boxes:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                # only track our target classes
                if label not in vehicle_ids:
                    continue

                obj_id = int(box.id[0]) if box.id is not None else None
                if obj_id is not None:
                    vehicle_ids[label].add(obj_id)
                    if live:
                        live_counts[label].add(obj_id)

                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = CLASS_COLORS.get(label, DEFAULT_COLOR)
                thickness = 3
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Label text (bigger)
                label_text = f"{label} {obj_id}" if obj_id is not None else label
                # Place text just above the box with black bg
                text_org = (x1 + 4, max(30, y1 - 8))
                draw_text_with_bg(frame, label_text, text_org,
                                  font_scale=1.0, text_thickness=2,
                                  text_color=(255, 255, 255), bg_color=(0, 0, 0))

        # Top banner example (keep if you want a constant header)
        # draw_text_with_bg(frame, "Traffic Violation Detected", (50, 60),
        #                   font_scale=2.0, text_thickness=4)

        # --- Vehicle Counts (larger with background) ---
        y0 = 100  # vertical start
        line_gap = 60
        for i, (lbl, id_set) in enumerate(vehicle_ids.items()):
            count = len(id_set)
            text = f"{lbl.capitalize()}: {count}"
            y = y0 + i * line_gap
            draw_text_with_bg(frame, text, (20, y),
                              font_scale=1.5, text_thickness=3,
                              text_color=(0, 255, 255), bg_color=(0, 0, 0))

        # Encode & stream
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# -----------------------
# Main
# -----------------------
if __name__ == '__main__':
    # Tip: for production, set threaded=True and consider a proper server (gunicorn/uwsgi)
    app.run(debug=True, port=5001)
