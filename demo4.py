import cv2
import numpy as np
import time
import joblib
import pandas as pd
import mediapipe as mp
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# -----------------------------
# 1️⃣ Load ML model & label encoder
# -----------------------------
clf = joblib.load("yoga_pose_model.pkl")
le = joblib.load("label_encoder.pkl")

# Load canonical landmarks per pose
df = pd.read_csv("yoga_landmarks.csv")  # Each row: x1,y1,z1,x2,y2,z2,...,label
pose_templates = {}
for pose_name in df['label'].unique():
    pose_data = df[df['label'] == pose_name].drop(columns=['label']).values
    pose_templates[pose_name] = np.mean(pose_data, axis=0)

# -----------------------------
# 2️⃣ Joint mapping for angles
# -----------------------------
JOINT_INDICES = {
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
    "left_shoulder": (13, 11, 23),
    "right_shoulder": (14, 12, 24)
}

# -----------------------------
# 3️⃣ Helper functions
# -----------------------------
class SimpleLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def calculate_angle(a,b,c):
    a,b,c = np.array(a), np.array(b), np.array(c)
    ba, bc = a-b, c-b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos_angle,-1.0,1.0)))

def extract_angles(landmarks):
    angles = {}
    for joint,(a,b,c) in JOINT_INDICES.items():
        angles[joint] = calculate_angle(
            [landmarks[a].x, landmarks[a].y],
            [landmarks[b].x, landmarks[b].y],
            [landmarks[c].x, landmarks[c].y]
        )
    return angles

def compute_accuracy(user_angles, canonical_angles):
    joint_accuracy = {}
    for joint in user_angles:
        diff = abs(user_angles[joint] - canonical_angles[joint])
        joint_accuracy[joint] = max(0, 100 - diff)
    overall_accuracy = np.mean(list(joint_accuracy.values()))
    return joint_accuracy, overall_accuracy

def generate_feedback(joint_accuracy, threshold=85):
    feedback = []
    for joint, acc in joint_accuracy.items():
        if acc < threshold:
            feedback.append(f"{joint.replace('_',' ').title()} needs adjustment")
    return feedback

# -----------------------------
# 4️⃣ Initialize MediaPipe PoseLandmarker
# -----------------------------
latest_result = None
def handle_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_result
)
landmarker = PoseLandmarker.create_from_options(options)

# -----------------------------
# 5️⃣ Start Webcam Capture
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(time.time()*1000)
    landmarker.detect_async(mp_image, timestamp_ms)

    if latest_result and latest_result.pose_landmarks:
        landmarks = latest_result.pose_landmarks[0]

        # Flatten landmarks for ML model
        feature_vector = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).reshape(1,-1)
        pred_label = le.inverse_transform(clf.predict(feature_vector))[0]

        # Extract angles
        user_angles = extract_angles(landmarks)

        # Canonical landmarks & angles
        canonical_vector = pose_templates[pred_label]
        canonical_landmarks = []
        for i in range(0, len(canonical_vector), 3):
            canonical_landmarks.append(SimpleLandmark(
                x=canonical_vector[i],
                y=canonical_vector[i+1],
                z=canonical_vector[i+2]
            ))
        canonical_angles = extract_angles(canonical_landmarks)

        # Accuracy & feedback
        joint_accuracy, overall_accuracy = compute_accuracy(user_angles, canonical_angles)
        feedback = generate_feedback(joint_accuracy)

        # Draw joints (all)
        for joint, (_, b, _) in JOINT_INDICES.items():
            x, y = int(landmarks[b].x * frame.shape[1]), int(landmarks[b].y * frame.shape[0])
            color = (0, 255, 0) if joint_accuracy[joint] >= 85 else (0, 0, 255)
            cv2.circle(frame, (x, y), 6, color, -1)

        # Draw arrows for 3 worst joints
        worst_joints = sorted(joint_accuracy.items(), key=lambda x: x[1])[:3]
        for joint, _ in worst_joints:
            a, b, c = JOINT_INDICES[joint]
            x, y = int(landmarks[b].x * frame.shape[1]), int(landmarks[b].y * frame.shape[0])
            dx = int((canonical_landmarks[b].x - landmarks[b].x) * frame.shape[1] * 0.5)
            dy = int((canonical_landmarks[b].y - landmarks[b].y) * frame.shape[0] * 0.5)
            max_len = 50
            length = np.sqrt(dx*dx + dy*dy)
            if length > max_len:
                scale = max_len / length
                dx = int(dx*scale)
                dy = int(dy*scale)
            cv2.arrowedLine(frame, (x, y), (x + dx, y + dy), (0, 0, 255), 3, tipLength=0.3)

        # Display pose & accuracy
        cv2.putText(frame, f"Pose: {pred_label}", (30,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(frame, f"Accuracy: {overall_accuracy:.1f}%", (30,90), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

        # Feedback text
        for i, fb in enumerate(feedback[:5]):
            cv2.putText(frame, fb, (30,130+30*i), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    else:
        cv2.putText(frame, "Pose: Not detected", (30,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("Yoga Pose Detection - ML Feedback", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
