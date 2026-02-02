import cv2
import numpy as np
import time
import joblib
import pandas as pd
import threading
import queue
import os
from dotenv import load_dotenv
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import azure.cognitiveservices.speech as speechsdk
from playsound import playsound

# -----------------------------
# Load environment variables for Azure
# -----------------------------
load_dotenv()
SUBSCRIPTION_KEY = os.environ.get('SUBSCRIPTION_KEY')
REGION_ID = os.environ.get('REGION_ID')

# -----------------------------
# Azure Speech setup
# -----------------------------
speech_config = speechsdk.SpeechConfig(subscription=SUBSCRIPTION_KEY, endpoint=REGION_ID)
speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

speech_queue = queue.Queue()

def speech_worker():
    audio_config = speechsdk.audio.AudioOutputConfig(filename="output.wav")
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    last_text = ""
    while True:
        text = speech_queue.get()
        if text == "STOP":
            break
        # Only speak if feedback has changed
        if text != last_text:
            synthesizer.speak_text_async(text).get()
            playsound("output.wav")
            last_text = text
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# -----------------------------
# Load ML model & label encoder
# -----------------------------
clf = joblib.load("yoga_pose_model.pkl")
le = joblib.load("label_encoder.pkl")

df = pd.read_csv("yoga_landmarks.csv")
pose_templates = {}
for pose_name in df['label'].unique():
    pose_data = df[df['label'] == pose_name].drop(columns=['label']).values
    pose_templates[pose_name] = np.mean(pose_data, axis=0)

# -----------------------------
# Joint mapping & helper functions
# -----------------------------
JOINT_INDICES = {
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
    "left_shoulder": (13, 11, 23),
    "right_shoulder": (14, 12, 24)
}

class SimpleLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a-b, c-b
    cos_angle = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos_angle,-1.0,1.0)))

def extract_angles(landmarks):
    angles = {}
    for joint, (a, b, c) in JOINT_INDICES.items():
        angles[joint] = calculate_angle(
            [landmarks[a].x, landmarks[a].y],
            [landmarks[b].x, landmarks[b].y],
            [landmarks[c].x, landmarks[c].y]
        )
    return angles

def compute_accuracy(user_angles, canonical_angles):
    joint_accuracy = {}
    for joint in user_angles:
        diff = abs(user_angles[joint]-canonical_angles[joint])
        joint_accuracy[joint] = max(0,100-diff)
    overall_accuracy = np.mean(list(joint_accuracy.values()))
    return joint_accuracy, overall_accuracy

def generate_feedback(joint_accuracy, threshold=85):
    feedback = []
    for joint, acc in joint_accuracy.items():
        if acc < threshold:
            feedback.append(f"{joint.replace('_',' ').title()} needs adjustment")
    return feedback

# -----------------------------
# MediaPipe PoseLandmarker
# -----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import mediapipe as mp_tasks
BaseOptions = mp_tasks.tasks.BaseOptions
PoseLandmarker = mp_tasks.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp_tasks.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp_tasks.tasks.vision.RunningMode

latest_landmarks = None
ml_results = {"pose": None, "accuracy": None, "feedback": [], "canonical_landmarks": []}
last_spoken_feedback = ""

def handle_result(result, output_image, timestamp_ms):
    global latest_landmarks
    if result.pose_landmarks:
        latest_landmarks = result.pose_landmarks[0]

def ml_worker(landmarks):
    global ml_results, last_spoken_feedback
    try:
        feature_vector = np.array([[lm.x,lm.y,lm.z] for lm in landmarks]).reshape(1,-1)
        pred_label = le.inverse_transform(clf.predict(feature_vector))[0]

        user_angles = extract_angles(landmarks)

        canonical_vector = pose_templates[pred_label]
        canonical_landmarks = []
        for i in range(0, len(canonical_vector), 3):
            canonical_landmarks.append(SimpleLandmark(
                x=canonical_vector[i],
                y=canonical_vector[i+1],
                z=canonical_vector[i+2]
            ))
        canonical_angles = extract_angles(canonical_landmarks)

        joint_accuracy, overall_accuracy = compute_accuracy(user_angles, canonical_angles)
        feedback = generate_feedback(joint_accuracy)

        ml_results["pose"] = pred_label
        ml_results["accuracy"] = overall_accuracy
        ml_results["feedback"] = feedback
        ml_results["canonical_landmarks"] = canonical_landmarks

        # Only enqueue speech if feedback has changed
        speech_text = ""
        if feedback:
            speech_text = f"{pred_label} detected. " + ". ".join(feedback)
        else:
            speech_text = f"{pred_label} detected. Good pose."
        if speech_text != last_spoken_feedback:
            speech_queue.put(speech_text)
            last_spoken_feedback = speech_text

    except:
        pass

# -----------------------------
# Initialize PoseLandmarker
# -----------------------------
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_result
)
landmarker = PoseLandmarker.create_from_options(options)

# -----------------------------
# Webcam loop
# -----------------------------
cap = cv2.VideoCapture(0)
ml_thread = None

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp_tasks.Image(image_format=mp_tasks.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(time.time()*1000)
    landmarker.detect_async(mp_image, timestamp_ms)

    # Draw skeleton instantly
    if latest_landmarks:
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in latest_landmarks
        ])
        mp_drawing.draw_landmarks(
            frame,
            landmarks_proto,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
        )

        # ML prediction in thread
        if ml_thread is None or not ml_thread.is_alive():
            ml_thread = threading.Thread(target=ml_worker, args=(latest_landmarks,))
            ml_thread.start()

    # Display ML results
    if ml_results["pose"]:
        cv2.putText(frame, f"Pose: {ml_results['pose']}", (30,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(frame, f"Accuracy: {ml_results['accuracy']:.1f}%", (30,90), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        for i, fb in enumerate(ml_results["feedback"][:5]):
            cv2.putText(frame, fb, (30,130+30*i), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    else:
        cv2.putText(frame, "Pose: Detecting...", (30,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    cv2.imshow("Yoga Pose Detection - Smart Speech", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Stop speech thread
speech_queue.put("STOP")
speech_thread.join()
cap.release()
cv2.destroyAllWindows()
