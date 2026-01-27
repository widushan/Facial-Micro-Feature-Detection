import mediapipe as mp
try:
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    print("MediaPipe Tasks API is available.")
except AttributeError as e:
    print(f"MediaPipe Tasks API missing: {e}")
