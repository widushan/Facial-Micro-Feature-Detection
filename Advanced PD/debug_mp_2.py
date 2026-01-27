try:
    import mediapipe.python.solutions as mp_solutions
    print("Imported mediapipe.python.solutions successfully")
    print(dir(mp_solutions))
except ImportError as e:
    print(f"ImportError: {e}")

try:
    from mediapipe import solutions
    print("from mediapipe import solutions success")
except ImportError as e:
    print(f"from mediapipe import solutions failed: {e}")
