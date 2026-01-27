import mediapipe as mp
print(dir(mp))
try:
    print(mp.solutions)
except AttributeError as e:
    print(e)
