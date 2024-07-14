import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Remove existing data.pickle file if it exists
if os.path.exists('data.pickle'):
    os.remove('data.pickle')
    print('Removed existing data.pickle')

DATA_DIR = './data'
data = []
labels = []

# Define a fixed length for each data point (21 landmarks * 2 values per landmark)
FIXED_LENGTH = 42

def pad_or_truncate(sequence, target_length):
    if len(sequence) > target_length:
        return sequence[:target_length]
    elif len(sequence) < target_length:
        return sequence + [0] * (target_length - len(sequence))
    else:
        return sequence

# Process each image in the dataset
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Pad or truncate the data to ensure a fixed length
            data_aux = pad_or_truncate(data_aux, FIXED_LENGTH)
            data.append(data_aux)
            labels.append(int(dir_))

# Save the dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print('Dataset created successfully')
