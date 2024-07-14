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
filenames = []

# Define a fixed length for each data point (21 landmarks * 2 values per landmark)
EXPECTED_LENGTH = 42

def extract_hand_landmarks(image):
    data_aux = []
    x_ = []
    y_ = []

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

        # Only return data if it matches the expected length
        if len(data_aux) == EXPECTED_LENGTH:
            return data_aux
        else:
            return None
    else:
        return None

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        data_aux = extract_hand_landmarks(img)

        if data_aux:
            data.append(data_aux)
            labels.append(int(dir_))
            filenames.append(os.path.join(dir_, img_path))
        else:
            print(f"Skipping {img_path} due to inconsistent landmark extraction")

# Inspect and remove faulty entries
clean_data = []
clean_labels = []
clean_filenames = []
for i, d in enumerate(data):
    if len(d) == EXPECTED_LENGTH:
        clean_data.append(d)
        clean_labels.append(labels[i])
        clean_filenames.append(filenames[i])
    else:
        print(f"Removing faulty entry at index {i} with shape {len(d)} and filename {filenames[i]}")

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': clean_data, 'labels': clean_labels, 'filenames': clean_filenames}, f)

print('Dataset created successfully')
