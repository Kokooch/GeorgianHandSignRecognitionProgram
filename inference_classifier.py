import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary for label mapping
labels_dict = {
    0: 'ა', 1: 'ბ', 2: 'გ', 3: 'დ', 4: 'ე', 5: 'ვ', 6: 'ზ',
    7: 'თ', 8: 'ი', 9: 'კ', 10: 'ლ', 11: 'მ', 12: 'ნ', 13: 'ო',
    14: 'პ', 15: 'ჟ', 16: 'რ', 17: 'ს', 18: 'ტ', 19: 'უ', 20: 'ფ',
    21: 'ქ', 22: 'ღ', 23: 'ყ', 24: 'შ', 25: 'ჩ', 26: 'ც', 27: 'ძ',
    28: 'წ', 29: 'ჭ', 30: 'ხ', 31: 'ჯ', 32: 'ჰ'
}

# Load a TTF font for rendering Unicode characters
font_path = "./DejaVuSans.ttf"  # Path to DejaVu Sans font
def put_text_with_pil(frame, text, position, font, color=(0, 0, 0)):
    """Overlay text on an image using PIL for Unicode support."""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    # Convert the frame color to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Process only the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
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

        # Ensure the feature vector matches the training format
        data_aux = np.asarray(data_aux).reshape(1, -1)

        # Make prediction
        prediction = model.predict(data_aux)

        # Get the predicted character
        predicted_character = labels_dict[int(prediction[0])]

        # Draw bounding box and predicted label
        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
        x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        # Draw the predicted character using PIL
        font = ImageFont.truetype(font_path, 32)
        frame = put_text_with_pil(frame, predicted_character, (x1, y1 - 40), font, color=(0, 0, 0))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
