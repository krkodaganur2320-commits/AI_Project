
import cv2
import numpy as np
import tensorflow as tf
import requests

# -------------------------------
# 1. Load TensorFlow Lite Model
# -------------------------------

interpreter = tf.lite.Interpreter(
    model_path="models/model_unquant.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = [
    "plastic",
    "trash",
    "paper",
    "metal",
    "glass",
    "cardboard"
]

# -------------------------------
# 2. Connect to Ollama
# -------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:0.5b"


def get_ai_advice(waste_type, confidence):

    prompt = f"""
    An AI waste classifier detected: {waste_type}.
    Its confidence is {confidence:.2f} percent.

    Give a short and simple response containing:
    1. What this waste category means.
    2. How to dispose of it properly.
    3. How it can be recycled.

    If the classification confidence is low,
    mention that the category needs verification.
    Do not assume the exact material of the object.
    """

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        return response.json()["response"]

    except requests.exceptions.RequestException as e:
        return f"Ollama connection error: {e}"


# -------------------------------
# 3. Start Camera
# -------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected!")
    exit()

print("\nAI WASTE MANAGEMENT SYSTEM STARTED")
print("-----------------------------------")
print("Press A = Get AI recycling advice")
print("Press Q = Quit")

# -------------------------------
# 4. Live Prediction
# -------------------------------

while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera frame not received")
        break

    height, width = input_details[0]["shape"][1:3]

    img = cv2.resize(frame, (width, height))

    if input_details[0]["dtype"] == np.float32:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(input_details[0]["dtype"])

    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(
        input_details[0]["index"], img
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    index = np.argmax(output)

    confidence = float(output[index]) * 100

    predicted_class = labels[index]

    if confidence < 70:
        result = "Uncertain"
    else:
        result = predicted_class

    # Display prediction
    cv2.putText(
        frame,
        f"{result}: {confidence:.2f}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press A for AI Advice | Q to Quit",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2
    )

    cv2.imshow("AI Waste Management System", frame)

    key = cv2.waitKey(1) & 0xFF

    # -------------------------------
    # 5. Ollama AI Advice
    # -------------------------------

    if key == ord('a'):

        print("\n========== AI WASTE REPORT ==========")

        print("Detected:", predicted_class)
        print(f"Confidence: {confidence:.2f}%")

        print("\nGenerating AI recycling advice...")

        advice = get_ai_advice(
            predicted_class,
            confidence
        )

        print("\nOLLAMA AI RECOMMENDATION:")
        print(advice)

        print("\n=====================================")

    if key == ord('q'):
        break


# -------------------------------
# 6. Release Resources
# -------------------------------

camera.release()
cv2.destroyAllWindows()

print("AI Waste Management System Stopped")
