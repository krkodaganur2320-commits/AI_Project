
from ai_edge_litert.interpreter import Interpreter
from PIL import Image
import numpy as np

interpreter = Interpreter(model_path="models/model_unquant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = Image.open("test_images/test.jpg.jpg").convert("RGB")
image = image.resize((224, 224))

image_array = np.asarray(image, dtype=np.float32)
image_array = np.expand_dims(image_array, axis=0)

interpreter.set_tensor(input_details[0]["index"], image_array)
interpreter.invoke()

predictions = interpreter.get_tensor(output_details[0]["index"])[0]

with open("models/labels.txt", "r") as file:
    labels = [line.strip().split(" ", 1)[1] for line in file]

predicted_index = int(np.argmax(predictions))

print("\nPrediction:", labels[predicted_index])
print("Confidence:", round(float(predictions[predicted_index]) * 100, 2), "%")

print("\nAll predictions:")
for label, score in zip(labels, predictions):
    print(label, ":", round(float(score) * 100, 2), "%")
