import tensorflow as tf
import numpy as np
import cv2

# ----------------------------
# 1. Load the TFLite model
# ----------------------------
MODEL_PATH = "best_float16.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# ----------------------------
# 2. Get input / output details
# ----------------------------
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# ----------------------------
# 3. Load and preprocess an image
# ----------------------------
IMG_PATH = "test.jpg"  # change to your image name
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMG_PATH}")

# resize and normalize
h, w = input_details[0]['shape'][1:3]
img_resized = cv2.resize(img, (w, h))
input_data = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

# ----------------------------
# 4. Run inference
# ----------------------------
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# ----------------------------
# 5. Get model outputs
# ----------------------------
output_data = interpreter.get_tensor(output_details[0]['index'])
print("✅ Inference done!")
print("Output shape:", output_data.shape)
print("Sample output:", output_data[0][:5])

# 6️⃣ Optional: Live camera preview
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # preprocess and run inference same as above
    resized = cv2.resize(frame, (w, h))
    input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # just show camera feed for now
    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
