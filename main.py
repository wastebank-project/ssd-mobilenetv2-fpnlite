from flask import Flask, request, send_file, jsonify
from PIL import Image, ExifTags
import io
import numpy as np
import cv2
from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)

# Initialize TFLite interpreter
model_path = 'detect.tflite'
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
float_input = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Load the label map into memory
with open('labelmap.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, 1)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

    return image

def get_prediction(image_bytes, threshold=0.5):
    image = Image.open(image_bytes)
    image_rgb = image.convert('RGB')
    image_resized = image_rgb.resize((width, height))
    input_data = np.expand_dims(np.array(image_resized), axis=0)

    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    img = np.array(image_rgb)
    return img, boxes, classes, scores

def draw_boxes(image, boxes, classes, scores, threshold=0.5):
    for i in range(len(scores)):
        if scores[i] > threshold:
            ymin = int(max(1, (boxes[i][0] * image.shape[0])))
            xmin = int(max(1, (boxes[i][1] * image.shape[1])))
            ymax = int(min(image.shape[0], (boxes[i][2] * image.shape[0])))
            xmax = int(min(image.shape[1], (boxes[i][3] * image.shape[1])))

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(image, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin + base_line - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    img_pil = Image.fromarray(image)
    return img_pil

@app.route("/")
def main():
    return "Response Successful!"

@app.route('/image', methods=['POST'])
def image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'}), 400

    image_file = request.files['image']
    image_bytes = io.BytesIO(image_file.read())
    image_format = Image.open(image_bytes).format  # Detect image format
    image_bytes.seek(0)  # Reset the stream to the beginning

    # Correct image orientation
    image = Image.open(image_bytes)
    image = correct_image_orientation(image)

    # Resize the image to 750x1000 pixels
    image = image.resize((750, 1000))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image_format)
    image_bytes.seek(0)  # Reset the stream to the beginning

    # Get predictions
    img, boxes, classes, scores = get_prediction(image_bytes, threshold=0.29)
    img_with_boxes = draw_boxes(img, boxes, classes, scores)

    # Save the image with boxes to a BytesIO object
    img_io = io.BytesIO()
    img_with_boxes.save(img_io, format=image_format)  # Save in the original format
    img_io.seek(0)

    return send_file(img_io, mimetype=f'image/{image_format.lower()}')

@app.route('/text', methods=['POST'])
def text():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'}), 400

    image_file = request.files['image']
    image_bytes = io.BytesIO(image_file.read())
    image_format = Image.open(image_bytes).format  # Detect image format
    image_bytes.seek(0)  # Reset the stream to the beginning

    # Correct image orientation
    image = Image.open(image_bytes)
    image = correct_image_orientation(image)

    # Resize the image to 750x1000 pixels
    image = image.resize((750, 1000))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image_format)
    image_bytes.seek(0)  # Reset the stream to the beginning

    # Get predictions
    img, boxes, classes, scores = get_prediction(image_bytes, threshold=0.5)

    # Prepare the JSON response
    predictions = []
    for box, label, score in zip(boxes, classes, scores):
        if score > 0.5:
            ymin = int(max(1, (box[0] * image.height)))
            xmin = int(max(1, (box[1] * image.width)))
            ymax = int(min(image.height, (box[2] * image.height)))
            xmax = int(min(image.width, (box[3] * image.width)))

            predictions.append({
                'label': labels[int(label)],
                'score': float(score),
                'box': [xmin, ymin, xmax, ymax]
            })

    response = {
        'predictions': predictions
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
