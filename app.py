import streamlit as st
import numpy as np
from PIL import Image, ImageFile
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import os
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

st.set_page_config(
    page_title="Yoga Pose Recognition",
    page_icon="🧘",
    layout="centered"
)

st.title("🧘 Yoga Pose Recognition")
st.markdown("Upload a yoga pose image and get instant prediction with joint angle analysis!")

@st.cache_resource
def load_models():
    model = tf.keras.models.load_model(
        r'C:\Users\yasha\OneDrive\Desktop\yoga recognition\best_angle_model.keras'
    )
    interpreter = tf.lite.Interpreter(
        model_path=r'C:\Users\yasha\OneDrive\Desktop\yoga recognition\movenet.tflite'
    )
    interpreter.allocate_tensors()
    return model, interpreter

CLASSES = ['downdog', 'goddess', 'plank', 'tree', 'warrior']

KEYPOINTS = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}

ANGLE_NAMES = [
    'Left elbow', 'Right elbow',
    'Left shoulder', 'Right shoulder',
    'Left hip', 'Right hip',
    'Left knee', 'Right knee',
    'Torso left', 'Torso right'
]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def get_keypoints(img, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img_resized = img.resize((257, 257))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    # Heatmaps se keypoints nikalo
    heatmaps = interpreter.get_tensor(output_details[0]['index'])[0]  # (9,9,17)
    offsets  = interpreter.get_tensor(output_details[1]['index'])[0]  # (9,9,34)
    
    height, width = 257, 257
    num_keypoints = 17
    keypoints = np.zeros((num_keypoints, 3))
    
    for kp_idx in range(num_keypoints):
        heatmap = heatmaps[:, :, kp_idx]
        conf = np.max(heatmap)
        idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        y_cell, x_cell = idx
        y_offset = offsets[y_cell, x_cell, kp_idx]
        x_offset = offsets[y_cell, x_cell, kp_idx + num_keypoints]
        y = (y_cell / 8.0 * 257 + y_offset) / height
        x = (x_cell / 8.0 * 257 + x_offset) / width
        keypoints[kp_idx] = [
            np.clip(y, 0, 1),
            np.clip(x, 0, 1),
            float(1 / (1 + np.exp(-conf)))
        ]
    
    return keypoints

def extract_angles(kps):
    def get(name):
        idx = KEYPOINTS[name]
        if idx >= len(kps):
            return [0.0, 0.0]
        return [kps[idx][1], kps[idx][0]]
    angles = []
    try:
        angles.append(calculate_angle(get('left_shoulder'),  get('left_elbow'),   get('left_wrist')))
        angles.append(calculate_angle(get('right_shoulder'), get('right_elbow'),  get('right_wrist')))
        angles.append(calculate_angle(get('left_elbow'),     get('left_shoulder'), get('left_hip')))
        angles.append(calculate_angle(get('right_elbow'),    get('right_shoulder'),get('right_hip')))
        angles.append(calculate_angle(get('left_shoulder'),  get('left_hip'),     get('left_knee')))
        angles.append(calculate_angle(get('right_shoulder'), get('right_hip'),    get('right_knee')))
        angles.append(calculate_angle(get('left_hip'),       get('left_knee'),    get('left_ankle')))
        angles.append(calculate_angle(get('right_hip'),      get('right_knee'),   get('right_ankle')))
        angles.append(calculate_angle(get('left_shoulder'),  get('left_hip'),     get('right_hip')))
        angles.append(calculate_angle(get('right_shoulder'), get('right_hip'),    get('left_hip')))
    except:
        angles = [0.0] * 10
    return np.array(angles, dtype=np.float32)

def preprocess_image(img):
    img = img.convert('RGB')
    img.thumbnail((224, 224), Image.Resampling.LANCZOS)
    new_img = Image.new('RGB', (224, 224), (255, 255, 255))
    offset = ((224 - img.width) // 2, (224 - img.height) // 2)
    new_img.paste(img, offset)
    img_array = img_to_array(new_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, new_img

def draw_keypoints(img, kps):
    img_cv = np.array(img.convert('RGB'))
    h, w = img_cv.shape[:2]
    num_kps = len(kps)
    connections = [
        (0, 1), (1, 3), (0, 2), (2, 4),
        (0, 5), (0, 6), (5, 6),
        (5, 7), (6, 8)
    ]
    for a, b in connections:
        if a < num_kps and b < num_kps:
            if kps[a][2] > 0.3 and kps[b][2] > 0.3:
                pt1 = (int(kps[a][1] * w), int(kps[a][0] * h))
                pt2 = (int(kps[b][1] * w), int(kps[b][0] * h))
                cv2.line(img_cv, pt1, pt2, (255, 165, 0), 2)
    for kp in kps:
        if kp[2] > 0.3:
            x, y = int(kp[1] * w), int(kp[0] * h)
            cv2.circle(img_cv, (x, y), 5, (0, 200, 255), -1)
    return Image.fromarray(img_cv)

def predict(img, model, interpreter):
    img_array, img_224 = preprocess_image(img)
    kps = get_keypoints(img.convert('RGB'), interpreter)
    angles = extract_angles(kps)
    angles_input = np.expand_dims(angles, axis=0)
    pred = model.predict([img_array, angles_input], verbose=0)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx] * 100
    skeleton_img = draw_keypoints(img_224, kps)
    return class_idx, confidence, pred[0], angles, skeleton_img

model, interpreter = load_models()
st.success("Model loaded!")

option = st.radio("Choose input method:",
                  ["Upload single image", "Upload folder"])

if option == "Upload single image":
    uploaded_file = st.file_uploader(
        "Choose a yoga pose image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
    )
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        class_idx, confidence, probs, angles, skeleton = predict(img, model, interpreter)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_column_width=True)
        with col2:
            st.image(skeleton, caption="Detected Joints", use_column_width=True)

        st.markdown("---")
        st.markdown(f"### Predicted Pose: **{CLASSES[class_idx].upper()}**")
        st.markdown(f"### Confidence: **{confidence:.1f}%**")

        st.markdown("#### Class probabilities:")
        for i, cls in enumerate(CLASSES):
            st.progress(int(probs[i] * 100), text=f"{cls}: {probs[i]*100:.1f}%")

        st.markdown("#### Joint angles detected:")
        cols = st.columns(2)
        for i, (name, angle) in enumerate(zip(ANGLE_NAMES, angles)):
            cols[i % 2].metric(name, f"{angle:.1f}°")

else:
    folder_path = st.text_input("Enter folder path:")
    if folder_path and os.path.exists(folder_path):
        IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        files = [f for f in os.listdir(folder_path)
                 if os.path.splitext(f)[1].lower() in IMG_EXTS]
        st.info(f"Found {len(files)} images")

        for fname in files:
            fpath = os.path.join(folder_path, fname)
            try:
                img = Image.open(fpath).convert('RGB')
                class_idx, confidence, probs, angles, skeleton = predict(img, model, interpreter)

                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    st.image(img, caption=fname, use_column_width=True)
                with col2:
                    st.image(skeleton, caption="Joints", use_column_width=True)
                with col3:
                    st.markdown(f"**{CLASSES[class_idx].upper()}**")
                    st.markdown(f"Confidence: **{confidence:.1f}%**")
                    for i, cls in enumerate(CLASSES):
                        st.progress(int(probs[i]*100), text=f"{cls}: {probs[i]*100:.1f}%")
                st.markdown("---")
            except Exception as e:
                st.warning(f"Could not process {fname}: {e}")
    elif folder_path:
        st.error("Folder not found!")