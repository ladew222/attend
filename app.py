from flask import Flask, jsonify, request, send_from_directory, render_template, redirect, url_for
import os
import csv
import cv2
import dlib
import numpy as np
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask import render_template, flash, redirect, url_for, request
from datetime import datetime
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time


# This will store the last time we saved an image
last_saved_time = time.time()


app = Flask(__name__)

app.secret_key = "3efae3"
# Constants
CAPTURED_IMAGES_DIR = "captured_images"
PRELOADED_IMAGES_DIR = "preloaded_images"
STATIC_IMAGES_DIR = os.path.join("static", "img")
FACE_TAGS_CSV = "face_tags.csv"
FACE_EMBEDDINGS_CSV = "face_embeddings.csv"
ITEMS_PER_PAGE = 20
FACE_BOUNDING_BOX_CSV = "face_bounding_boxes.csv"
# Paths for storing images and the CSV
image_storage_path = 'stored_images'
face_detections_csv = 'face_detections.csv'
MATCHES_CSV_PATH = 'matches.csv'


# Lock for thread-safe operations on the CSV file
csv_file_lock = threading.Lock()


def save_face_image(face_image, match_id, timestamp):
    # Save the face image with a filename that includes the match_id and the timestamp
    filename = f"match_{match_id}_{timestamp.replace(':', '')}.jpg"
    filepath = os.path.join(CAPTURED_IMAGES_DIR, filename)
    cv2.imwrite(filepath, face_image)
    print(f"Saved face image of match {match_id}: {filename}")

def process_video_feed():
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process the frame
            process_frame(frame)

            # Save an image every 30 seconds
            global last_saved_time
            if time.time() - last_saved_time > 30:
                save_frame_image(frame)
                last_saved_time = time.time()

            # Press 'q' to close the video window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

def process_frame(frame):
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    rects = detector(gray, 1)

    # Loop over each detected face and check for matches
    for rect in enumerate(rects):
        shape = sp(gray, rect)
        embedding = np.array(facerec.compute_face_descriptor(frame, shape))

        # Use your search_similar_faces function to check for a match
        match_id = search_similar_faces(embedding)

        if match_id is not None:
            # If there's a match, record the time and match_id
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record_match(match_id, timestamp)
            
            # Optionally, save the image of the matched face
            matched_face_image = frame[rect.top():rect.bottom(), rect.left():rect.right()]
            save_face_image(matched_face_image, match_id, timestamp)

        # Draw a bounding box around the face
        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame in a window
    cv2.imshow('Video', frame)
    
    
def record_match(match_id, timestamp):
    """Record a match with a timestamp to a CSV file."""
    with open(MATCHES_CSV_PATH, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([match_id, timestamp])

def save_frame_image(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"frame_{timestamp}.jpg"
    filepath = os.path.join(CAPTURED_IMAGES_DIR, filename)
    cv2.imwrite(filepath, frame)
    print(f"Saved image: {filename}")


# Global variable to keep track of the face recognition state
face_recognition_enabled = False


def search_similar_faces(new_embedding, threshold=0.5):
    # Load the embeddings and the tags into pandas DataFrames
    df_embeddings = pd.read_csv(FACE_EMBEDDINGS_CSV)
    df_tags = pd.read_csv(FACE_TAGS_CSV)
    
    # Join the embeddings with the tags based on the embedding index
    df_combined = df_embeddings.join(df_tags.set_index('embedding_index'), on=df_embeddings.index)

    # Calculate the similarity of the new_embedding with all embeddings in the DataFrame
    df_combined['similarity'] = df_combined.apply(lambda row: get_similarity(new_embedding, row[['dim_' + str(i) for i in range(128)]]), axis=1)
    
    # Filter the results based on the threshold and sort by similarity
    matched_faces = df_combined[df_combined['similarity'] >= threshold]
    matched_faces = matched_faces.sort_values(by='similarity', ascending=False)
    
    # Return the matched faces
    return matched_faces[['directory_name', 'image_name', 'face_index', 'student_id', 'similarity']]


def get_face_embedding(directory_name, image_name, face_index):
    if os.path.exists(FACE_EMBEDDINGS_CSV):
        df = pd.read_csv(FACE_EMBEDDINGS_CSV)
        matching_row = df[(df["directory_name"] == directory_name) & (df["image_name"] == image_name) & (df["face_index"] == face_index)]
        
        # If a matching row is found, return the embedding values (dim_0, dim_1, ... dim_127)
        if not matching_row.empty:
            embedding = matching_row.iloc[0][["dim_" + str(i) for i in range(128)]].values
            return embedding
    return None

def ensure_csv_exists(file_path, headers):
    """
    Create a CSV file with the specified headers if it doesn't exist.
    """
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
 
def save_face_details_to_csv(directory_name, image_name, face_index, student_id, time, date, class_name):
    """
    Save face details to a CSV file.
    """
    # Check if CSV exists, if not create it with headers
    ensure_csv_exists(face_detections_csv, ["directory_name", "image_name", "face_index", "student_id", "time", "date", "class_name"])
    
    with csv_file_lock:
        with open(face_detections_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([directory_name, image_name, face_index, student_id, time, date, class_name])           

def get_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]
            
ensure_csv_exists(FACE_EMBEDDINGS_CSV, ["directory_name", "image_name", "face_index"] + [f"dim_{i}" for i in range(128)])

def save_embedding_to_csv(directory_name, image_name, face_index, embedding):
    """
    Save face embedding to the face_embeddings.csv using pandas.
    """
    # Define the columns and the data to be saved
    columns = ["directory_name", "image_name", "face_index"] + [f"dim_{i}" for i in range(128)]
    data = [directory_name, image_name, face_index] + list(embedding)
    
    # Convert the data to a DataFrame
    df = pd.DataFrame([data], columns=columns)

    # Append the new data to the CSV if it exists; otherwise, create it
    if os.path.exists(FACE_EMBEDDINGS_CSV):
        df.to_csv(FACE_EMBEDDINGS_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(FACE_EMBEDDINGS_CSV, index=False)


def get_all_items(directory_path):
    return [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

def save_face_embeddings(directory_name, image_name, face_index, embedding):
    df = pd.DataFrame(columns=["directory_name", "image_name", "face_index"] + [f"dim_{i}" for i in range(128)])
    df.loc[0] = [directory_name, image_name, face_index] + list(embedding)
    if os.path.exists(FACE_EMBEDDINGS_CSV):
        df.to_csv(FACE_EMBEDDINGS_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(FACE_EMBEDDINGS_CSV, index=False)

def load_students_from_csv():
    return pd.read_csv('students.csv').to_dict('records')

def save_face_bounding_boxes(directory_name, image_name, face_index, coordinates):
    df = pd.DataFrame(columns=["directory_name", "image_name", "face_index", "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y"])
    df.loc[0] = [directory_name, image_name, face_index] + list(coordinates)
    if os.path.exists(FACE_BOUNDING_BOX_CSV):
        df.to_csv(FACE_BOUNDING_BOX_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(FACE_BOUNDING_BOX_CSV, index=False)

def get_face_bounding_boxes(directory_name, image_name):
    if os.path.exists(FACE_BOUNDING_BOX_CSV):
        df = pd.read_csv(FACE_BOUNDING_BOX_CSV)
        matching_rows = df[(df["directory_name"] == directory_name) & (df["image_name"] == image_name)]
        for _, row in matching_rows.iterrows():
            yield (row["top_left_x"], row["top_left_y"], row["bottom_right_x"], row["bottom_right_y"])

# Ensure the CSV file for face bounding boxes exists
if not os.path.exists(FACE_BOUNDING_BOX_CSV):
    with open(FACE_BOUNDING_BOX_CSV, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["directory_name", "image_name", "face_index", "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y"])
        
        
class UploadForm(FlaskForm):
    photo = FileField('Upload Image:', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'png'], 'Images only!')
    ])

def get_sorted_images(directory_path):
    # Get all image filenames
    all_images = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    # Sort the images by their modification date
    all_images.sort(key=lambda x: os.path.getmtime(os.path.join(directory_path, x)), reverse=True)
    
    return all_images

def detect_faces(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    coordinates = []
    for rect in rects:
        top_left = (rect.left(), rect.top())
        bottom_right = (rect.right(), rect.bottom())
        coordinates.append((top_left, bottom_right))
        
    return coordinates

# Flask-Uploads configuration
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = STATIC_IMAGES_DIR
configure_uploads(app, photos)

# Initialize dlib's face recognition model and shape predictor
predictor_path = "models/shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"

sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
detector = dlib.get_frontal_face_detector()

# Ensure the CSV file for face embeddings exists
if not os.path.exists(FACE_EMBEDDINGS_CSV):
    with open(FACE_EMBEDDINGS_CSV, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["directory_name", "image_name", "face_index", "embedding"])



def save_bounding_box_to_csv(directory_name, image_name, face_index, rect):
    """
    Save bounding box coordinates to the face_bounding_box.csv using pandas.
    """
    # Define the columns and the data to be saved
    columns = ["directory_name", "image_name", "face_index", "left", "top", "right", "bottom"]
    data = [directory_name, image_name, face_index, rect.left(), rect.top(), rect.right(), rect.bottom()]
    
    # Convert the data to a DataFrame
    df = pd.DataFrame([data], columns=columns)

    # Append the new data to the CSV if it exists; otherwise, create it
    if os.path.exists(FACE_BOUNDING_BOX_CSV):
        df.to_csv(FACE_BOUNDING_BOX_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(FACE_BOUNDING_BOX_CSV, index=False)


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        image_path = os.path.join(STATIC_IMAGES_DIR, filename)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        # Calculate face embeddings and save bounding boxes
        for i, rect in enumerate(rects):
            shape = sp(gray, rect)
            embedding = facerec.compute_face_descriptor(image, shape)
            
            save_embedding_to_csv(STATIC_IMAGES_DIR, filename, i, embedding)
            save_bounding_box_to_csv(STATIC_IMAGES_DIR, filename, i, rect)

        flash('Photo uploaded, embeddings and bounding boxes saved successfully', 'success')
        return redirect(url_for('index'))
    return render_template('upload.html', form=form)



@app.route('/serve_image/<directory_name>/<image_name>')
def serve_image(directory_name, image_name):
    # Logic to serve the image
    base_dir = os.getcwd()  # get the current directory
    image_path = os.path.join(base_dir, directory_name, image_name)
    return send_from_directory(directory_name, image_name)


@app.route('/browse_images/<path:directory_name>', methods=['GET'])
def browse_images(directory_name):
    all_items = get_all_items(directory_name)
    page = request.args.get('page', 1, type=int)
    items_per_page = 10  # or whatever value you choose
    total_items = len(all_items)
    total_pages = -(-total_items // items_per_page)  # This ensures a ceiling division

    
    all_images = get_sorted_images(directory_name)
    start_index = (page - 1) * items_per_page
    paginated_images = all_images[start_index:start_index + items_per_page]

    # Detect faces and get their coordinates
    face_coordinates = {}
    for image_name in paginated_images:
        image_path = os.path.join(directory_name, image_name)
        coordinates = detect_faces(image_path)  # Implement this
        face_coordinates[image_name] = coordinates

    # Get a list of students
    students = load_students_from_csv()


    # Fetch bounding boxes from CSV
    face_bounding_boxes = {}
    for image_name in paginated_images:
        image_path = os.path.join(directory_name, image_name)
        boxes = list(get_face_bounding_boxes(directory_name, image_name))
        face_bounding_boxes[image_name] = boxes

    return render_template(
        'browse_images.html', 
        images=paginated_images, 
        directory_name=directory_name, 
        face_coordinates=face_coordinates, 
        face_bounding_boxes=face_bounding_boxes,
        students=students,
        total_pages=total_pages,
        current_page=page
    )

FACE_TAGS_CSV = "face_tags.csv"
@app.route('/api_tag_face', methods=['POST'])
def api_tag_face():
    # Assuming the filename is in the format "directory_name/image_name"
    filename = request.form.get("filename")
    directory_name, image_name = os.path.split(filename)  # Splitting the filename into directory and image name
    directory_name = directory_name.lstrip('/')
    face_index = int(request.form.get("face_index"))
    student_id = request.form.get("student_id")

    # Get the index of the embedding
    embedding_index = None
    if os.path.exists(FACE_EMBEDDINGS_CSV):
        df_embeddings = pd.read_csv(FACE_EMBEDDINGS_CSV)
        matching_row = df_embeddings[
            (df_embeddings["directory_name"] == directory_name) & 
            (df_embeddings["image_name"] == image_name) & 
            (int(df_embeddings["face_index"]) == face_index)
        ]
        if not matching_row.empty:
            embedding_index = matching_row.index[0]  # Assuming index is a sequential integer
    
    # Save the tagging data using pandas
    df_tags = pd.DataFrame(columns=["directory_name", "image_name", "face_index", "embedding_index", "student_id"])
    df_tags.loc[0] = [directory_name, image_name, face_index, embedding_index, student_id]
    if os.path.exists(FACE_TAGS_CSV):
        df_tags.to_csv(FACE_TAGS_CSV, mode='a', header=False, index=False)
    else:
        df_tags.to_csv(FACE_TAGS_CSV, index=False)

    return jsonify({"status": "success"})

    # You may want to run the video feed processing in a separate thread
    # This allows the Flask server to run concurrently with the video feed
    video_thread = Thread(target=process_video_feed)
    video_thread.start()

if __name__ == "__main__":
    app.run(debug=True)