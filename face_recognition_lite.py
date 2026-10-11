import dlib
import numpy as np
import face_recognition_models

# Load pre-trained models from face_recognition_models
face_detector = dlib.get_frontal_face_detector()

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.dlib_face_recognition_model_v1(face_recognition_model)

def face_locations(img, number_of_times_to_upsample=1):
    """Finds face locations in an image."""
    rects = face_detector(img, number_of_times_to_upsample)
    return [(rect.top(), rect.right(), rect.bottom(), rect.left()) for rect in rects]

def raw_face_landmarks(img, face_locations=None, model="large"):
    """Internal function to get raw shape landmarks."""
    if face_locations is None:
        rects = face_detector(img, 1)
    else:
        rects = [dlib.rectangle(css[3], css[0], css[2], css[1]) for css in face_locations]
    
    return [pose_predictor_68_point(img, rect) for rect in rects]

def face_encodings(face_image, known_face_locations=None, num_jitters=1, model="small"):
    """Generates 128-d face encodings for faces in an image."""
    raw_landmarks = raw_face_landmarks(face_image, known_face_locations, model)
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark, num_jitters)) for raw_landmark in raw_landmarks]

def face_distance(face_encodings, face_to_compare):
    """Calculates Euclidean distance between face encodings."""
    if len(face_encodings) == 0:
        return np.empty((0,))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)
