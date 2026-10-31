import cv2
import numpy as np
import os
import tflite_runtime.interpreter as tflite

def generate_database():
    # Seting up folder paths
    input_folder = 'known_faces'
    output_path = './db/database_embeddings.npz'
    model_path = './db/verifyme.tflite'
    
    if not os.path.exists('./db'):
        os.makedirs('./db')

    # Loading the Model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    embeddings_dict = {}

    # Processing known faces
    print("Starting embedding generation...")
    
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print(f"Error: No images found in '{input_folder}' folder.")
        return

    for filename in image_files:
        # The matricule is the filename without extension (e.g., CT23A004)
        matricule = os.path.splitext(filename)[0].upper()
        img_path = os.path.join(input_folder, filename)

        try:
            # Load and Preprocess exactly like the capture script
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(img_rgb, (100, 100))
            normalized = resized.astype(np.float32) / 255.0
            input_data = np.expand_dims(normalized, axis=0)

            # Generate Embedding via TFLite Interpreter
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            embedding = interpreter.get_tensor(output_details[0]['index']).flatten()

            # Store in dictionary
            embeddings_dict[matricule] = embedding
            print(f"Success: Generated embedding for {matricule}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    if embeddings_dict:
        np.savez_compressed(output_path, **embeddings_dict)
        print(f"\nDatabase successfully saved to {output_path}")
        print(f"Total students registered: {len(embeddings_dict)}")

if __name__ == "__main__":
    generate_database()