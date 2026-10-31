import cv2
import numpy as np
import os
import time

def bulk_capture_interface():
    storage_path = 'known_faces'
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    matricule = ""
    error_message = ""
    error_time = 0
    window_name = 'VerifyMe Enrollment'
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        side = min(h, w)
        start_x, start_y = (w - side) // 2, (h - side) // 2
        
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (start_x, start_y), (start_x+side, start_y+side), (0, 255, 0), 2)
        
        # Validation Counter
        is_valid_length = len(matricule) == 8
        counter_color = (0, 255, 0) if is_valid_length else (0, 0, 255)

        # Bottom Bar Input
        cv2.rectangle(display_frame, (0, h-50), (w, h), (0, 0, 0), -1)
        cv2.putText(display_frame, f"MATRICULE: {matricule} ({len(matricule)}/8)", (20, h-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, counter_color, 2)

        # Error Message at Top of window displaying for 2 seconds
        if error_message and (time.time() - error_time < 2):
            # Draw a small background for the error to make it readable
            cv2.rectangle(display_frame, (0, 0), (w, 40), (0, 0, 150), -1)
            cv2.putText(display_frame, error_message, (w//2 - 180, 28), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            error_message = ""

        cv2.imshow(window_name, display_frame)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(1) & 0xFF

        # Input handling
        if (key >= 48 and key <= 57) or (key >= 65 and key <= 90) or (key >= 97 and key <= 122):
            if len(matricule) < 8:
                matricule += chr(key).upper()
        
        elif key == 8: # Backspace
            matricule = matricule[:-1]

        # On-screen error
        elif key == 13 or key == 10:
            if is_valid_length:
                cropped = frame[start_y:start_y+side, start_x:start_x+side]
                file_path = os.path.join(storage_path, f"{matricule}.jpg")
                cv2.imwrite(file_path, cropped)
                
                # Visual success indicator
                error_message = f"SAVED: {matricule}"
                error_time = time.time()
                matricule = "" 
            else:
                # Triggering the onscreen error
                error_message = "MUST BE 8 CHARACTERS!"
                error_time = time.time()

        elif key == ord('q') or key == ord('Q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    bulk_capture_interface()