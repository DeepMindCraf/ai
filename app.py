#this code will ilustrate how an ai model describe what it see in using the help of relatime webcam 
#this model need an api key and  internet connection 


import cv2
import requests
from google import genai
from google.genai import types
import pyttsx3
import time


engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()

# initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Optimize resolution and frame rate to reduce lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

cv2.namedWindow('Real-Time Detection', cv2.WINDOW_NORMAL)

client = genai.Client(api_key="your_Api_key")

if not cap.isOpened():
    speak("Unable to access the camera.")
    print("Error: Unable to access the camera.")
    exit()

speak("Camera access successful. Starting real-time detection.")
print("Camera access successful. Starting real-time detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        speak("Failed to capture frame.")
        print("Error: Failed to capture frame.")
        break

    
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()

    try:
       
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                "What is this image?",
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            ]
        )

        
        description = response.text
        print(f"Current Detection: {description}")
        speak(description)  

    except Exception as e:
        error_message = f"Error while processing the frame: {e}"
        print(error_message)
        speak(error_message)

   
    cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Real-Time Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
speak("Exiting real-time detection.")
print("Exiting real-time detection...")
