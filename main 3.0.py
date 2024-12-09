import spacy
import pyttsx3
import speech_recognition as sr
import joblib
import numpy as np
import time

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Load the trained model
best_model = joblib.load('intent_classification_model.joblib')

# Text preprocessing function
def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

# Function to get the intent with confidence score
def get_intent(text):
    processed_text = preprocess_text(text)
    probabilities = best_model.predict_proba([processed_text])[0]
    intent = best_model.classes_[np.argmax(probabilities)]
    confidence = np.max(probabilities)
    return intent, confidence

# Function to get a response based on intent
def get_response(intent):
    responses = {
        "reset_password": "To reset your password, click on 'Forgot Password' at the login screen and follow the instructions.",
        "order_status": "You can check the status of your order in the 'My Orders' section of our website.",
        "contact_support": "You can contact support via email at support@example.com or call us at 123-456-7890.",
        "shipping_details": "You can find your shipping information, including tracking details, in the 'My Orders' section under 'Shipping Info'.",
        "cancel_order": "To cancel your order, go to 'My Orders', select the order, and click 'Cancel Order'.",
        "return_order": "To return an item, go to 'My Orders', select the order, and choose 'Request a Return'. Refunds are processed within 5-7 business days.",
        "payment_issue": "Please verify your payment method details in 'Payment Options' or contact support if you were charged incorrectly.",
        "exit": "Goodbye! Have a great day!",
    }
    return responses.get(intent, "This seems like a complex issue. Connecting to Human Customer Support...")

# Voice output using pyttsx3
def speak_response(response):
    engine.say(response)
    engine.runAndWait()

# Function to listen to the user input using speech recognition
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=7)
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text.lower()
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase.")
            return ""
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return ""
        except sr.RequestError:
            print("Sorry, my speech service is down.")
            return ""

# Voice-enabled chatbot function with confidence threshold
def voice_chatbot():
    start_time = time.time()
    print("Welcome to the Improved Voice Support Bot. How can I assist you today?")
    speak_response("Welcome to the Improved Voice Support Bot. How can I assist you today?")
    
    while True:
        elapsed_time = time.time() - start_time

        if elapsed_time > 180:
            print("Bot: Connecting you to human support...")
            speak_response("Connecting you to human support...")
            break

        user_input = listen()

        if user_input:
            intent, confidence = get_intent(user_input)
            
            if confidence < 0.6:
                response = "I'm not quite sure what you mean. Could you please rephrase that?"
            else:
                response = get_response(intent)

            if intent == "exit":
                print(f"Bot: {response}")
                speak_response(response)
                break
            
            print(f"Bot: {response}")
            speak_response(response)

# Run the voice chatbot
if __name__ == "__main__":
    voice_chatbot()