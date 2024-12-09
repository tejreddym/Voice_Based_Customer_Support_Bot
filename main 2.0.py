import spacy
import pyttsx3
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import time
import numpy as np

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize speech engine (unchanged)
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Define the data (unchanged)
data = [
    # Password Reset Intent
    ("how can i reset my password", "reset_password"),
    ("i forgot my password", "reset_password"),
    ("i need to change my password", "reset_password"),
    ("can you help me reset my password", "reset_password"),
    ("how do i reset my account password", "reset_password"),
    ("i can't remember my password", "reset_password"),
    ("i want to reset my password", "reset_password"),
    ("how do i change my password", "reset_password"),
    ("reset my password please", "reset_password"),
    ("i can't log in, need to reset my password", "reset_password"),
    ("help me with password recovery", "reset_password"),
    
    # Order Status Intent
    ("what is the status of my order", "order_status"),
    ("where is my order", "order_status"),
    ("track my order", "order_status"),
    ("i want to check my order status", "order_status"),
    ("can you tell me the status of my order", "order_status"),
    ("how is my order doing", "order_status"),
    ("can you track my order for me", "order_status"),
    ("i need an update on my order", "order_status"),
    ("check the status of my order", "order_status"),
    ("what's happening with my order", "order_status"),
    ("any update on my order", "order_status"),
    ("i want to know where my order is", "order_status"),

    # Contact Support Intent
    ("how can i contact support", "contact_support"),
    ("i need help", "contact_support"),
    ("how do i reach customer service", "contact_support"),
    ("can you give me support contact details", "contact_support"),
    ("i need to talk to customer support", "contact_support"),
    ("how do i get in touch with support", "contact_support"),
    ("give me customer support information", "contact_support"),
    ("i need assistance, how do i contact support", "contact_support"),
    ("what's the support phone number", "contact_support"),
    ("how do i contact someone for help", "contact_support"),
    ("how can i reach out to support", "contact_support"),
    ("how do i send a message to support", "contact_support"),
    ("can i call support", "contact_support"),

    # New Intents: Shipping Details Intent
    ("where can i find my shipping information", "shipping_details"),
    ("what's the shipping info for my order", "shipping_details"),
    ("tell me the shipping details", "shipping_details"),
    ("can you give me the tracking number", "shipping_details"),
    ("where is my order being shipped from", "shipping_details"),
    ("i need my tracking number", "shipping_details"),
    ("where can i track my shipment", "shipping_details"),
    ("can you provide the shipping status", "shipping_details"),

    # Cancellation Intent
    ("how can i cancel my order", "cancel_order"),
    ("i want to cancel my order", "cancel_order"),
    ("can you help me cancel my order", "cancel_order"),
    ("please cancel my order", "cancel_order"),
    ("i need to cancel my order", "cancel_order"),
    ("cancel my order immediately", "cancel_order"),
    ("can i still cancel my order", "cancel_order"),
    ("how do i request an order cancellation", "cancel_order"),

    # Returns and Refunds Intent
    ("how can i return my order", "return_order"),
    ("i want to return my order", "return_order"),
    ("how do i get a refund", "return_order"),
    ("can you help me with a return", "return_order"),
    ("i need to return my purchase", "return_order"),
    ("what is the return process", "return_order"),
    ("how do i initiate a return", "return_order"),
    ("can i get a refund for my order", "return_order"),
    ("how long does it take to get a refund", "return_order"),
    ("what's the refund policy", "return_order"),

    # Payment Issues Intent
    ("i have a problem with my payment", "payment_issue"),
    ("my payment didn't go through", "payment_issue"),
    ("i was charged incorrectly", "payment_issue"),
    ("why was my card declined", "payment_issue"),
    ("there was an error with my payment", "payment_issue"),
    ("can you check my payment status", "payment_issue"),
    ("i need help with my payment", "payment_issue"),
    ("i was charged twice", "payment_issue"),
    ("how can i update my payment details", "payment_issue"),
    
    # Exit Intent
    ("thank you", "exit"),
    ("bye", "exit"),
    ("goodbye", "exit"),
    ("thanks", "exit")
]

# Text preprocessing function
def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_texts, train_labels = zip(*train_data)
test_texts, test_labels = zip(*test_data)

# Preprocess the texts
train_texts_processed = [preprocess_text(text) for text in train_texts]
test_texts_processed = [preprocess_text(text) for text in test_texts]

# Create an improved text classification pipeline
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_features=1000),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

# Perform grid search for hyperparameter tuning
param_grid = {
    'tfidfvectorizer__max_features': [500, 1000, 1500],
    'randomforestclassifier__n_estimators': [50, 100, 150],
    'randomforestclassifier__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(train_texts_processed, train_labels)

# Use the best model
best_model = grid_search.best_estimator_

# Test the model
predicted_labels = best_model.predict(test_texts_processed)
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(test_labels, predicted_labels))

# Function to get the intent with confidence score
def get_intent(text):
    processed_text = preprocess_text(text)
    probabilities = best_model.predict_proba([processed_text])[0]
    intent = best_model.classes_[np.argmax(probabilities)]
    confidence = np.max(probabilities)
    return intent, confidence

# Function to get a response based on intent (unchanged)
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

# Voice output using pyttsx3 (unchanged)
def speak_response(response):
    engine.say(response)
    engine.runAndWait()

# Function to listen to the user input using speech recognition (unchanged)
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for ambient noise
        try:
            # Timeout after 3 seconds of silence, or listen for a max of 7 seconds
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

# Updated voice-enabled chatbot function with confidence threshold
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
            
            if confidence < 0.6:  # Adjust this threshold as needed
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