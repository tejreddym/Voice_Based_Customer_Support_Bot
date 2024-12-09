import spacy
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import VotingClassifier

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Define the data
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
    ("thanks", "exit"),
    ("I want to know my shipping details", "shipping_details"),
    ("Can you provide shipping information?", "shipping_details"),
    ("When will my order be delivered?", "shipping_details"),
    ("What's the estimated delivery date?", "shipping_details"),
    ("Thanks for your help", "exit"),
    ("That's all I needed", "exit"),
    ("I'm done, thank you", "exit"),
    ("You've been very helpful, goodbye", "exit"),
]

# Text preprocessing function
def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=[d[1] for d in data])
train_texts, train_labels = zip(*train_data)
test_texts, test_labels = zip(*test_data)

# Preprocess the texts
train_texts_processed = [preprocess_text(text) for text in train_texts]
test_texts_processed = [preprocess_text(text) for text in test_texts]

# Create advanced TF-IDF vectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=2000, use_idf=True, smooth_idf=True, sublinear_tf=True)

# Create multiple classifiers
rf = RandomForestClassifier(n_estimators=200, random_state=42)
svm = SVC(probability=True, random_state=42)
nb = MultinomialNB()
gb = GradientBoostingClassifier(n_estimators=200, random_state=42)

# Create pipelines for each classifier
pipelines = {
    'RandomForest': Pipeline([('tfidf', tfidf), ('select', SelectKBest(chi2, k='all')), ('clf', rf)]),
    'SVM': Pipeline([('tfidf', tfidf), ('select', SelectKBest(chi2, k='all')), ('clf', svm)]),
    'NaiveBayes': Pipeline([('tfidf', tfidf), ('select', SelectKBest(chi2, k='all')), ('clf', nb)]),
    'GradientBoosting': Pipeline([('tfidf', tfidf), ('select', SelectKBest(chi2, k='all')), ('clf', gb)])
}

# Train and evaluate each model
best_accuracy = 0
best_model = None
best_model_name = None

for name, pipeline in pipelines.items():
    print(f"\nTraining {name}...")
    pipeline.fit(train_texts_processed, train_labels)
    predicted_labels = pipeline.predict(test_texts_processed)
    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_labels))
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline
        best_model_name = name

print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy * 100:.2f}%")

# Create and train ensemble model
ensemble = VotingClassifier(
    estimators=[(name, pipeline) for name, pipeline in pipelines.items()],
    voting='soft'
)
ensemble.fit(train_texts_processed, train_labels)
ensemble_predicted_labels = ensemble.predict(test_texts_processed)
ensemble_accuracy = accuracy_score(test_labels, ensemble_predicted_labels)

print("\nEnsemble Model Results:")
print(f"Accuracy: {ensemble_accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(test_labels, ensemble_predicted_labels))

# Compare ensemble with best individual model
if ensemble_accuracy > best_accuracy:
    best_model = ensemble
    best_model_name = "Ensemble"
    best_accuracy = ensemble_accuracy

# Error analysis
misclassified = [(true, pred, text) for true, pred, text in zip(test_labels, best_model.predict(test_texts_processed), test_texts) if true != pred]
print("\nMisclassified examples:")
for true, pred, text in misclassified:
    print(f"True: {true}, Predicted: {pred}, Text: {text}")

# Save the best model
joblib.dump(best_model, 'intent_classification_model.joblib')
print(f"\nBest model ({best_model_name}) saved as 'intent_classification_model.joblib'")