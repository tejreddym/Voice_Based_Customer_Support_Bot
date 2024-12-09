Here are the step-by-step instructions from starting to set up the project, create a virtual environment, activate it, and install the necessary modules for your voice-based AI bot project on macOS, Windows, and Linux.

---

# Instructions to Set Up the Project

## Prerequisites

Make sure you have the following installed on your system:
- **Python 3.6 or higher**
- **pip** (Python package installer)
- **virtualenv** (optional but recommended for isolated environments)

## 1. Clone the Repository (if applicable)

If you are working from a remote repository, clone the project by running:

```bash
git clone https://github.com/your-username/your-repo.git
```

Otherwise, navigate to your project folder.

```bash
cd /path/to/your/project
```

## 2. Create a Virtual Environment

### On macOS / Linux:

Run the following command to create a virtual environment called `venv`:

```bash
python3 -m venv venv
```

### On Windows:

Run the following command to create the virtual environment:

```bash
python -m venv venv
```

This command will create a directory named `venv` which will hold an isolated Python environment.

## 3. Activate the Virtual Environment

### On macOS / Linux:

Activate the virtual environment by running the following command in your terminal:

```bash
source venv/bin/activate
```

### On Windows:

Activate the virtual environment by running this command in the command prompt:

```bash
venv\Scripts\activate
```

When activated, your terminal or command prompt will show the `(venv)` prefix indicating you are now inside the virtual environment.

## 4. Install the Required Dependencies

With the virtual environment activated, install the necessary Python packages using `pip`:

```bash
pip install spacy pyttsx3 SpeechRecognition scikit-learn
```

These are the primary modules your project requires:
- **spaCy** for natural language processing (NLP)
- **pyttsx3** for text-to-speech (TTS)
- **SpeechRecognition** for capturing and recognizing speech
- **scikit-learn** for building the text classification model

Next, download the spaCy English language model:

```bash
python -m spacy download en_core_web_sm
```

This will ensure spaCy has the necessary language model to process English text.

## 5. Run the Application

Once all dependencies are installed, you can run your Python script by executing the following command:

```bash
python main.py
```

The bot will start listening for user queries via the microphone, classify the intent, and provide responses via text-to-speech. To exit the bot, you can say "exit," "quit," or "bye."

---

Following these instructions will guide you from setting up your environment to running your voice-based AI bot.