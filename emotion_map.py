"""Maps the 28 GoEmotions labels to EchoBuddy's wellbeing groups."""
import re

GROUPS = {
    "sad": ["sadness", "grief", "disappointment", "remorse"],
    "anxiety": ["fear", "nervousness"],
    "anger": ["anger", "annoyance", "disapproval", "disgust"],
    "happy": ["joy", "amusement", "excitement", "gratitude", "love",
              "optimism", "pride", "admiration", "relief"],
}
NEGATIVE = ["sad", "anxiety", "anger"]

# Safety net: serious statements always trigger an alert, regardless of the model.
CRISIS_RE = re.compile(
    r"(kill myself|end my life|want to die|suicide|hurt myself|no reason to live)",
    re.I)
