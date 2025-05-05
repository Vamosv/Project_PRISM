import pandas as pd
import json
from groq import Groq
from tqdm import tqdm

# Read the data
df = pd.read_csv('merged_data.csv')

# Initialize Groq client
client = Groq(
    api_key="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",
)

# Define the topics
TOPICS = [
    "Agriculture",
    "Architecture",
    "Biology",
    "Chemistry",
    "Climate+Weather",
    "ComplexNetworks",
    "ComputerNetworks",
    "CyberSecurity",
    "DataChallenges",
    "EarthScience",
    "Economics",
    "Education",
    "Energy",
    "Entertainment",
    "Finance",
    "GIS",
    "Government",
    "Healthcare",
    "ImageProcessing",
    "MachineLearning",
    "Museums",
    "NaturalLanguage",
    "Neuroscience",
    "Physics",
    "ProstateCancer",
    "Psychology+Cognition",
    "PublicDomains",
    "SearchEngines",
    "SocialNetworks",
    "SocialSciences",
    "Software",
    "Sports",
    "TimeSeries",
    "Transportation",
    "eSports",
    "Complementary Collections"
]

def classify_conversation(conversation):
    prompt = f"""Given the following conversation, classify it into one of these topics: {', '.join(TOPICS)}.
    If none of these topics fit well, respond with "Other".
    Only respond with the exact topic name or "Other".
    
    Conversation:
    {conversation}
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.1,
            max_tokens=50
        )
        topic = response.choices[0].message.content.strip()
        # If the response isn't in our topics list, mark it as "Other"
        if topic not in TOPICS:
            topic = "Other"
        return topic
    except Exception as e:
        print(f"Error classifying conversation: {e}")
        return "Error"

# Add topic classification with progress bar
print("Classifying conversations...")
tqdm.pandas(desc="Classifying conversations")
df['topic'] = df['conversation_history'].progress_apply(classify_conversation)

# Calculate topic frequencies
topic_frequencies = df['topic'].value_counts().to_dict()

# Save topic frequencies to JSON
with open('topic_frequencies.json', 'w') as f:
    json.dump(topic_frequencies, f, indent=4)

# Save updated dataframe
df.to_csv('merged_data_with_topics.csv', index=False)

print("Classification complete. Results saved to 'merged_data_with_topics.csv' and 'topic_frequencies.json'")