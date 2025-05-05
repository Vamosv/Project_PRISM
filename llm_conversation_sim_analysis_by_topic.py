# Do analysis here
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from tqdm import tqdm
import statsmodels.formula.api as smf
from textstat import flesch_reading_ease, flesch_kincaid_grade
from datetime import datetime
import os # Import os module for directory creation

# Set the style for plots
plt.style.use('default')  # Use default style instead of ggplot to remove gray background
# Custom color palette with blue
blue_gray_palette = ['#1E88E5', '#777777']  # Blue for points, gray for lines
sns.set_palette(blue_gray_palette)
# Define colors
orange_color = '#FF7E29'  # Orange for polarity dots
blue_color = '#1E88E5'    # Blue for subjectivity dots
gray_color = '#777777'    # Gray for regression lines

# Generate timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"analysis_output_{timestamp}"
os.makedirs(output_dir, exist_ok=True) # Create a directory for the output files

# Function to generate safe filenames
def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

def extract_score(content):
    """Extract score from the message content if present"""
    match = re.search(r'\{(\d+)\}', content)
    if match:
        return int(match.group(1))
    return None

def calculate_text_metrics(text):
    """Calculate various text metrics for a given text"""
    # Remove score indicators like {85}
    cleaned_text = re.sub(r'\{\d+\}', '', text).strip()
    
    # Calculate TextBlob sentiment metrics
    blob = TextBlob(cleaned_text)
    tb_polarity = blob.sentiment.polarity
    tb_subjectivity = blob.sentiment.subjectivity
    
    # Calculate readability metrics
    try:
        fre = flesch_reading_ease(cleaned_text)
        fkg = flesch_kincaid_grade(cleaned_text)
    except:
        fre = np.nan
        fkg = np.nan
    
    return {
        'text_length': len(cleaned_text),
        'flesch_reading_ease': fre,
        'flesch_kincaid_grade': fkg,
        'tb_polarity': tb_polarity,
        'tb_subjectivity': tb_subjectivity
    }

# Load the conversation data
print("Loading conversation data...")
# Corrected filename based on user feedback
with open('all_conversations_dump_matched_frequencies.json', 'r') as f:
    conversations_data = json.load(f)

# Create a list to hold all conversation turns
all_turns = []

# Process each conversation
print("Processing conversations...")
for conversation in tqdm(conversations_data):
    conv_id = conversation['conversation_id']
    topic = conversation['topic']
    
    # Extract user personality traits
    user_personality = conversation['personality_profiles']['User']
    
    # Process each turn in the conversation
    for turn in conversation['conversation_history']:
        turn_data = {
            'conversation_id': conv_id,
            'topic': topic, # Make sure topic is added here
            'turn': turn['turn'],
            'role': turn['role'],
            'model_name': turn['model_name'],
            'content': turn['content'],
            'score': turn['score']
        }
        
        # Add user personality traits
        for trait, value in user_personality.items():
            turn_data[f'user_{trait}'] = value
        
        # Extract score from content if "User" is scoring "Agent"
        # Adjusted logic: score is given by User in turn N based on Agent's response in turn N-1
        # So, we look for scores in User turns. The score applies to the preceding Agent turn.
        if turn['model_name'] == 'User':
             extracted_score = extract_score(turn['content'])
             if extracted_score is not None:
                 turn_data['extracted_score_in_user_turn'] = extracted_score # Temporarily store it

        # Calculate text metrics
        metrics = calculate_text_metrics(turn['content'])
        turn_data.update(metrics)
        
        all_turns.append(turn_data)

# Convert to DataFrame
df = pd.DataFrame(all_turns)

# Shift the extracted score to align with the Agent's turn it's rating
df['score_for_agent_turn'] = df.groupby('conversation_id')['extracted_score_in_user_turn'].shift(-1)
# Drop rows where the score is irrelevant (e.g., Agent turns, User's first turn)
df['score'] = np.where(df['model_name'] == 'Agent', df['score_for_agent_turn'], np.nan)
# Drop temporary columns
df.drop(columns=['extracted_score_in_user_turn', 'score_for_agent_turn'], inplace=True)

# Check scores
print(f"\nScore column stats after alignment:")
valid_scores = df['score'].dropna()
print(f"Number of valid scores: {len(valid_scores)}")
if len(valid_scores) > 0:
    print(f"Score range: {valid_scores.min()} to {valid_scores.max()}")
    print(f"Score mean: {valid_scores.mean():.2f}")
else:
    print("No valid scores found!")


# Analysis 1: Average score per turn (overall)
print("\nCalculating average scores per turn (overall)...")
# Filter for Agent turns which now have the score assigned to them
avg_scores_overall = df[df['model_name'] == 'Agent'].groupby('turn')['score'].agg(['mean', 'std', 'count']).reset_index()
print(avg_scores_overall)

# Plotting average scores per turn (overall)
plt.figure(figsize=(10, 6))
plt.errorbar(avg_scores_overall['turn'], avg_scores_overall['mean'], yerr=avg_scores_overall['std'],
             fmt='o-', capsize=5, linewidth=2, markersize=8)
plt.xlabel('Turn Number (Agent Response)', fontsize=14)
plt.ylabel('Average Score (0-100)', fontsize=14)
plt.title('Overall Average User Satisfaction Score per Turn', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(avg_scores_overall['turn'])
plt.ylim(0, 100)
for i, row in avg_scores_overall.iterrows():
    plt.annotate(f"n={int(row['count'])}", (row['turn'], row['mean']), textcoords="offset points", xytext=(0,10), ha='center')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'average_score_per_turn_overall_{timestamp}.png'), dpi=300)
plt.show()

# Analysis 1b: Average score per topic
print("\nCalculating average scores per topic...")
avg_scores_topic = df[df['model_name'] == 'Agent'].groupby('topic')['score'].agg(['mean', 'std', 'count']).reset_index().sort_values('mean', ascending=False)
print(avg_scores_topic)

# Plotting average scores per topic
plt.figure(figsize=(12, 8)) # Increased size for better topic readability
sns.barplot(x='mean', y='topic', data=avg_scores_topic, palette='viridis')
# Add error bars manually if needed, sns.barplot shows confidence intervals by default
plt.errorbar(x=avg_scores_topic['mean'], y=avg_scores_topic['topic'], xerr=avg_scores_topic['std'], fmt='none', c='black', capsize=3)
plt.xlabel('Average Score (0-100)', fontsize=14)
plt.ylabel('Topic', fontsize=14)
plt.title('Average User Satisfaction Score per Topic', fontsize=16)
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'average_score_per_topic_{timestamp}.png'), dpi=300)
plt.show()


# Analysis 2: Prepare data for regression including topic
print("\nPreparing data for regression analysis...")
# We need user's turn N and agent's turn N-1. Let's realign the main df.
df_user = df[df['model_name'] == 'User'].copy()
df_agent = df[df['model_name'] == 'Agent'].copy()

# Add a turn column to merge on
df_user['merge_turn'] = df_user['turn']
df_agent['merge_turn'] = df_agent['turn'] + 1 # Agent turn N-1 should merge with User turn N

df_merged_reg = pd.merge(
    df_user,
    df_agent,
    on=['conversation_id', 'merge_turn'],
    suffixes=('_user', '_agent'), # Suffixes clarify origin
    how='inner'
)

# The score is now correctly associated with the agent's turn data (df_agent part)
# Use score_agent which corresponds to the rating of the agent's response
df_analysis = df_merged_reg.rename(columns={'score_agent': 'score'}).copy()
df_analysis = df_analysis.dropna(subset=['score', 'tb_polarity_agent', 'topic_agent']).copy() # Ensure required columns and topic are not null

print(f"Shape of data for regression: {df_analysis.shape}")
if df_analysis.empty:
    print("No data available for regression after processing. Exiting analysis.")
    exit()

print("\n--- Regression: Predicting User Score from Agent Polarity + Topic ---")
try:
    # Ensure topic is treated as categorical
    model_score_topic = smf.ols(formula="score ~ tb_polarity_agent + C(topic_agent)", data=df_analysis).fit()
    print(model_score_topic.summary())

    # Plot overall relationship (same as before, topic effect is in the model summary)
    plt.figure(figsize=(10, 6))
    sns.regplot(x='tb_polarity_agent', y='score', data=df_analysis, scatter_kws={'alpha':0.5})
    plt.xlabel("Agent's Response Polarity", fontsize=14)
    plt.ylabel("User's Satisfaction Score", fontsize=14)
    plt.title("Overall Relationship: Agent Sentiment vs. User Satisfaction", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sentiment_satisfaction_correlation_overall_{timestamp}.png'), dpi=300)
    plt.show()

except Exception as e:
    print(f"Could not run score regression model: {e}")


# Analysis 3: User emotional inputs triggering agent responses, considering topic
print("\n--- Regression: Predicting Agent Sentiment from User Sentiment + Topic ---")
df_analysis = df_analysis.dropna(subset=[
    'tb_polarity_agent', 'tb_polarity_user', 'tb_subjectivity_agent', 'tb_subjectivity_user', 'topic_agent'
]).copy()

if df_analysis.empty:
     print("No data available for sentiment influence regression after processing.")
else:
    try:
        # Polarity
        formula_polarity_topic = "tb_polarity_agent ~ tb_polarity_user + C(topic_agent)"
        model_polarity_topic = smf.ols(formula=formula_polarity_topic, data=df_analysis).fit()
        print("\nPredicting Agent Polarity from User Polarity + Topic:")
        print(model_polarity_topic.summary())

        # Subjectivity
        formula_subjectivity_topic = "tb_subjectivity_agent ~ tb_subjectivity_user + C(topic_agent)"
        model_subjectivity_topic = smf.ols(formula=formula_subjectivity_topic, data=df_analysis).fit()
        print("\nPredicting Agent Subjectivity from User Subjectivity + Topic:")
        print(model_subjectivity_topic.summary())

        # Plot overall emotional influence (topic effects captured in regression summary)
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        # Polarity plot - ORANGE dots
        sns.regplot(x='tb_polarity_user', y='tb_polarity_agent', data=df_analysis,
                    scatter_kws={'alpha':0.6, 'color': orange_color},
                    line_kws={'color': gray_color, 'lw': 2},
                    ax=axs[0])
        axs[0].set_xlabel("User's Input Polarity", fontsize=12, fontweight='bold')
        axs[0].set_ylabel("Agent's Response Polarity", fontsize=12, fontweight='bold')
        axs[0].set_title("Overall Emotional Influence: Polarity", fontsize=14, fontweight='bold')
        # Simplified styling
        axs[0].grid(True, alpha=0.3)


        # Subjectivity plot - BLUE dots
        sns.regplot(x='tb_subjectivity_user', y='tb_subjectivity_agent', data=df_analysis,
                    scatter_kws={'alpha':0.6, 'color': blue_color},
                    line_kws={'color': gray_color, 'lw': 2},
                    ax=axs[1])
        axs[1].set_xlabel("User's Input Subjectivity", fontsize=12, fontweight='bold')
        axs[1].set_ylabel("Agent's Response Subjectivity", fontsize=12, fontweight='bold')
        axs[1].set_title("Overall Emotional Influence: Subjectivity", fontsize=14, fontweight='bold')
        # Simplified styling
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'emotional_influence_overall_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # Analysis 3b: Average Agent Sentiment per Topic
        print("\nCalculating average agent sentiment per topic...")
        avg_sentiment_topic = df_analysis.groupby('topic_agent')[['tb_polarity_agent', 'tb_subjectivity_agent']].agg(['mean', 'std', 'count'])
        print(avg_sentiment_topic)

        # Plotting average agent sentiment per topic
        avg_sentiment_plot_data = avg_sentiment_topic.reset_index()
        # Use multi-level columns directly if needed, or flatten
        avg_sentiment_plot_data.columns = ['_'.join(col).strip('_') for col in avg_sentiment_plot_data.columns.values]

        fig, axs = plt.subplots(2, 1, figsize=(12, 16)) # Increased size

        # Polarity by Topic
        sns.barplot(x='tb_polarity_agent_mean', y='topic_agent', data=avg_sentiment_plot_data.sort_values('tb_polarity_agent_mean', ascending=False), palette='coolwarm', ax=axs[0])
        axs[0].errorbar(x=avg_sentiment_plot_data['tb_polarity_agent_mean'], y=avg_sentiment_plot_data['topic_agent'], xerr=avg_sentiment_plot_data['tb_polarity_agent_std'], fmt='none', c='black', capsize=3)
        axs[0].set_xlabel('Average Agent Polarity', fontsize=14)
        axs[0].set_ylabel('Topic', fontsize=14)
        axs[0].set_title('Average Agent Response Polarity per Topic', fontsize=16)
        axs[0].axvline(0, color='grey', linestyle='--') # Add line at zero polarity

        # Subjectivity by Topic
        sns.barplot(x='tb_subjectivity_agent_mean', y='topic_agent', data=avg_sentiment_plot_data.sort_values('tb_subjectivity_agent_mean', ascending=False), palette='viridis', ax=axs[1])
        axs[1].errorbar(x=avg_sentiment_plot_data['tb_subjectivity_agent_mean'], y=avg_sentiment_plot_data['topic_agent'], xerr=avg_sentiment_plot_data['tb_subjectivity_agent_std'], fmt='none', c='black', capsize=3)
        axs[1].set_xlabel('Average Agent Subjectivity', fontsize=14)
        axs[1].set_ylabel('Topic', fontsize=14)
        axs[1].set_title('Average Agent Response Subjectivity per Topic', fontsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'average_agent_sentiment_per_topic_{timestamp}.png'), dpi=300)
        plt.show()


    except Exception as e:
        print(f"Could not run sentiment influence regression model: {e}")


print(f"\nAnalysis complete! Results and visualizations saved in '{output_dir}'.")