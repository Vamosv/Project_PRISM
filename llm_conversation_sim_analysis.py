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

# Set the style for plots
plt.style.use('default')  # Use default style instead of ggplot to remove gray background
# Custom color palette with blue
blue_gray_palette = ['#1E88E5', '#777777']  # Blue for points, gray for lines
sns.set_palette(blue_gray_palette)
# Define colors
orange_color = '#FF7E29'  # Orange for polarity dots
blue_color = '#1E88E5'    # Blue for subjectivity dots
gray_color = '#777777'    # Gray for regression lines

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
with open('all_conversations_dump_two.json', 'r') as f:
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
            'topic': topic,
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
        if turn['model_name'] == 'User' and turn['turn'] > 0:
            extracted_score = extract_score(turn['content'])
            if extracted_score is not None:
                turn_data['score'] = extracted_score
                # Debug print for score extraction
                if turn_data['turn'] < 2:  # Only print for first few turns to avoid flooding output
                    print(f"Extracted score {extracted_score} from turn {turn_data['turn']}")
        
        # Calculate text metrics
        metrics = calculate_text_metrics(turn['content'])
        turn_data.update(metrics)
        
        all_turns.append(turn_data)

# Convert to DataFrame
df = pd.DataFrame(all_turns)

# Check scores
print(f"\nScore column stats before processing:")
if 'score' in df.columns:
    valid_scores = df[df['score'] > 0]['score']
    print(f"Number of valid scores (>0): {len(valid_scores)}")
    if len(valid_scores) > 0:
        print(f"Score range: {valid_scores.min()} to {valid_scores.max()}")
        print(f"Score mean: {valid_scores.mean():.2f}")
    else:
        print("No valid scores found!")
else:
    print("No score column found in the dataframe!")

# Create long format for metrics analysis
metrics_columns = ['text_length', 'flesch_reading_ease', 'flesch_kincaid_grade', 'tb_polarity', 'tb_subjectivity']
df_long_metrics = df[['conversation_id', 'turn', 'role', 'model_name', 'content', 'score'] + metrics_columns].copy()

# Analysis 1: Average score per turn
print("\nCalculating average scores per turn...")
avg_scores = df[df['score'] > 0].groupby('turn')['score'].agg(['mean', 'std', 'count']).reset_index()
print(avg_scores)

# Plotting average scores per turn
plt.figure(figsize=(10, 6))
plt.errorbar(avg_scores['turn'], avg_scores['mean'], yerr=avg_scores['std'], 
             fmt='o-', capsize=5, linewidth=2, markersize=8)
plt.xlabel('Turn Number', fontsize=14)
plt.ylabel('Average Score (0-100)', fontsize=14)
plt.title('Average User Satisfaction Score per Turn', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(avg_scores['turn'])
plt.ylim(0, 100)

# Add count information to the plot
for i, row in avg_scores.iterrows():
    plt.annotate(f"n={int(row['count'])}", 
                 (row['turn'], row['mean']),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')

plt.tight_layout()
plt.savefig('average_score_per_turn.png', dpi=300)
plt.show()

# Analysis 2: Correlation between psychological indicators and satisfaction
print("\nAnalyzing psychological indicators and satisfaction correlation...")

# Separate user and model data
df_user = df_long_metrics[df_long_metrics['model_name'] == 'User'].copy()
df_agent = df_long_metrics[df_long_metrics['model_name'] == 'Agent'].copy()

# Rename columns for clarity in joined dataframe
df_user.rename(columns={
    'turn': 'turn_user',
    'content': 'text_user',
    'flesch_reading_ease': 'flesch_reading_ease_user',
    'flesch_kincaid_grade': 'flesch_kincaid_grade_user',
    'tb_polarity': 'tb_polarity_user',
    'tb_subjectivity': 'tb_subjectivity_user'
}, inplace=True)

df_agent.rename(columns={
    'turn': 'turn_agent',
    'content': 'text_agent',
    'flesch_reading_ease': 'flesch_reading_ease_agent',
    'flesch_kincaid_grade': 'flesch_kincaid_grade_agent',
    'tb_polarity': 'tb_polarity_agent',
    'tb_subjectivity': 'tb_subjectivity_agent'
}, inplace=True)

# Create a column for user turns that follow agent turns
df_agent['turn_agent_plus1'] = df_agent['turn_agent'] + 1

# Merge USER turn t with AGENT turn t-1
df_merged = pd.merge(
    df_user,
    df_agent,
    left_on=['conversation_id', 'turn_user'], 
    right_on=['conversation_id', 'turn_agent_plus1'],
    how='inner'
)

print("Columns in merged dataframe:", df_merged.columns.tolist())
print(df_merged.head())

# Analyze correlation between agent's response sentiment and user's rating
print("\nChecking available columns in merged dataframe:")
print(df_merged.columns.tolist())

# Check if score column exists, if not rename appropriate column or create it
if 'score' not in df_merged.columns:
    if 'score_x' in df_merged.columns:
        df_merged['score'] = df_merged['score_x']
        print("Using 'score_x' as the score column")
    else:
        print("Warning: No score column found in the dataframe. Creating a placeholder.")
        # Create placeholder score for analysis - this should be adjusted based on actual data
        df_merged['score'] = np.nan

# Ensure we have the required columns for analysis
required_columns = ['score', 'tb_polarity_agent']
available_required = [col for col in required_columns if col in df_merged.columns]

df_analysis = df_merged.dropna(subset=available_required).copy()
print(f"DataFrame shape after dropping NaN values: {df_analysis.shape}")

# Only perform correlation analysis if we have score data
if 'score' in df_analysis.columns and not df_analysis['score'].isna().all():
    print("\nCorrelation between Agent's polarity and User's score:")
    correlation = df_analysis[available_required].corr()
    print(correlation)
    
    # Regression analysis
    print("\n--- Predicting User Score from Agent Polarity ---")
    model_score = smf.ols(formula="score ~ tb_polarity_agent", data=df_analysis).fit()
    print(model_score.summary())
    
    # Plot relationship between agent polarity and user score
    plt.figure(figsize=(10, 6))
    sns.regplot(x='tb_polarity_agent', y='score', data=df_analysis, scatter_kws={'alpha':0.5})
    plt.xlabel("Agent's Response Polarity", fontsize=14)
    plt.ylabel("User's Satisfaction Score", fontsize=14)
    plt.title("Relationship Between Agent's Response Sentiment and User Satisfaction", fontsize=16)
    plt.tight_layout()
    plt.savefig('sentiment_satisfaction_correlation.png', dpi=300)
    plt.show()
else:
    print("\nSkipping score correlation analysis due to missing score data")

# Analysis 3: User emotional inputs triggering agent responses
# Calculate changes in sentiment metrics from agent responses to user inputs
# Check if required columns exist before calculating changes
if all(col in df_analysis.columns for col in ['tb_polarity_user', 'tb_polarity_agent']):
    df_analysis['polarity_change'] = df_analysis['tb_polarity_user'] - df_analysis['tb_polarity_agent']
else:
    print("Warning: Cannot calculate polarity change due to missing columns")

if all(col in df_analysis.columns for col in ['tb_subjectivity_user', 'tb_subjectivity_agent']):
    df_analysis['subjectivity_change'] = df_analysis['tb_subjectivity_user'] - df_analysis['tb_subjectivity_agent']
else:
    print("Warning: Cannot calculate subjectivity change due to missing columns")

# Regression for user emotional inputs triggering corresponding indicators
print("\n--- User Emotional Inputs Triggering Agent Responses ---")
# Check if required columns exist before performing regression
if all(col in df_analysis.columns for col in ['tb_polarity_user', 'tb_polarity_agent']):
    formula_polarity = "tb_polarity_agent ~ tb_polarity_user"
    model_polarity = smf.ols(formula=formula_polarity, data=df_analysis).fit()
    print("Predicting Agent Polarity from User Polarity:")
    print(model_polarity.summary())

    # Plot emotional influence - stacked vertically with professional styling
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Polarity plot - ORANGE dots
    sns.regplot(x='tb_polarity_user', y='tb_polarity_agent', data=df_analysis, 
                scatter_kws={'alpha':0.6, 'color': orange_color}, 
                line_kws={'color': gray_color, 'lw': 2}, 
                ax=axs[0])
    axs[0].set_xlabel("User's Input Polarity", fontsize=12, fontweight='bold')
    axs[0].set_ylabel("Agent's Response Polarity", fontsize=12, fontweight='bold')
    axs[0].set_title("Emotional Influence: Polarity", fontsize=14, fontweight='bold')
    axs[0].spines['top'].set_visible(True)
    axs[0].spines['right'].set_visible(True)
    axs[0].spines['bottom'].set_visible(True)
    axs[0].spines['left'].set_visible(True)
    for spine in axs[0].spines.values():
        spine.set_linewidth(1.5)
    axs[0].grid(False)

    if all(col in df_analysis.columns for col in ['tb_subjectivity_user', 'tb_subjectivity_agent']):
        # Subjectivity plot - BLUE dots
        sns.regplot(x='tb_subjectivity_user', y='tb_subjectivity_agent', data=df_analysis, 
                    scatter_kws={'alpha':0.6, 'color': blue_color}, 
                    line_kws={'color': gray_color, 'lw': 2}, 
                    ax=axs[1])
        axs[1].set_xlabel("User's Input Subjectivity", fontsize=12, fontweight='bold')
        axs[1].set_ylabel("Agent's Response Subjectivity", fontsize=12, fontweight='bold')
        axs[1].set_title("Emotional Influence: Subjectivity", fontsize=14, fontweight='bold')
        axs[1].spines['top'].set_visible(True)
        axs[1].spines['right'].set_visible(True)
        axs[1].spines['bottom'].set_visible(True)
        axs[1].spines['left'].set_visible(True)
        for spine in axs[1].spines.values():
            spine.set_linewidth(1.5)
        axs[1].grid(False)

    plt.tight_layout()
    plt.savefig('emotional_influence.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("Skipping emotional influence analysis due to missing columns")

print("\nAnalysis complete! Results and visualizations have been generated.")