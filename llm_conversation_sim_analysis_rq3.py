# RQ3 Analysis: Demographic Moderation of Linguistic-Satisfaction and User-Model Interactions
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
# from statsmodels.graphics.interaction_plots import interaction_plot

# Set the style for plots
plt.style.use('ggplot')
sns.set_palette("viridis")

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
with open('all_conversations_dump_one.json', 'r') as f:
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
        
        # Calculate text metrics
        metrics = calculate_text_metrics(turn['content'])
        turn_data.update(metrics)
        
        all_turns.append(turn_data)

# Convert to DataFrame
df = pd.DataFrame(all_turns)

# Create long format for metrics analysis
metrics_columns = ['text_length', 'flesch_reading_ease', 'flesch_kincaid_grade', 'tb_polarity', 'tb_subjectivity']
df_long_metrics = df[['conversation_id', 'turn', 'role', 'model_name', 'content', 'score'] + metrics_columns + 
                    [col for col in df.columns if col.startswith('user_')]].copy()

# Prepare data for analysis
df_user = df_long_metrics[df_long_metrics['model_name'] == 'User'].copy()
df_agent = df_long_metrics[df_long_metrics['model_name'] == 'Agent'].copy()

# Rename columns for clarity
df_user.rename(columns={
    'turn': 'turn_user',
    'content': 'text_user',
    'flesch_reading_ease': 'flesch_reading_ease_user',
    'flesch_kincaid_grade': 'flesch_kincaid_grade_user',
    'tb_polarity': 'tb_polarity_user',
    'tb_subjectivity': 'tb_subjectivity_user',
    'text_length': 'text_length_user'
}, inplace=True)

df_agent.rename(columns={
    'turn': 'turn_agent',
    'content': 'text_agent',
    'flesch_reading_ease': 'flesch_reading_ease_agent',
    'flesch_kincaid_grade': 'flesch_kincaid_grade_agent',
    'tb_polarity': 'tb_polarity_agent',
    'tb_subjectivity': 'tb_subjectivity_agent',
    'text_length': 'text_length_agent'
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

# Ensure score is available
if 'score_x' in df_merged.columns and 'score' not in df_merged.columns:
    df_merged['score'] = df_merged['score_x']

# Print available demographic variables
demographic_cols = [col for col in df_merged.columns if col.startswith('user_')]
print("\nAvailable demographic variables:")
print(demographic_cols)

# ---------------------------------------------------------
# Analysis 1: How demographics moderate linguistic-satisfaction relationships
# ---------------------------------------------------------
print("\n\n=== ANALYSIS 1: DEMOGRAPHIC MODERATION OF LINGUISTIC-SATISFACTION RELATIONSHIPS ===\n")

# Select key demographics for analysis (if available) - using _x suffix
key_demographics = []
if 'user_education_x' in df_merged.columns:
    key_demographics.append('user_education_x')
if 'user_english_proficiency_x' in df_merged.columns:
    key_demographics.append('user_english_proficiency_x')
if 'user_age_x' in df_merged.columns:
    key_demographics.append('user_age_x')

if not key_demographics:
    print("No key demographic variables found for analysis")
else:
    print(f"Using demographics for analysis: {key_demographics}")

# Analyze how education moderates the relationship between complexity and satisfaction
if 'user_education_x' in key_demographics and 'score' in df_merged.columns:
    # Group education levels into fewer categories for clearer analysis
    education_mapping = {
        'Less than high school': 'Low',
        'High school graduate': 'Low',
        'Some college': 'Medium',
        'Vocational': 'Medium',
        'Bachelor\'s degree': 'High',
        'Master\'s degree': 'High',
        'Doctorate': 'High'
    }
    
    # Apply mapping
    if any(edu in education_mapping for edu in df_merged['user_education_x'].unique()):
        # Create a new column for simplified education levels, handling any values not in mapping
        df_merged['education_level'] = df_merged['user_education_x'].apply(
            lambda x: education_mapping.get(x, 'Other') if x in education_mapping else 'Other'
        )
        
        print("\n--- Education as Moderator between Linguistic Complexity and Satisfaction ---")
        print(f"Education levels found: {df_merged['user_education_x'].unique()}")
        print(f"Mapped education levels: {df_merged['education_level'].unique()}")
        
        # Run models for different education groups
        for level in sorted(df_merged['education_level'].unique()):
            if level == 'Other':
                continue  # Skip 'Other' category
                
            subset = df_merged[df_merged['education_level'] == level]
            
            if len(subset) > 10:  # Only analyze if enough data, reduced threshold
                # Model for reading ease
                if 'flesch_reading_ease_agent' in subset.columns:
                    try:
                        model_edu_fre = smf.ols(formula="score ~ flesch_reading_ease_agent", data=subset).fit()
                        print(f"\nEducation Level: {level} (n={len(subset)})")
                        print(f"Reading Ease -> Satisfaction: Coefficient = {model_edu_fre.params['flesch_reading_ease_agent']:.4f}, p-value = {model_edu_fre.pvalues['flesch_reading_ease_agent']:.4f}")
                    except Exception as e:
                        print(f"Could not analyze Reading Ease for Education Level {level}: {e}")
                
                # Model for grade level
                if 'flesch_kincaid_grade_agent' in subset.columns:
                    try:
                        model_edu_fkg = smf.ols(formula="score ~ flesch_kincaid_grade_agent", data=subset).fit()
                        print(f"Grade Level -> Satisfaction: Coefficient = {model_edu_fkg.params['flesch_kincaid_grade_agent']:.4f}, p-value = {model_edu_fkg.pvalues['flesch_kincaid_grade_agent']:.4f}")
                    except Exception as e:
                        print(f"Could not analyze Grade Level for Education Level {level}: {e}")
        
        # Create visualization of moderation effect
        if 'flesch_reading_ease_agent' in df_merged.columns:
            try:
                plt.figure(figsize=(12, 6))
                
                # Create categorical bins for easier visualization
                df_merged['fre_bin'] = pd.qcut(df_merged['flesch_reading_ease_agent'], 
                                            q=3, 
                                            labels=['Low', 'Medium', 'High'],
                                            duplicates='drop')
                
                # Plot the interaction
                ax = sns.boxplot(x='fre_bin', y='score', hue='education_level', data=df_merged)
                plt.xlabel('Agent Response Reading Ease', fontsize=12)
                plt.ylabel('User Satisfaction Score', fontsize=12)
                plt.title('Education Level Moderates the Effect of Reading Ease on Satisfaction', fontsize=14)
                plt.legend(title='Education Level')
                plt.tight_layout()
                plt.savefig('education_moderation_effect.png', dpi=300)
                plt.show()
            except Exception as e:
                print(f"Could not create education moderation plot: {e}")

# Analyze how English proficiency moderates the relationship
if 'user_english_proficiency_x' in key_demographics and 'score' in df_merged.columns:
    print("\n--- English Proficiency as Moderator between Linguistic Complexity and Satisfaction ---")
    print(f"English proficiency levels found: {df_merged['user_english_proficiency_x'].unique()}")
    
    for level in sorted(df_merged['user_english_proficiency_x'].unique()):
        subset = df_merged[df_merged['user_english_proficiency_x'] == level]
        
        if len(subset) > 10:  # Only analyze if enough data, reduced threshold
            # Model for reading ease
            if 'flesch_reading_ease_agent' in subset.columns:
                try:
                    model_eng_fre = smf.ols(formula="score ~ flesch_reading_ease_agent", data=subset).fit()
                    print(f"\nEnglish Proficiency: {level} (n={len(subset)})")
                    print(f"Reading Ease -> Satisfaction: Coefficient = {model_eng_fre.params['flesch_reading_ease_agent']:.4f}, p-value = {model_eng_fre.pvalues['flesch_reading_ease_agent']:.4f}")
                except Exception as e:
                    print(f"Could not analyze Reading Ease for English Proficiency {level}: {e}")
    
    # Create visualization for English proficiency moderation
    if 'flesch_kincaid_grade_agent' in df_merged.columns:
        try:
            plt.figure(figsize=(12, 6))
            
            # Create categorical bins for grade level
            df_merged['fkg_bin'] = pd.qcut(df_merged['flesch_kincaid_grade_agent'], 
                                        q=3, 
                                        labels=['Simple', 'Medium', 'Complex'],
                                        duplicates='drop')
            
            # Plot the interaction
            sns.boxplot(x='fkg_bin', y='score', hue='user_english_proficiency_x', data=df_merged)
            plt.xlabel('Agent Response Grade Level', fontsize=12)
            plt.ylabel('User Satisfaction Score', fontsize=12)
            plt.title('English Proficiency Moderates the Effect of Complexity on Satisfaction', fontsize=14)
            plt.legend(title='English Proficiency', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('english_proficiency_moderation.png', dpi=300)
            plt.show()
        except Exception as e:
            print(f"Could not create English proficiency moderation plot: {e}")

# ---------------------------------------------------------
# Analysis 2: How demographics influence triggering effects
# ---------------------------------------------------------
print("\n\n=== ANALYSIS 2: DEMOGRAPHIC INFLUENCE ON USER-MODEL INTERACTIONS ===\n")

# Analyze how user demographics affect the mirroring of linguistic features
if 'user_education_x' in key_demographics:
    print("\n--- Education's Influence on Linguistic Mirroring ---")
    
    # Calculate correlations for each education level
    for level in sorted(df_merged['user_education_x'].unique()):
        subset = df_merged[df_merged['user_education_x'] == level]
        
        if len(subset) > 10:  # Only analyze if enough data, reduced threshold
            # Text complexity mirroring
            if all(col in subset.columns for col in ['flesch_reading_ease_user', 'flesch_reading_ease_agent']):
                try:
                    formula = "flesch_reading_ease_agent ~ flesch_reading_ease_user"
                    model = smf.ols(formula=formula, data=subset).fit()
                    
                    print(f"\nEducation Level: {level} (n={len(subset)})")
                    print(f"Reading Ease Mirroring: Coefficient = {model.params['flesch_reading_ease_user']:.4f}, R-squared = {model.rsquared:.4f}, p-value = {model.pvalues['flesch_reading_ease_user']:.4f}")
                except Exception as e:
                    print(f"Could not analyze Reading Ease Mirroring for Education Level {level}: {e}")
            
            # Emotional mirroring (sentiment polarity)
            if all(col in subset.columns for col in ['tb_polarity_user', 'tb_polarity_agent']):
                try:
                    formula = "tb_polarity_agent ~ tb_polarity_user"
                    model = smf.ols(formula=formula, data=subset).fit()
                    
                    print(f"Emotional Mirroring: Coefficient = {model.params['tb_polarity_user']:.4f}, R-squared = {model.rsquared:.4f}, p-value = {model.pvalues['tb_polarity_user']:.4f}")
                except Exception as e:
                    print(f"Could not analyze Emotional Mirroring for Education Level {level}: {e}")
    
    # Create visualization of education's influence on mirroring
    if all(col in df_merged.columns for col in ['flesch_reading_ease_user', 'flesch_reading_ease_agent', 'user_education_x']):
        try:
            # Group education levels
            if 'education_level' in df_merged.columns:
                education_groups = sorted(df_merged['education_level'].unique())
                
                plt.figure(figsize=(15, 5))
                plot_index = 1
                
                for edu_level in education_groups:
                    if edu_level == 'Other':
                        continue
                        
                    subset = df_merged[df_merged['education_level'] == edu_level]
                    if len(subset) > 10:
                        plt.subplot(1, 3, plot_index)
                        sns.regplot(x='flesch_reading_ease_user', y='flesch_reading_ease_agent', 
                                data=subset, scatter_kws={'alpha':0.5})
                        
                        # Calculate correlation for the title
                        corr = subset[['flesch_reading_ease_user', 'flesch_reading_ease_agent']].corr().iloc[0,1]
                        
                        plt.xlabel("User's Reading Ease", fontsize=10)
                        plt.ylabel("Agent's Reading Ease", fontsize=10)
                        plt.title(f"{edu_level} Education (r={corr:.2f})", fontsize=12)
                        plot_index += 1
                
                if plot_index > 1:  # Only save if we have at least one plot
                    plt.tight_layout()
                    plt.savefig('education_mirroring_effect.png', dpi=300)
                    plt.show()
        except Exception as e:
            print(f"Could not create education mirroring plot: {e}")

# Analyze how English proficiency affects mirroring
if 'user_english_proficiency_x' in key_demographics:
    print("\n--- English Proficiency's Influence on Linguistic Mirroring ---")
    
    # Calculate correlations for each proficiency level
    for level in sorted(df_merged['user_english_proficiency_x'].unique()):
        subset = df_merged[df_merged['user_english_proficiency_x'] == level]
        
        if len(subset) > 10:  # Only analyze if enough data, reduced threshold
            # Calculate emotional and complexity mirroring coefficients
            if all(col in subset.columns for col in ['tb_polarity_user', 'tb_polarity_agent']):
                try:
                    formula = "tb_polarity_agent ~ tb_polarity_user"
                    model = smf.ols(formula=formula, data=subset).fit()
                    
                    print(f"\nEnglish Proficiency: {level} (n={len(subset)})")
                    print(f"Emotional Mirroring: Coefficient = {model.params['tb_polarity_user']:.4f}, R-squared = {model.rsquared:.4f}, p-value = {model.pvalues['tb_polarity_user']:.4f}")
                except Exception as e:
                    print(f"Could not analyze Emotional Mirroring for English Proficiency {level}: {e}")
            
            if all(col in subset.columns for col in ['flesch_kincaid_grade_user', 'flesch_kincaid_grade_agent']):
                try:
                    formula = "flesch_kincaid_grade_agent ~ flesch_kincaid_grade_user"
                    model = smf.ols(formula=formula, data=subset).fit()
                    
                    print(f"Grade Level Mirroring: Coefficient = {model.params['flesch_kincaid_grade_user']:.4f}, R-squared = {model.rsquared:.4f}, p-value = {model.pvalues['flesch_kincaid_grade_user']:.4f}")
                except Exception as e:
                    print(f"Could not analyze Grade Level Mirroring for English Proficiency {level}: {e}")
    
    # Visualize English proficiency's effect on mirroring
    if all(col in df_merged.columns for col in ['tb_polarity_user', 'tb_polarity_agent', 'user_english_proficiency_x']):
        try:
            plt.figure(figsize=(12, 8))
            
            proficiency_levels = sorted(df_merged['user_english_proficiency_x'].unique())
            colors = plt.cm.viridis(np.linspace(0, 1, len(proficiency_levels)))
            
            for i, level in enumerate(proficiency_levels):
                subset = df_merged[df_merged['user_english_proficiency_x'] == level]
                if len(subset) > 10:
                    plt.scatter(subset['tb_polarity_user'], subset['tb_polarity_agent'], 
                            alpha=0.6, label=level, color=colors[i])
                    
                    # Add regression line
                    m, b = np.polyfit(subset['tb_polarity_user'], subset['tb_polarity_agent'], 1)
                    x_range = np.linspace(subset['tb_polarity_user'].min(), subset['tb_polarity_user'].max(), 100)
                    plt.plot(x_range, m*x_range + b, color=colors[i])
            
            plt.xlabel("User's Sentiment Polarity", fontsize=12)
            plt.ylabel("Agent's Sentiment Polarity", fontsize=12)
            plt.title("English Proficiency's Effect on Emotional Mirroring", fontsize=14)
            plt.legend(title="English Proficiency")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('proficiency_emotional_mirroring.png', dpi=300)
            plt.show()
        except Exception as e:
            print(f"Could not create English proficiency mirroring plot: {e}")

# Add analysis for LM familiarity
if 'user_lm_familiarity_x' in df_merged.columns:
    print("\n--- LM Familiarity's Influence on User Satisfaction and Interactions ---")
    print(f"LM Familiarity levels found: {df_merged['user_lm_familiarity_x'].unique()}")
    
    # Group by LM familiarity and calculate mean scores
    familiarity_scores = df_merged.groupby('user_lm_familiarity_x')['score'].agg(['mean', 'std', 'count']).reset_index()
    print("\nAverage satisfaction scores by LM familiarity:")
    print(familiarity_scores)
    
    # Create a plot of average satisfaction by LM familiarity
    try:
        plt.figure(figsize=(10, 6))
        
        # Bar plot of mean scores
        ax = sns.barplot(x='user_lm_familiarity_x', y='mean', data=familiarity_scores, 
                         palette='viridis', errorbar=('ci', 95))
        
        # Add count annotations
        for i, row in enumerate(familiarity_scores.itertuples()):
            plt.text(i, row.mean, f"n={row.count}", ha='center', va='bottom')
        
        plt.xlabel("Language Model Familiarity", fontsize=12)
        plt.ylabel("Average Satisfaction Score", fontsize=12)
        plt.title("How LM Familiarity Affects User Satisfaction", fontsize=14)
        plt.ylim(0, 100)  # Set y-axis to full score range
        plt.tight_layout()
        plt.savefig('lm_familiarity_satisfaction.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"Could not create LM familiarity satisfaction plot: {e}")
    
    # Analyze mirroring by LM familiarity
    for level in sorted(df_merged['user_lm_familiarity_x'].unique()):
        subset = df_merged[df_merged['user_lm_familiarity_x'] == level]
        
        if len(subset) > 10:  # Only analyze if enough data
            # Emotional mirroring analysis
            if all(col in subset.columns for col in ['tb_polarity_user', 'tb_polarity_agent']):
                try:
                    formula = "tb_polarity_agent ~ tb_polarity_user"
                    model = smf.ols(formula=formula, data=subset).fit()
                    
                    print(f"\nLM Familiarity: {level} (n={len(subset)})")
                    print(f"Emotional Mirroring: Coefficient = {model.params['tb_polarity_user']:.4f}, R-squared = {model.rsquared:.4f}, p-value = {model.pvalues['tb_polarity_user']:.4f}")
                except Exception as e:
                    print(f"Could not analyze Emotional Mirroring for LM Familiarity {level}: {e}")
    
    # Create visualization of LM familiarity's influence on emotional mirroring
    try:
        plt.figure(figsize=(15, 8))
        familiarity_levels = sorted(df_merged['user_lm_familiarity_x'].unique())
        
        # Create a plot with one subplot per familiarity level
        num_levels = len(familiarity_levels)
        cols = min(3, num_levels)
        rows = (num_levels + cols - 1) // cols  # Ceiling division
        
        for i, level in enumerate(familiarity_levels):
            subset = df_merged[df_merged['user_lm_familiarity_x'] == level]
            if len(subset) > 10:
                plt.subplot(rows, cols, i+1)
                sns.regplot(x='tb_polarity_user', y='tb_polarity_agent', 
                          data=subset, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
                
                # Calculate correlation for the title
                corr = subset[['tb_polarity_user', 'tb_polarity_agent']].corr().iloc[0,1]
                
                plt.xlabel("User's Sentiment", fontsize=10)
                plt.ylabel("Agent's Sentiment", fontsize=10)
                plt.title(f"LM Familiarity: {level} (r={corr:.2f})", fontsize=12)
        
        plt.tight_layout()
        plt.savefig('lm_familiarity_emotional_mirroring.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"Could not create LM familiarity mirroring plot: {e}")

# Comprehensive interaction model - try both with education_x directly and education_level
print("\n--- Comprehensive Models: Demographics and Linguistic Feature Interactions ---")

# Try model with education_x
try:
    if 'user_education_x' in df_merged.columns and 'flesch_reading_ease_agent' in df_merged.columns:
        formula_comprehensive = "score ~ flesch_reading_ease_agent * C(user_education_x)"
        model_comprehensive = smf.ols(formula=formula_comprehensive, data=df_merged).fit()
        print("\nComprehensive Model (Education × Reading Ease):")
        print(model_comprehensive.summary())
except Exception as e:
    print(f"Could not run comprehensive model with education_x: {e}")

# Try model with english_proficiency_x
try:
    if 'user_english_proficiency_x' in df_merged.columns and 'flesch_reading_ease_agent' in df_merged.columns:
        formula_proficiency = "score ~ flesch_reading_ease_agent * C(user_english_proficiency_x)"
        model_proficiency = smf.ols(formula=formula_proficiency, data=df_merged).fit()
        print("\nComprehensive Model (English Proficiency × Reading Ease):")
        print(model_proficiency.summary())
except Exception as e:
    print(f"Could not run comprehensive model with english_proficiency_x: {e}")

# Try model with LM familiarity
try:
    if 'user_lm_familiarity_x' in df_merged.columns and 'tb_polarity_agent' in df_merged.columns:
        formula_lm = "score ~ tb_polarity_agent * C(user_lm_familiarity_x)"
        model_lm = smf.ols(formula=formula_lm, data=df_merged).fit()
        print("\nComprehensive Model (LM Familiarity × Agent Polarity):")
        print(model_lm.summary())
except Exception as e:
    print(f"Could not run comprehensive model with lm_familiarity_x: {e}")

print("\nAnalysis complete!")
