import os
import uuid
import random
import json
from typing import List
import re # Import the regular expression module
from tqdm import tqdm  # <-- Add this line
import time # Import the time module
import pandas as pd

# Import Groq and initialize the client using your environment variable for the API key.
from groq import Groq, APITimeoutError

client = Groq(
    api_key="YOUR API KEY HERE",
)

# =============================================================================
# Personality Profile Module (with Agenda)
# =============================================================================


class PersonalityProfile:
    def __init__(self,
                 lm_familiarity: str = "Unknown",
                 lm_frequency_use: str = "Unknown",
                 age: str = "Unknown",
                 gender: str = "Unknown",
                 employment_status: str = "Unknown",
                 education: str = "Unknown",
                 marital_status: str = "Unknown",
                 english_proficiency: str = "Unknown",
                 study_locale: str = "Unknown",
                 religion: str = "Unknown",
                 ethnicity: str = "Unknown",
                 location: str = "Unknown",
                 agenda: str = "No specific agenda"):
        self.lm_familiarity = lm_familiarity
        self.lm_frequency_use = lm_frequency_use
        self.age = age
        self.gender = gender
        self.employment_status = employment_status
        self.education = education
        self.marital_status = marital_status
        self.english_proficiency = english_proficiency
        self.study_locale = study_locale
        self.religion = religion
        self.ethnicity = ethnicity
        self.location = location
        self.agenda = agenda

    def to_dict(self) -> dict:
        return {
            "lm_familiarity": self.lm_familiarity,
            "lm_frequency_use": self.lm_frequency_use,
            "age": self.age,
            "gender": self.gender,
            "employment_status": self.employment_status,
            "education": self.education,
            "marital_status": self.marital_status,
            "english_proficiency": self.english_proficiency,
            "study_locale": self.study_locale,
            "religion": self.religion,
            "ethnicity": self.ethnicity,
            "location": self.location,
            "agenda": self.agenda,
        }
    def __repr__(self):
        parts = []
        if self.age != "Unknown":
            parts.append(f"I am a {self.age}")
        if self.gender != "Unknown":
            parts.append(f"{self.gender}")
        if parts:
            parts[0] = "I am a " + parts[0]
            parts = [" ".join(parts) + " person"]
        if self.education != "Unknown":
            parts.append(f"My education level is {self.education}")
        if self.employment_status != "Unknown":
            parts.append(f"I am {self.employment_status}")
        if self.marital_status != "Unknown":
            parts.append(f"I am {self.marital_status}")
        if self.english_proficiency != "Unknown":
            parts.append(f"I am {self.english_proficiency}")
        if self.location != "Unknown":
            parts.append(f"I live in {self.location}")
        if self.study_locale != "Unknown":
            parts.append(f"studied in {self.study_locale}")
        if self.ethnicity != "Unknown":
            parts.append(f"My ethnicity is {self.ethnicity}")
        if self.religion != "Unknown":
            parts.append(f"my religion is {self.religion}")
        if self.lm_familiarity != "Unknown":
            parts.append(f"Regarding language models, I am {self.lm_familiarity} with them")
        if self.lm_frequency_use != "Unknown":
            parts.append(f"use them {self.lm_frequency_use}")
        parts.append(f"My agenda is: {self.agenda}")
        
        return ". ".join(parts)
        
# Define buckets of options for random personality generation
PERSONALITY_OPTIONS = {
    "lm_familiarity": ["Very familiar", "Somewhat familiar", "Not familiar", "Never heard of them"],
    "lm_frequency_use": ["Daily", "Weekly", "Once per month", "Less than once per month", "Never"],
    "age": ["18-24 years old", "25-34 years old", "35-44 years old", "45-54 years old", "55-64 years old", "65+ years old"],
    "gender": ["Male", "Female", "Non-binary", "Prefer not to say"],
    "employment_status": ["Working full-time", "Working part-time", "Self-employed", "Unemployed", "Student", "Retired", "Homemaker"],
    "education": ["Less than high school", "High school graduate", "Some college", "Vocational", "Bachelor's degree", "Master's degree", "Doctorate"],
    "marital_status": ["Never been married", "Married", "Divorced", "Widowed", "Separated"],
    "english_proficiency": ["Native speaker", "Fluent", "Intermediate", "Beginner"],
    "study_locale": ["USA", "Canada", "UK", "Australia", "India", "Germany", "France", "Other Europe", "Asia", "Africa", "South America"],
    "religion": ["Christianity", "Islam", "Hinduism", "Buddhism", "Judaism", "No Affiliation", "Agnostic", "Atheist", "Other"],
    "ethnicity": ["White", "Black or African American", "Hispanic or Latino", "Asian", "Native American or Alaska Native", "Native Hawaiian or Other Pacific Islander", "Middle Eastern or North African", "Mixed Race", "Prefer not to say"],
    "location": ["Northern America", "Europe", "Asia", "South America", "Africa", "Oceania"],
    "agenda": [
        "I am here to learn about the topic.",
        "I want to understand practical applications of this subject.",
        "I aim to have a friendly and informative chat.",
        "I'm curious to explore creative possibilities within the topic.",
        "I want to share my personal experiences and learn from others.",
        "I hope to gain insights that could help me in my daily life.",
        "I'm interested in understanding different perspectives on this topic.",
        "My goal is to deepen my knowledge in this area.",
        "I want to have an engaging discussion about real-world examples.",
        "I'm excited to learn about recent developments in this field.",
        "I want to discuss how this topic relates to society and ethics.",
        "I'm hoping to discover new ideas and approaches.",
    ]
}

def generate_random_personality() -> PersonalityProfile:
    """Generates a PersonalityProfile with randomly selected attributes."""
    return PersonalityProfile(
        lm_familiarity=random.choice(PERSONALITY_OPTIONS["lm_familiarity"]),
        lm_frequency_use=random.choice(PERSONALITY_OPTIONS["lm_frequency_use"]),
        age=random.choice(PERSONALITY_OPTIONS["age"]),
        gender=random.choice(PERSONALITY_OPTIONS["gender"]),
        employment_status=random.choice(PERSONALITY_OPTIONS["employment_status"]),
        education=random.choice(PERSONALITY_OPTIONS["education"]),
        marital_status=random.choice(PERSONALITY_OPTIONS["marital_status"]),
        english_proficiency=random.choice(PERSONALITY_OPTIONS["english_proficiency"]),
        study_locale=random.choice(PERSONALITY_OPTIONS["study_locale"]),
        religion=random.choice(PERSONALITY_OPTIONS["religion"]),
        ethnicity=random.choice(PERSONALITY_OPTIONS["ethnicity"]),
        location=random.choice(PERSONALITY_OPTIONS["location"]),
        agenda=random.choice(PERSONALITY_OPTIONS["agenda"])
    )

# =============================================================================
# Conversation Turn and Log Module
# =============================================================================
class ConversationTurn:
    def __init__(self,
                 turn: int,
                 role: str,  # 'model'
                 content: str,
                 model_provider: str = None,
                 model_name: str = None,
                 score: int = None,
                 within_turn_id: int = None):
        self.turn = turn
        self.role = role
        self.content = content
        self.model_provider = model_provider
        self.model_name = model_name
        self.score = score
        self.within_turn_id = within_turn_id

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "role": self.role,
            "content": self.content,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "score": self.score,
            "within_turn_id": self.within_turn_id
        }

class Conversation:
    def __init__(self, user_id: str, topic: str):
        self.conversation_id = f"c{uuid.uuid4().hex[:4]}"
        self.user_id = user_id
        self.topic = topic
        self.conversation_history: List[ConversationTurn] = []

    def add_turn(self,
                 turn: int,
                 role: str,
                 content: str,
                 model_provider: str = None,
                 model_name: str = None,
                 score: int = None,
                 within_turn_id: int = None):
        ct = ConversationTurn(turn, role, content,
                              model_provider=model_provider,
                              model_name=model_name,
                              score=score,
                              within_turn_id=within_turn_id)
        self.conversation_history.append(ct)

    def export_history(self) -> List[dict]:
        return [turn.to_dict() for turn in self.conversation_history]

    def __repr__(self):
        return (f"Conversation(conversation_id='{self.conversation_id}', "
                f"user_id='{self.user_id}', topic='{self.topic}', "
                f"conversation_history={self.export_history()})")

# =============================================================================
# LM Agent Module Using Groq for Chat Completions
# =============================================================================
class LM_Agent:
    def __init__(self, name: str, personality: PersonalityProfile, provider: str, model: str, debug: bool = False):
        """
        model: The Groq model identifier, e.g. "meta-llama/llama-4-scout-17b-16e-instruct"
        """
        self.name = name
        self.personality = personality
        self.provider = provider
        self.model = model
        self.debug = debug

    def generate_response(self,
                          conversation_context: List[ConversationTurn],
                          self_name: str,
                          other_name: str,
                          turn_index: int,
                          topic: str = None,
                          is_first_speaker_on_turn_0: bool = False) -> dict:
        """
        Build the prompt with awareness of:
         - The agent's own name (self_name)
         - The other agent's name (other_name)
         - The current turn index
         - The random topic (if needed)
         - Whether this is the first speaker on turn 0 (if so, ask a question about the topic)
         - If this agent is the persona agent and responding, parse the rating.

        Returns a dictionary with response details.
        """
        prompt = self._build_prompt(
            conversation_context=conversation_context,
            self_name=self_name,
            other_name=other_name,
            turn_index=turn_index,
            topic=topic,
            is_first_speaker_on_turn_0=is_first_speaker_on_turn_0
        )
        if self.debug:
            print(f"--- Prompt for {self_name} ---")
            print(prompt)
            print("--------------------------------")
            
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
            )
        except APITimeoutError:
            if self.debug:
                print(f"--- Timeout error for {self_name}, retrying after 10 seconds... ---")
            time.sleep(10) # Wait for 10 seconds
            # Retry the call once
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
            )

        raw_content = chat_completion.choices[0].message.content
        content = raw_content # Keep the original content regardless of score parsing
        score = -1 # Default score
        within_turn_id = 0 # Default within_turn_id

        # Check if this is the persona agent responding (not the first speaker)
        is_persona_agent_responding = (
            self.name == "User" and
            (turn_index > 0 or (turn_index == 0 and not is_first_speaker_on_turn_0)) and
            len(conversation_context) > 0 # Ensure there's a previous message to rate
        )
        
        if self.debug:
            print(f"--- Raw content from {self.name} ---")
            print(raw_content)
            print("--------------------------------")

        if is_persona_agent_responding:
            # Try to find a rating like {85} anywhere in the response
            match = re.search(r"\{(\d{1,3})\}", raw_content) # Use re.search to find pattern anywhere
            if self.debug:
                print(f"--- Search result for score in {self.name}'s response ---")
                print(match)
                print("--------------------------------")
            if match:
                parsed_score = int(match.group(1)) # Extract the number
                # Ensure score is within 1-100 range
                if 1 <= parsed_score <= 100:
                    score = parsed_score
                    if self.debug:
                        print(f"--- Parsed score {score} from {self.name} ---")
                else:
                    # Handle invalid score range if needed, maybe log a warning
                    if self.debug:
                        print(f"--- Warning: Parsed score {parsed_score} out of range (1-100) from {self.name}. Using default. ---")
                    # Keep score as -1 (default)
            else:
                # Handle case where rating is missing if needed, maybe log a warning
                if self.debug:
                    print(f"--- Warning: Rating pattern {{NUMBER}} not found in response from {self.name}. Using default score. ---")
                # Keep score as -1 (default)

        return {"content": content, "score": score, "within_turn_id": within_turn_id}

    def _build_prompt(self,
                      conversation_context: List[ConversationTurn],
                      self_name: str,
                      other_name: str,
                      turn_index: int,
                      topic: str = None,
                      is_first_speaker_on_turn_0: bool = False) -> str:
        """
        Construct the prompt text. If it's turn 0 and this agent is the first speaker,
        they must ask a question or make a request about the topic.
        If this agent is the persona agent and responding, instruct it to rate first.
        Removes score indicators from the history before adding to the prompt.
        """
        # Build conversation history text
        history_str = "\n".join(
            # Use the actual agent name stored in the turn
            f"{turn.model_name}: {turn.content}" for turn in conversation_context
        )
        
        # Remove score indicators (e.g., "{85} ") from the history string
        history_str = re.sub(r"\{\d{1,3}\}\s*", "", history_str)

        # Base system instructions: Let the agent know who it is and who it's talking to.
        system_description = (
            f"You are {self_name}.\n"
            f"The other agent is {other_name}.\n"
            f"Your personality: {self.personality}\n"
            f"Agenda: {self.personality.agenda}\n"
            f"Current turn index: {turn_index}\n\n"
        )

        # Determine the core instruction based on turn and speaker order
        if turn_index == 0 and is_first_speaker_on_turn_0 and topic:
            # Force them to ask or request about the chosen topic
            instruction = (
                f"On turn 0, you must ask a question or make a request specifically about this topic:\n"
                f"TOPIC: {topic}\n"
                f"Don't answer; just pose a question or request focusing on the topic.\n"
            )
        else:
            # Otherwise, just continue the conversation normally.
            instruction = (
                "Continue the conversation based on the existing messages.\n"
                "If you're responding to a question, try to answer it.\n"
                "If the conversation continues, you can ask follow-up questions or give further insights.\n"
            )
            if topic and turn_index == 0 and not is_first_speaker_on_turn_0:
                 # If still turn 0 but not the first speaker, it means you should respond to the question.
                 instruction += (
                     f"This is turn 0, second speaker. Respond to the question or request about the topic:\n"
                     f"TOPIC: {topic}\n"
                 )

        # Add rating instruction if this is the persona agent responding
        is_persona_agent_responding = (
            self_name == "User" and
            (turn_index > 0 or (turn_index == 0 and not is_first_speaker_on_turn_0)) and
            len(conversation_context) > 0 # Check if there's a previous message
        )
        if is_persona_agent_responding:
            # Updated instruction: Ask for the rating anywhere, but preferably at the start.
            rating_instruction = (
                f"IMPORTANT: Rate the previous response from {other_name} on a scale of 1 to 100, "
                f"based on how well it aligns with your persona and agenda. "
                f"Be objective. Include the rating *anywhere* in your response, enclosed in curly braces, like {{rating}}. "
                f"For example: {{85}}. It's helpful if it's near the beginning.\n\n"
            )
            # Prepend the rating instruction to the main instruction.
            instruction = rating_instruction + instruction

        # Put everything together
        full_prompt = (
            system_description
            + "Conversation so far:\n"
            + (history_str if history_str else "(No prior messages)") + "\n\n"
            + instruction
        )

        return full_prompt

    def __repr__(self):
        return (f"LM_Agent(name='{self.name}', provider='{self.provider}', "
                f"model='{self.model}', personality={self.personality})")

# =============================================================================
# Conversation Orchestrator Module
# =============================================================================
class ConversationOrchestrator:
    def __init__(self, user_id: str, lm_agents: List[LM_Agent], topic: str):
        # Use the provided topic instead of random selection
        self.topic = topic
        # Create the conversation object
        self.conversation = Conversation(user_id, self.topic)

        # We expect exactly two LM agents for this scenario
        if len(lm_agents) != 2:
            raise ValueError("This script expects exactly two agents.")
        self.agent0 = lm_agents[0]
        self.agent1 = lm_agents[1]

        self.current_turn = 0
        # Store personality profiles
        self.personality_profiles = {
            self.agent0.name: self.agent0.personality.to_dict(),
            self.agent1.name: self.agent1.personality.to_dict()
        }

    def next_round(self):
        """
        On turn 0:
          - agent0 asks a question or request about the chosen topic
          - agent1 responds to that question
        On subsequent turns (1, 2, etc.):
          - agent0 responds
          - agent1 responds
        """
        if self.current_turn == 0:
            # Turn 0: agent0 asks question or request about the topic
            context = []
            response0 = self.agent0.generate_response(
                conversation_context=context,
                self_name=self.agent0.name,
                other_name=self.agent1.name,
                turn_index=self.current_turn,
                topic=self.topic,
                is_first_speaker_on_turn_0=True
            )
            self.conversation.add_turn(
                turn=self.current_turn,
                role="model",
                content=response0["content"],
                model_provider=self.agent0.provider,
                model_name=self.agent0.name,
                score=response0["score"],
                within_turn_id=response0["within_turn_id"]
            )

            # Then agent1 responds to that new message
            context = self.conversation.conversation_history
            response1 = self.agent1.generate_response(
                conversation_context=context,
                self_name=self.agent1.name,
                other_name=self.agent0.name,
                turn_index=self.current_turn,
                topic=self.topic,
                is_first_speaker_on_turn_0=False  # second speaker on turn 0
            )
            self.conversation.add_turn(
                turn=self.current_turn,
                role="model",
                content=response1["content"],
                model_provider=self.agent1.provider,
                model_name=self.agent1.name,
                score=response1["score"],
                within_turn_id=response1["within_turn_id"]
            )
        else:
            # On subsequent turns: agent0 -> agent1
            context = self.conversation.conversation_history
            response0 = self.agent0.generate_response(
                conversation_context=context,
                self_name=self.agent0.name,
                other_name=self.agent1.name,
                turn_index=self.current_turn
            )
            self.conversation.add_turn(
                turn=self.current_turn,
                role="model",
                content=response0["content"],
                model_provider=self.agent0.provider,
                model_name=self.agent0.name,
                score=response0["score"],
                within_turn_id=response0["within_turn_id"]
            )

            context = self.conversation.conversation_history
            response1 = self.agent1.generate_response(
                conversation_context=context,
                self_name=self.agent1.name,
                other_name=self.agent0.name,
                turn_index=self.current_turn
            )
            self.conversation.add_turn(
                turn=self.current_turn,
                role="model",
                content=response1["content"],
                model_provider=self.agent1.provider,
                model_name=self.agent1.name,
                score=response1["score"],
                within_turn_id=response1["within_turn_id"]
            )

        # Increment turn index at the end of the round
        self.current_turn += 1

    def run_conversation(self, max_rounds: int = 3):
        for _ in range(max_rounds):
            self.next_round()

    def export_conversation(self) -> dict:
        return {
            "conversation_id": self.conversation.conversation_id,
            "user_id": self.conversation.user_id,
            "topic": self.conversation.topic,
            "personality_profiles": self.personality_profiles,
            "conversation_history": self.conversation.export_history()
        }

# =============================================================================
# Main Script Entry
# =============================================================================
def main(debug=False):
    # Read the topic frequencies from the real dataset
    df = pd.read_csv('merged_data_with_topics.csv')
    topic_frequencies = df['topic'].value_counts().to_dict()
    
    # Calculate total number of conversations needed
    total_conversations = sum(topic_frequencies.values())
    
    # Create a list of topics that matches the exact frequencies
    topics_list = []
    for topic, count in topic_frequencies.items():
        topics_list.extend([topic] * count)
    
    # Shuffle the list to randomize the order
    random.shuffle(topics_list)
    
    all_conversations_data = []

    if debug:
        print(f"Starting simulation for {total_conversations} conversations...")
        print("Topic distribution:")
        for topic, count in topic_frequencies.items():
            print(f"{topic}: {count}")
    
    # Wrap the conversation loop with tqdm
    for i in tqdm(range(total_conversations), desc="Simulating conversations"):
        if debug:
            print(f"\n--- Running Conversation {i+1}/{total_conversations} ---")
        
        persona_with_agenda = generate_random_personality()
        default_persona = PersonalityProfile(agenda="I am here to assist without bias.")
        model_name = "llama3-8b-8192"

        # Add debug flag to agents
        lm_agent0 = LM_Agent(name=f"User", personality=persona_with_agenda, 
                           provider="groq", model=model_name, debug=debug)
        lm_agent1 = LM_Agent(name=f"Agent", personality=default_persona,
                           provider="groq", model=model_name, debug=debug)
        
        agents = [lm_agent0, lm_agent1]
        user_id = f"user_sim_{i}"
        
        # Use the pre-determined topic from our shuffled list
        orchestrator = ConversationOrchestrator(user_id, agents, topics_list[i])
        
        if debug:
            print(f"Conversation Topic: {orchestrator.topic}")

        orchestrator.run_conversation(max_rounds=4)
        conversation_data = orchestrator.export_conversation()
        all_conversations_data.append(conversation_data)

        if debug:
            print(f"--- Finished Conversation {i+1} (ID: {conversation_data['conversation_id']}) ---")

    output_filename = "all_conversations_dump_matched_frequencies.json"
    if debug:
        print(f"\nSaving data for {len(all_conversations_data)} conversations to {output_filename}...")
    
    with open(output_filename, "w") as f:
        json.dump(all_conversations_data, f, indent=2)
    
    if debug:
        print(f"Successfully saved all conversation data to {output_filename}.")


if __name__ == "__main__":
    main(debug=False)