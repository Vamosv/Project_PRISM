import os
import uuid
import random
import json
from typing import List

# Import Groq and initialize the client using your environment variable for the API key.
from groq import Groq

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
        return (f"PersonalityProfile(lm_familiarity='{self.lm_familiarity}', "
                f"lm_frequency_use='{self.lm_frequency_use}', age='{self.age}', "
                f"gender='{self.gender}', employment_status='{self.employment_status}', "
                f"education='{self.education}', marital_status='{self.marital_status}', "
                f"english_proficiency='{self.english_proficiency}', study_locale='{self.study_locale}', "
                f"religion='{self.religion}', ethnicity='{self.ethnicity}', "
                f"location='{self.location}', agenda='{self.agenda}')")

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
    def __init__(self, name: str, personality: PersonalityProfile, provider: str, model: str):
        """
        model: The Groq model identifier, e.g. "meta-llama/llama-4-scout-17b-16e-instruct"
        """
        self.name = name
        self.personality = personality
        self.provider = provider
        self.model = model

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
        print(prompt)
        print("--------------------------------")
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
        )
        content = chat_completion.choices[0].message.content
        return {"content": content, "score": 100, "within_turn_id": 0}

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
        """
        # Build conversation history text
        history_str = "\n".join(
            f"{turn.model_name}: {turn.content}" for turn in conversation_context
        )

        # Base system instructions: Let the agent know who it is and who it's talking to.
        system_description = (
            f"You are {self_name}.\n"
            f"The other agent is {other_name}.\n"
            f"Your personality: {self.personality}\n"
            f"Agenda: {self.personality.agenda}\n"
            f"Current turn index: {turn_index}\n\n"
        )

        # If turn 0 and this is the first speaker, the agent is told to ask a question or request about the topic.
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
            if topic and turn_index == 0:
                # If still turn 0 but not the first speaker, it means you should respond to the question.
                instruction += (
                    f"This is turn 0, second speaker. Respond to the question or request about the topic:\n"
                    f"TOPIC: {topic}\n"
                )

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
    def __init__(self, user_id: str, lm_agents: List[LM_Agent]):
        # Pick a random topic
        self.topic = self.get_random_topic()
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

    def get_random_topic(self) -> str:
        topics = [
            "The future of renewable energy.",
            "The impact of social media on society.",
            "Space exploration and colonization.",
            "The evolution of artificial intelligence."
        ]
        return random.choice(topics)

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
def main():
    # Create a personality profile for an agent with a persona.
    persona_with_agenda = PersonalityProfile(
        lm_familiarity="Somewhat familiar",
        lm_frequency_use="Once per month",
        age="25-34 years old",
        gender="Female",
        employment_status="Working full-time",
        education="Vocational",
        marital_status="Never been married",
        english_proficiency="Native speaker",
        study_locale="canada",
        religion="No Affiliation",
        ethnicity="White",
        location="Northern America",
        agenda="I am here to absorb all the information the universe has to offer."
    )

    # Create a default personality profile for the other agent.
    default_persona = PersonalityProfile(
        agenda="I am here to assist without bias."
    )

    # Specify the Groq model name.
    model_name = "meta-llama/llama-4-scout-17b-16e-instruct"

    # Create two LM agents (exactly two for this logic).
    lm_agent0 = LM_Agent(name="LM-Agent-Persona", personality=persona_with_agenda, provider="groq", model=model_name)
    lm_agent1 = LM_Agent(name="LM-Agent-Default", personality=default_persona, provider="groq", model=model_name)
    agents = [lm_agent0, lm_agent1]

    # Define the user ID. (No explicit user message; the conversation is seeded by the random topic.)
    user_id = "user9"

    # Initialize the orchestrator
    orchestrator = ConversationOrchestrator(user_id, agents)
    # Run the conversation for 3 rounds (turns). Turn 0 => two messages. Turn 1 => two messages. Turn 2 => two messages.
    orchestrator.run_conversation(max_rounds=3)

    # Export the conversation
    conversation_data = orchestrator.export_conversation()

    # Save to JSON
    with open("conversation_dump.json", "w") as f:
        json.dump(conversation_data, f, indent=2)

    # Print results
    print("Final Conversation Log:")
    print(json.dumps(conversation_data, indent=2))

if __name__ == "__main__":
    main()