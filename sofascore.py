from typing import Dict, Any, List
from google.cloud import bigquery
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json
import time
from openai import APIConnectionError

class SportsQA:
    # Knowledge base for common sports questions
    SPORTS_KNOWLEDGE = {
        'basketball': {
            'metrics': ['points', 'rebounds', 'assists', 'steals', 'blocks', 
                       'field_goals_made', 'field_goals_attempted'],
            'key_concepts': {
                'scoring': "Basketball scoring involves field goals (2 points), three-pointers (3 points), and free throws (1 point)",
                'efficiency': "Efficiency in basketball is measured through shooting percentages, points per attempt, and overall impact",
                'defense': "Defensive success is measured through blocks, steals, and opponent's shooting percentage"
            }
        },
        'football_nfl': {
            'metrics': ['passing_yards', 'rushing_yards', 'touchdowns', 'interceptions', 'sacks'],
            'key_concepts': {
                'quarterback': "Success factors include accuracy, decision-making, leadership, arm strength, and ability to read defenses",
                'offense': "Effective offense combines passing and rushing strategies with good play-calling",
                'defense': "Strong defense requires tackling ability, coverage skills, and defensive coordination"
            }
        },
        'tennis': {
            'metrics': ['aces', 'double_faults', 'first_serve_percentage', 'break_points_saved'],
            'key_concepts': {
                'roland_garros': "Unique for its clay courts, which affect play style, requiring endurance and tactical play",
                'serving': "Effective serving combines power, accuracy, and variety to keep opponents off-balance",
                'strategy': "Success requires adapting to different surfaces, managing energy, and exploiting opponent weaknesses"
            }
        }
    }

    def __init__(self, credentials_path: str = 'helium-4-ai-1af28ad29823.json'):
        self.client = bigquery.Client.from_service_account_json(
            credentials_path,
            project='helium-4-ai'
        )
        
        # Initialize LLM with retry mechanism
        self.setup_llm()

    def setup_llm(self, max_retries: int = 3):
        """Setup LLM with retry mechanism"""
        for attempt in range(max_retries):
            try:
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                # Test the connection
                self.llm.invoke("Test connection")
                return
            except APIConnectionError:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                print("Warning: OpenAI API connection failed. Using knowledge base only.")
                self.llm = None
            except Exception as e:
                print(f"Warning: LLM initialization failed: {e}")
                self.llm = None
                break

    def _get_sport_from_question(self, question: str) -> str:
        """Determine the sport from the question using keyword matching"""
        question = question.lower()
        
        # Sport-specific keywords
        keywords = {
            'basketball': ['basketball', 'nba', 'wnba', 'points', 'rebounds', 'court'],
            'football_nfl': ['football', 'nfl', 'quarterback', 'touchdown', 'yards'],
            'tennis': ['tennis', 'roland garros', 'serve', 'court', 'grand slam']
        }
        
        # Count keyword matches
        matches = {sport: sum(1 for kw in kws if kw in question)
                  for sport, kws in keywords.items()}
        
        # Return the sport with most matches, default to basketball
        return max(matches.items(), key=lambda x: x[1])[0] if any(matches.values()) else 'basketball'

    def _get_knowledge_base_response(self, question: str, sport: str) -> str:
        """Generate response using internal knowledge base"""
        question = question.lower()
        knowledge = self.SPORTS_KNOWLEDGE[sport]
        
        # Match question with relevant knowledge
        if 'quarterback' in question and sport == 'football_nfl':
            return """
A successful NFL quarterback requires several key attributes:

1. Physical Skills:
   - Strong and accurate arm
   - Mobility in the pocket
   - Quick release

2. Mental Attributes:
   - Decision-making under pressure
   - Ability to read defenses
   - Leadership qualities
   - Game management skills

3. Technical Abilities:
   - Precise ball placement
   - Understanding of offensive schemes
   - Ability to make pre-snap adjustments

4. Intangibles:
   - Work ethic and preparation
   - Clutch performance
   - Team leadership
   - Communication skills

Success also depends heavily on the supporting cast, including offensive line protection, receiver quality, and coaching system.
"""
        elif 'roland garros' in question and sport == 'tennis':
            return """
Roland Garros is unique among tennis tournaments for several key reasons:

1. Clay Court Surface:
   - Only Grand Slam played on clay
   - Slower pace of play
   - Different playing style required
   - Tests endurance and patience

2. Historical Significance:
   - One of the four Grand Slams
   - Rich history dating back to 1891
   - Named after a French aviation pioneer

3. Playing Characteristics:
   - Longer rallies
   - Higher bounces
   - Favors defensive players
   - Tests physical endurance

4. Unique Challenges:
   - Sliding techniques required
   - Different strategy needed
   - Weather can significantly impact play

The tournament is particularly challenging for players who specialize in faster surfaces, making it one of the most demanding tennis events.
"""
        else:
            # Generic response based on sport's key concepts
            concepts = knowledge['key_concepts']
            metrics = knowledge['metrics']
            
            return f"""
Key aspects of {sport.replace('_', ' ')}:

1. Important Metrics:
   {', '.join(metrics).title()}

2. Key Concepts:
   {' '.join(concepts.values())}

3. Success Factors:
   - Physical skills and athleticism
   - Technical proficiency
   - Strategic understanding
   - Mental toughness
   - Consistent performance
"""

    def answer_question(self, question: str) -> str:
        """Main method to answer sports questions"""
        try:
            # Detect sport from question
            sport = self._get_sport_from_question(question)
            print(f"\nAnalyzing {sport.replace('_', ' ')} question...")

            # Try to get response from LLM if available
            if self.llm:
                try:
                    response = self.llm.invoke(
                        f"Answer this sports question about {sport}: {question}"
                    )
                    return str(response.content)
                except Exception as e:
                    print(f"LLM response failed: {e}")
                    # Fall back to knowledge base
                    
            # Use knowledge base response
            return self._get_knowledge_base_response(question, sport)
            
        except Exception as e:
            return f"""
I apologize, but I encountered an error while processing your question.
Here's a general response based on common sports knowledge:

{self._get_knowledge_base_response(question, self._get_sport_from_question(question))}

Feel free to rephrase your question or ask about specific aspects you're interested in.
"""

def main():
    # Load environment variables
    load_dotenv()

    # Initialize the QA system
    qa = SportsQA(credentials_path="helium-4-ai-1af28ad29823.json")

    print("\nWelcome to Advanced Sports Q&A System!")
    print("Ask any question about basketball, NFL football, or tennis - statistical, analytical, historical, or strategic")
    print("\nExample questions:")
    print("- What makes a quarterback successful in the NFL?")
    print("- Who are the top scorers in basketball this season?")
    print("- How do players perform on clay courts at Roland Garros?")
    print("- What are the key defensive statistics in football?")
    
    while True:
        print("\nAsk your question (or type 'quit' to exit):")
        question = input("> ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the Advanced Sports Q&A System!")
            break
            
        if not question:
            print("Please ask a question!")
            continue
            
        answer = qa.answer_question(question)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()