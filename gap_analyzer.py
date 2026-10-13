import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class SkillGapAnalyzer:
    def __init__(self):
        print("⚙️ Initializing Gemini Orchestrator for Skill-Gap Analysis...")

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = "gemini-3.6-flash"
        print("✅ Gemini API connection active.")

    def generate_text(self, prompt: str):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )
        return response.text.strip()

    def analyze_profile_and_gap(self, user_profile, target_role):
        print(f"🧠 Analyzing profile against target role: '{target_role}'...")
        
        prompt = f"""
        You are an advanced AI Career Counselor.
        Analyze the User Profile and Target Role to identify skill gaps.

        User Profile: {user_profile}
        Target Role: {target_role}

        Return ONLY a strict JSON object with no markdown wrappers:
        {{
            "extracted_skills": ["skill1", "skill2"],
            "target_core_skills": ["skillA", "skillB"],
            "calculated_gaps": ["missing_skill1", "missing_skill2"]
        }}
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            clean_text = response.text.strip()

            if clean_text.startswith("```"):
                clean_text = clean_text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
            if clean_text.startswith("json"):
                clean_text = clean_text.split("json", 1)[1].strip()

            return json.loads(clean_text)

        except Exception as e:
            print(f"❌ Error during AI parsing sequence: {e}")
            return None