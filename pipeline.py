import os
from gap_analyzer import SkillGapAnalyzer
from query_courses import RAGSearchEngine

class PersonalizedLearningPipeline:
    def __init__(self):
        print("\n🚀 Assembling End-to-End AI Learning Pipeline...")
        # Load your analyzer component
        self.analyzer = SkillGapAnalyzer()
        # Connect to your persistent database directory (pointing to Rajesh's vector_db)
        self.search_engine = RAGSearchEngine()
        print("🔗 Pipeline components linked successfully.")

    def run_recommendation_engine(self, user_profile, target_role):
        print(f"\n🎬 Starting Recommendation Cycle for Target Role: {target_role}")
        gap_results = self.analyzer.analyze_profile_and_gap(user_profile, target_role)
        
        if not gap_results or 'calculated_gaps' not in gap_results:
            return None
            
        missing_gaps = gap_results['calculated_gaps']
        all_recommendations = {}
        
        for gap in missing_gaps:
            matched_courses = self.search_engine.search_courses(skill_gap_query=gap, top_k=2)
            all_recommendations[gap] = matched_courses
            
        return all_recommendations