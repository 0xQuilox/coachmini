
import os
import time
import threading
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, Any

class BaseAnalyzer:
    def __init__(self):
        # Load environment variables and configure Gemini API
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
            print("Warning: GEMINI_API_KEY not found in .env file")
        
        # Rate limiting setup
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        self.request_lock = threading.Lock()
    
    def _rate_limit_request(self):
        """Ensure we don't exceed API rate limits"""
        with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def _make_gemini_request(self, analysis_parts):
        """Make a rate-limited request to Gemini API"""
        if not self.model:
            return None
        
        self._rate_limit_request()
        return self.model.generate_content(analysis_parts)
