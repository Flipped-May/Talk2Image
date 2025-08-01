# Note: Complete code will be released after the review process is completed.
# Key implementation details are temporarily omitted for review, with basic structure retained.

import json  
import re    

DETECT_PROPMT="""
"""

class DetectAgent:
    def __init__(self, client):
        self.client = client

    def detect(self, prompt):
        messages = [{"role": "system", "content": DETECT_PROPMT}]
        user_text = f"Input: {prompt}"
        user_message = {"role": "user", "content": [{"type": "text", "text": user_text}]}
        messages.append(user_message)

        # OpenAI API
        response = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=messages
        )
        
    
        detect_text = response.choices[0].message.content
        print(f"detect_text:{detect_text}")
        detect_objects = json.loads(detect_text)

        try:
            return json.loads(detect_text)
        except json.JSONDecodeError:
            if detect_text.startswith("[") and detect_text.endswith("]"):
                try:
                    fixed_text = detect_text.replace("'", '"')
                    return json.loads(fixed_text)
                except json.JSONDecodeError:
                    pass
                    
            if re.match(r'^\[\s*\w+(?:\s*,\s*\w+)*\s*\]$', detect_text):
                try:
                    items = [item.strip() for item in detect_text[1:-1].split(',')]
                    fixed_text = json.dumps(items)
                    return json.loads(fixed_text)
                except Exception:
                    pass
                    
            raise ValueError(f"Invalid response format: {detect_text}")
        