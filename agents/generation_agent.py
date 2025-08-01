# Note: Complete code will be released after the review process is completed.
# Key implementation details are temporarily omitted for review, with basic structure retained.

import json


GEN_PROMPT = """
"""

class GenerationAgent:
    def __init__(self, client):
        self.client = client

    def generate_text_and_bbox(self, prompt):
        messages = [{"role": "system", "content": GEN_PROMPT}]
        user_text = f"Input: {prompt}"
        print(f"user_text:{user_text}")
        user_message = {"role": "user", "content": [{"type": "text", "text": f"Input: {user_text}"}]}
        messages.append(user_message)

        try:
            # OpenAI API
            response = self.client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                messages=messages
            )
            
            # JSON
            gen_text = response.choices[0].message.content
            print(f"gen_text:{gen_text}")

            # Markdown
            cleaned_text = gen_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[len("```json"):].strip()
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-len("```")].strip()
            open_braces = gen_text.count("{")
            close_braces = gen_text.count("}")
            if open_braces > close_braces:
                gen_text += "}" * (open_braces - close_braces)

            gen_instructions = json.loads(cleaned_text)
            if "input" in gen_instructions:
                if "bg_prompt" not in gen_instructions["input"]:
                    gen_instructions["input"]["bg_prompt"] = ""  

            else:
                raise ValueError("JSON lack input")
            # gen_instructions["tool"] = "text_to_image_SDXL"
            print(f"gen_instructions:{gen_instructions}")
            
            # self._validate_response(gen_instructions)
            
            return gen_instructions
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse response: {e}")
            raise
        except Exception as e:
            print(f"Error generating instructions: {e}")
            raise

    def _validate_response(self, response):
        required_fields = ["tool", "input", "input.text", "input.layout"]
        
        for field in required_fields:
            parts = field.split('.')
            obj = response
            
            for part in parts:
                if part not in obj:
                    raise ValueError(f"Missing required field: {field}")
                obj = obj[part]
        
        layout = response["input"]["layout"]
        if not isinstance(layout, list):
            raise ValueError("Layout must be a list")
            
        for item in layout:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(f"Invalid layout item: {item}")
            if not isinstance(item[1], list) or len(item[1]) != 4:
                raise ValueError(f"Invalid bounding box: {item[1]}")