# Note: Complete code will be released after the review process is completed.
# Key implementation details are temporarily omitted for review, with basic structure retained.
import base64
import json
import re
from PIL import Image
import os

CLARIFY_PROMPT = """
"""


class ClarifyAgent:
    def __init__(self, client, image_captioning_agent, editing_agent, output_dir='outputs'):
        self.client = client
        self.image_captioning_agent = image_captioning_agent
        self.editing_agent = editing_agent
        self.output_dir = output_dir
        # self.messages = [{"role": "system", "content": system_init}]

    def get_latest_image_path(self):
        def is_valid_image(f):
            return (
                f.endswith('.png') and
                not f.endswith('_mask.png') and
                re.match(r'^\d+(\.png|_\w+\.png)$', f)  
            )
    
        def extract_index(f):
            match = re.match(r'^(\d+)', f)
            return int(match.group(1)) if match else -1
    
        images = [f for f in os.listdir(self.output_dir) if is_valid_image(f)]
        images.sort(key=extract_index, reverse=True)
        return os.path.join(self.output_dir, images[0]) if images else None

    def generate_and_apply_clarification(self, original_prompt, image_path):
        # Step 1: Generate clarification question
        img_caption = self.image_captioning_agent.inference(image_path)

        messages = [{"role": "system", "content": CLARIFY_PROMPT}]
        user_text = f"Original text description: {original_prompt}, Image's actual description: {img_caption}"
        user_message = {"role": "user", "content": [{"type": "text", "text": user_text}]}
        messages.append(user_message)

        response = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=messages
        )

        clarify_question = response.choices[0].message.content
        print(f"Generated clarification question: {clarify_question}")

        # Step 2: Directly apply edits via editing agent
        img_path = self.get_latest_image_path()
        self.editing_agent.edit_image(clarify_question, img_path)
