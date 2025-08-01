# Note: Complete code will be released after the review process is completed.
# Key implementation details are temporarily omitted for review, with basic structure retained.

import base64
import re
from PIL import Image


HISTORY_PROMPT = """
"""


class HistoryAgent:
    def __init__(self, client):
        self.client = client
        self.messages = []

    def init(self):
        self.messages = [{"role": "system", "content": HISTORY_PROMPT}]

    def encode_image(self, img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def summarize(self, user_input):
        """
        conversation_history: list of dicts in chat message format
        returns: summarized instruction as a plain string
        """
        user_instruction = {'role': 'user', 'content': [{'type': 'text', 'text': user_input}]}
        self.messages.append(user_instruction)

        response = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=self.messages
        )
        result = response.choices[0].message.content.strip()
        print(f"self.messages:{self.messages}")
        print(f"result:{result}")
        return result

    def add_system(self, text, img_path=None):
        content = []
        if img_path:
            base64_img = self.encode_image(img_path)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})
        content.append({"type": "text", "text": text})
        system_message = {"role": "system", "content": content}
        self.messages.append(system_message)

    def add_user(self, text, img_path=None):
        content = []
        if img_path:
            base64_img = self.encode_image(img_path)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})
        content.append({"type": "text", "text": text})
        user_message = {"role": "user", "content": content}
        self.messages.append(user_message)
