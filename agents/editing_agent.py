# Note: Complete code will be released after the review process is completed.
# Key implementation details are temporarily omitted for review, with basic structure retained.

import os
import base64
import json
import re
from PIL import Image


EDITING_PROMPT = """
"""

class EditingAgent:
    def __init__(self, client, output_dir, detect_object_agent):
        self.client = client
        # self.command_parser = command_parser
        # self.run_script = run_script
        # self.save_json = save_json
        # self.load_json = load_json
        # self.encode_image = encode_image
        self.output_dir = output_dir
        self.detect_object_agent = detect_object_agent

    def encode_image(self, img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def command_parse(self, commands, text, text_bg, dir='outputs'):
        from command_parser import command_parse
        return command_parse(commands, text, text_bg, dir)

    def save_json(self, data, path):
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def run_script(self, script, json_file='results/input.json'):
        os.system(f'python {script} --json_out True --json_file {json_file}')

    def edit_image(self, user_instruction, prompt, image_path, negative_p="", text_bg=""):

        print("*************")
        detection = self.detect_object_agent.detect(image_path)
        print(f"detection_agent:{detection}")
        user_text = (
            f"user_instruction: {user_instruction}\nPrompt: {prompt}\nNegative Prompt:{negative_p}\nI can give you the position of all objects in the image. Detected Objects: {str(detection)}\n"
            "Please only output the editing operations through dict, do not output other analysis process."
            "Return the result with only plain text, do not use any markdown or other style. All characters must be in English."
        )
        # user_text = f"Prompt: {prompt}\n Detected Objects: {str(detection)}\n"
      
        # correction_out = self.query_prompt('prompts/correction.txt', user_text, image_path=image_path)
        messages = [{"role": "system", "content": EDITING_PROMPT}]
        # user_text = f"Original text description: {original_prompt}, Image's actual description: {img_caption}"
        base64_img = self.encode_image(image_path)
        user_message = {
            "role": "user", 
            "content": [
                {"type": "text", "text": user_text}, 
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]
            }
        messages.append(user_message)

        response = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=messages
        )

        edit_text = response.choices[0].message.content

        print(f"edit_text0: {edit_text}")
        self.save_json(edit_text, 'results/correction_text.json')
        print(f"edit_text: {edit_text}")


        edit_text = re.sub(r',\s*]', ']', edit_text.strip()) 
        try:
            edit_instructions = json.loads(edit_text)
        except json.JSONDecodeError as e:
            print(f"error: {e}")
            edit_instructions = []

        # correction_text = eval(correction_out)
        # edit_instructions = json.loads(edit_text)
        # print(f"edit_instructions:{edit_instructions}")
        # print("0.4")
    
        # Step 3: Parse + Apply Edits
        # commands = [gen_text] + (correction_text if isinstance(correction_text, list) else [correction_text])
        commands = edit_instructions if isinstance(edit_instructions, list) else [edit_instructions]
        full_seq = self.command_parse(commands, prompt, text_bg, self.output_dir)
        print(f"full_seq:{full_seq}")
    
        # for arg in full_seq[1:]:
        for arg in full_seq[0:]:
            self.save_json(arg, 'results/input.json')
            if arg['tool'] in ["object_addition_anydoor", "segmentation"]:
                self.run_script('agent_tool_aux.py')
            elif arg['tool'] in ["addition_anydoor", "replace_anydoor", "remove", "instruction", "attribute_diffedit"]:
                self.run_script('agent_tool_edit.py')
            elif arg['tool'] in ['text_to_image_SDXL', 'image_to_image_SD2', 'layout_to_image_LMD',
                                 'layout_to_image_BoxDiff', 'superresolution_SDXL']:
                self.run_script('agent_tool_generate.py')

    def _clean_and_fix_json(self, json_str):
        json_str = json_str.strip()
        
        if json_str.startswith("```json"):
            json_str = json_str[7:].lstrip()
        if json_str.endswith("```"):
            json_str = json_str[:-3].rstrip()
            
        json_str = self._handle_special_characters(json_str)
            
        json_str = re.sub(r'}\s*{', '},{', json_str)
        
        json_str = re.sub(r'("\w+":\s*(?:".*?"|\d+\.?\d*|\[.*?\]|\{.*?\}))\s*([}\]])', r'\1,\2', json_str)
        
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1 "\2":', json_str)
        
        json_str = self._fix_unclosed_quotes(json_str)
        
        return json_str
    
    def _handle_special_characters(self, json_str):
        special_chars = {
            "\n": r"\n",
            "\t": r"\t",
            "\r": r"\r",
            "\\": r"\\",
            "\"": r"\"",
        }
        
        for char, replacement in special_chars.items():
            json_str = json_str.replace(char, replacement)
            
        return json_str
    
    def _fix_unclosed_quotes(self, json_str):
        in_string = False
        new_json = []
        
        for char in json_str:
            if char == '"':
                in_string = not in_string
            new_json.append(char)
            
        if in_string:
            new_json.append('"')
            
        return ''.join(new_json)
    
    def _get_error_context(self, text, pos, context_length=50):
        start = max(0, pos - context_length)
        end = min(len(text), pos + context_length)
        return text[start:end]
    
    def _partial_json_parse(self, json_str):
        
        try:
            mid = len(json_str) // 2
            json.loads(json_str[:mid])
        except json.JSONDecodeError as e:
            print(f"error: {e}")
            
        try:
            json.loads(json_str[mid:])

        except json.JSONDecodeError as e:
            print(f"error: {e}")
    
    def _validate_edit_instructions(self, instructions):
        if not isinstance(instructions, list):
            raise ValueError(f"Expected a list of instructions, but got {type(instructions).__name__}")
            
        for i, instruction in enumerate(instructions):
            if not isinstance(instruction, dict):
                raise ValueError(f"Instruction at index {i} is not a dictionary: {instruction}")
                
            if 'tool' not in instruction:
                raise ValueError(f"Instruction at index {i} missing 'tool' key: {instruction}")
                
            tool = instruction['tool']
            if tool == 'addition':
                required_keys = {'input', 'box'}
                missing_keys = required_keys - set(instruction.keys())
                if missing_keys:
                    raise ValueError(f"Tool 'addition' at index {i} missing required keys: {missing_keys}")