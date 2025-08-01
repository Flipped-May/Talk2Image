# Note: Complete code will be released after the review process is completed.
# Key implementation details are temporarily omitted for review, with basic structure retained.

import os
import json
import base64
import re
from PIL import Image
from openai import OpenAI
import torch
import numpy as np  
from collections import Counter


from agents.history_agent import HistoryAgent
from agents.generation_agent import GenerationAgent
from agents.editing_agent import EditingAgent
from agents.clarify_agent import ClarifyAgent
from agents.detect_agent import DetectAgent
from agents.detect_image_agent import QwenObjectDetector

from utils.ImageCaptioning import ImageCaptioning
from CLIPScore_eval.clip_class import CLIPScore

from evaluation.styleEvaluation import StyleEvaluator

import json
from pathlib import Path


class ImageEditingPipeline:
    def __init__(self, api_key, api_base, output_dir='outputs0627_temp', output_dir_results='outputs0627'):
        self.api_key = api_key
        self.api_base = api_base
        self.output_dir = output_dir
        self.output_dir_results = output_dir_results
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        self.clip_T = CLIPScore()
        self.style_evaluator = StyleEvaluator()

        self.history_agent = HistoryAgent(self.client)
        self.generation_agent = GenerationAgent(self.client)
        self.image_captioning_agent = ImageCaptioning(device="cuda" if torch.cuda.is_available() else "cpu")

        self.detect_object_agent = QwenObjectDetector(self.client)

        self.editing_agent = EditingAgent(
            self.client,
            self.output_dir,
            self.detect_object_agent
        )
        self.detect_agent = DetectAgent(
            self.client
        )

        self.clarify_agent = ClarifyAgent(
            self.client,
            self.image_captioning_agent,
            self.editing_agent,
            self.output_dir
        )

    def get_latest_image_path(self, folder=None):
        folder = folder or self.output_dir
        def is_valid_image(f):
            return (
                f.endswith('.png') and
                not f.endswith('_mask.png') and
                re.match(r'^\d+(\.png|_\w+\.png)$', f)
            )
    
        def extract_index(f):
            match = re.match(r'^(\d+)', f)
            return int(match.group(1)) if match else -1
    
        images = [f for f in os.listdir(folder) if is_valid_image(f)]
        images.sort(key=extract_index, reverse=True)
        return os.path.join(folder, images[0]) if images else None

    def encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')


    def save_json(self, data, path):
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def run_script(self, script, json_file='results/input.json'):
        os.system(f'python {script} --json_out True --json_file {json_file}')

    def command_parse(self, commands, text, text_bg, dir='outputs'):
        from command_parser import command_parse
        return command_parse(commands, text, text_bg, dir)

    def calculate_object_match_score(self, A, B):
        count_A = Counter(A)
        count_B = Counter(B)
        
        intersection = 0
        for obj in count_A:
            intersection += min(count_A[obj], count_B.get(obj, 0))
        
        len_A = len(A)
        len_B = len(B)
        recall = intersection / len_A if len_A > 0 else 0
        precision = intersection / len_B if len_B > 0 else 0
        
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        return round(f1_score, 3)

    def calculate_combined_score(self, clip_score, style_score, detect_score, weights=[0.4, 0.3, 0.3]):
        """
        Calculate the combined score (direct weighting since all scores are normalized to [0,1])
        :param clip_score: CLIP matching score (ensure it's in [0,1])
        :param style_score: Style similarity score (normalized to [0,1])
        :param detect_score: Object matching score (F1 score, naturally in [0,1])
        :param weights: Weight list, default [CLIP weight, style weight, detection weight], sum to 1
        :return: Combined score (0~1)
        """
        # Verify weight validity
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("The sum of weights must be 1")
        
        # Verify score ranges (optional, to ensure inputs meet expectations)
        for score, name in zip(
            [clip_score, style_score, detect_score],
            ["clip_score", "style_score", "detect_score"]
        ):
            if not (0 <= score <= 1):
                raise ValueError(f"{name} must be in [0,1] range, current value: {score}")
        
        # Calculate weighted combined score
        w1, w2, w3 = weights
        combined_score = w1 * clip_score + w2 * style_score + w3 * detect_score
        
        # Ensure the result is in [0,1] (to prevent minor overflow due to floating-point errors)
        return round(max(0.0, min(1.0, combined_score)), 3)

    def gen_calculate_combined_score(self, clip_score, detect_score, weights=[0.5, 0.5]):
        """
        Calculate the combined score (direct weighting since all scores are normalized to [0,1])
        :param clip_score: CLIP matching score (ensure it's in [0,1])
        :param style_score: Style similarity score (normalized to [0,1])
        :param detect_score: Object matching score (F1 score, naturally in [0,1])
        :param weights: Weight list, default [CLIP weight, style weight, detection weight], sum to 1
        :return: Combined score (0~1)
        """
        # Verify weight validity
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("The sum of weights must be 1")
        
        # Verify score ranges (optional, to ensure inputs meet expectations)
        for score, name in zip(
            [clip_score, detect_score],
            ["clip_score", "detect_score"]
        ):
            if not (0 <= score <= 1):
                raise ValueError(f"{name} must be in [0,1] range, current value: {score}")
        
        # Calculate weighted combined score
        w1, w2 = weights
        combined_score = w1 * clip_score + w2 * detect_score
        
        # Ensure the result is in [0,1] (to prevent minor overflow due to floating-point errors)
        return round(max(0.0, min(1.0, combined_score)), 3)

    def edit_calculate_all_scores(
        self,  # Pass the class instance (like self of the current class) to call internal methods
        parsed_prompt,
        edited_img_path,
        current_img_path,
        weights=[0.4, 0.3, 0.3]
    ):
        """
        Calculate all evaluation scores and return the combined score
        :param self: Class instance, used to call internal tools (like clip_T, detect_agent, etc.)
        :param parsed_prompt: Parsed prompt (for CLIP scoring)
        :param edited_img_path: Path of the generated image
        :param current_img_path: Path of the target style reference image
        :param weights: Weights for the combined score
        :return: Dictionary containing each sub-score and the combined score, e.g., {"clip":..., "style":..., "detect":..., "combined":...}
        """

        # Note: Complete code will be released upon acceptance of the manuscript.
        return scores

    def gen_calculate_all_scores(
        self,  # Pass the class instance (like self of the current class) to call internal methods
        parsed_prompt,
        gen_img_path,
        weights=[0.5, 0.5]
    ):
        """
        Calculate all evaluation scores and return the combined score
        :param self: Class instance, used to call internal tools (like clip_T, detect_agent, etc.)
        :param parsed_prompt: Parsed prompt (for CLIP scoring)
        :param gen_img_path: Path of the generated image
        :param weights: Weights for the combined score
        :return: Dictionary containing each sub-score and the combined score, e.g., {"clip":..., "style":..., "detect":..., "combined":...}
        """

        # Note: Complete code will be released upon acceptance of the manuscript.
        return scores
    
    def run_gen(self, user_prompt, iter_count):
        # Step 1: Generate history summary
        print(f"user_prompt:{user_prompt}")
        summarized_prompts = self.history_agent.summarize(user_prompt)
        try:
            # Step 1: Clean non-JSON content (remove ```json markers and surrounding whitespace)
            cleaned_prompts = summarized_prompts.strip()  # Remove leading and trailing whitespace
            if cleaned_prompts.startswith('```json'):
                cleaned_prompts = cleaned_prompts[len('```json'):].strip()  # Remove opening marker
            if cleaned_prompts.endswith('```'):
                cleaned_prompts = cleaned_prompts[:-len('```')].strip()  # Remove closing marker (if present)
            
            # Step 2: Parse the cleaned JSON string
            parsed_prompts = json.loads(cleaned_prompts)
            
            # Extract fields (use get method to ensure null return if key doesn't exist)
            prompt = parsed_prompts[0]['prompt']
            negative_prompt = parsed_prompts[0].get('negative_prompt', '')  # Note original format is "negative_prompt" (with underscore)
            text_bg = parsed_prompts[0].get('text_bg', '')
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Cleaned content: {cleaned_prompts}")
            # Degraded processing
            prompt = f'"{user_prompt}"'
            negative_prompt = ""
            text_bg = ""
    
        # Step 2: Generate text and bounding boxes
        gen_instructions = self.generation_agent.generate_text_and_bbox(prompt)
    
        # Step 3: Generate initial image
        seq_args = self.command_parse(
            [gen_instructions],
            prompt,
            gen_instructions['input']['bg_prompt'],
            self.output_dir
        )
input.json')
        self.run_script('agent_tool_generate.py')
    
        # Get the path of the generated image
        gen_img_path = self.get_latest_image_path()
        if not gen_img_path:
            print("Failed to generate image")
            return None, None
    
        # Calculate evaluation scores
        scores = self.gen_calculate_all_scores(parsed_prompt=prompt, gen_img_path=gen_img_path)
        print(f"Evaluation scores: {scores}")
    
        # Check for errors
        if "error" in scores:
            print(f"Processing aborted: {scores['error']}")
        else:
            # Trigger quality optimization logic
            combined_score = scores["combined"]
            if (combined_score <= 0.4 
                or scores["clip"] <= 0.2  
                or scores["detect"] <= 0.2):
                print("Triggering quality optimization...")
                self.clarify_agent.generate_and_apply_clarification(
                    prompt, gen_img_path
                )
    
        # Return final results
        txt = 'generation complete'
        final_img_path = self.get_latest_image_path()
        # final_clip_score = self.clip_T.compute_similarity(final_img_path, prompt)
        # print(f"clip_score2: {final_clip_score}")
    
        img = Image.open(final_img_path)
        print("Original size:", img.size, "→ 512x512")
        if img.size != (512, 512):
            img = img.resize((512, 512), Image.LANCZOS) 
        if iter_count == 1:
            name = f"{self.output_dir_results.split('/')[-1]}_1.png"
            img_name = os.path.join(self.output_dir_results, name)
            print(f"img_name:{img_name}")
            img.save(img_name)
    
        # Record history
        # self.history_agent.add_system(txt)
        return txt, img_name
    
    def run_edit(self, user_prompt, iter_count):
        # Step 1: Generate history summary
        # summarized_prompt = self.history_agent.summarize(user_prompt)
        # try:
        #     parsed_prompt = json.loads(summarized_prompt)
        # except json.JSONDecodeError:
        #     parsed_prompt = summarized_prompt
        print(f"user_prompt:{user_prompt}")
        # print(f"self.history_agent.messages:{self.history_agent.messages}")
        summarized_prompts = self.history_agent.summarize(user_prompt)
        try:
            # Step 1: Clean non-JSON content (remove ```json markers and surrounding whitespace)
            cleaned_prompts = summarized_prompts.strip()  # Remove leading and trailing whitespace
            if cleaned_prompts.startswith('```json'):
                cleaned_prompts = cleaned_prompts[len('```json'):].strip()  # Remove opening marker
            if cleaned_prompts.endswith('```'):
                cleaned_prompts = cleaned_prompts[:-len('```')].strip()  # Remove closing marker (if present)
            
            # Step 2: Parse the cleaned JSON string
            parsed_prompts = json.loads(cleaned_prompts)
            
            # Extract fields (use get method to ensure null return if key doesn't exist)
            prompt = parsed_prompts[0]['prompt']
            negative_prompt = parsed_prompts[0].get('negative_prompt', '')  # Note original format is "negative_prompt" (with underscore)
            text_bg = parsed_prompts[0].get('text_bg', '')
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Cleaned content: {cleaned_prompts}")
            # Degraded processing
            prompt = f'"{user_prompt}"'
            negative_prompt = ""
            text_bg = ""
    
        print(f"summarized_prompts: {summarized_prompts}")
        # Step 2: Perform editing
        current_img_path = self.get_latest_image_path()
        print(f"current_img_path: {current_img_path}")
        print(f"prompt:{prompt}")
        self.editing_agent.edit_image(user_prompt, prompt, current_img_path, negative_p=negative_prompt, text_bg=text_bg)
    
        # Get the path of the edited image
        edited_img_path = self.get_latest_image_path()
        if not edited_img_path:
            print("Failed to edit image")
            return None, None
    
        # Calculate evaluation scores
        scores = self.edit_calculate_all_scores(
            parsed_prompt=prompt,
            edited_img_path=edited_img_path,
            current_img_path=current_img_path
        )
        print(f"Evaluation scores: {scores}")
    
        # Check for errors
        if "error" in scores:
            print(f"Processing aborted: {scores['error']}")
        else:
            # Trigger quality optimization logic
            combined_score = scores["combined"]
            if (combined_score <= 0.4 
                or scores["clip"] <= 0.2 
                or scores["detect"] <= 0.2 
                or scores["style"] <= 0.2):
                print("Triggering quality optimization...")
                self.clarify_agent.generate_and_apply_clarification(
                    prompt, edited_img_path
                )
    
        # # Save editing results
        txt = 'edit complete'
    
        ## First generation and subsequent generations
        final_img_path = self.get_latest_image_path()
        img = Image.open(final_img_path)
        print(f"Image type: {type(Image)}")  # Should output <class 'module'>
        print("Original size:", img.size, "→ 512x512")
        if img.size != (512, 512):
            img = img.resize((512, 512), Image.LANCZOS) 
    
        if iter_count == 1:
            name = f"{self.output_dir_results.split('/')[-1]}_1.png"
            img_name = os.path.join(self.output_dir_results, name)
            print(f"img_name:{img_name}")
            img.save(img_name)
        else:
            name_iter = f"{self.output_dir_results.split('/')[-1]}_iter_{iter_count}.png"
            img_name_iter = os.path.join(self.output_dir_results, name_iter)
            print(f"img_name_iter:{img_name_iter}")
            img.save(img_name_iter)
            name_inde = f"{self.output_dir_results.split('/')[-1]}_inde_{iter_count}.png"
            img_name_inde = os.path.join(self.output_dir_results, name_inde)
            img.save(img_name_inde)
    
        # Record history
        # self.history_agent.add_system(txt)
        return txt, final_img_path


def main():
    openai_api_key = ""
    openai_api_base = ""

    print("===== Multi-turn Image Generation & Editing =====")
    print("Commands: 'quit', 'clear', 'help'")

    output_path = 'outputs0714/v0_gen_0730'
    os.makedirs(output_path, exist_ok=True)

    dialog_path = os.path.join(output_path, 'test_0')
    dialog_temp_path = os.path.join(dialog_path, "temp")
    os.makedirs(dialog_path, exist_ok=True)
    os.makedirs(dialog_temp_path, exist_ok=True)

    # Configure pipeline output paths
    # pipeline.output_dir = dialog_temp_path
    # pipeline.output_dir_results = dialog_path

    # Initialize pipeline
    pipeline = ImageEditingPipeline(
        api_key=openai_api_key,
        api_base=openai_api_base,
        output_dir = dialog_temp_path,
        output_dir_results = dialog_path,
    )
    # Initialize history information
    pipeline.history_agent.init()

    # Process input image, since this is editing, mainly check the role of editing here
    is_first_turn = True
    iter_count = 1

    while True:
        # if conversation_state["current_image"] and os.path.exists(conversation_state["current_image"]):
        #     print(f"Current image: {conversation_state['current_image']}")
        user_prompt = input("\nEnter image generation or editing prompt (type 'help' for instructions): ")
        
        if user_prompt.lower() == 'quit':
            print("Conversation ended. Thank you!")
            break
        elif user_prompt.lower() == 'clear':
            pipeline.history_agent.init()
            is_first_turn = True
            iter_count += 1 
            print("Conversation cleared")
            continue
        elif user_prompt.lower() == 'help':
            print("Help:")
            print("  'quit' - Exit the program")
            print("  'clear' - Reset the conversation")
            print("  'new image' - Generate a fresh image")
            print("  Other inputs - Edit the current image")
            continue
        
        try:
            # Parse user input that may contain image_url
            print(f"Processing prompt: {user_prompt}")
            if is_first_turn:
                text, image_path = pipeline.run_gen(user_prompt, iter_count)
                is_first_turn = False
                iter_count += 1
            

            else:
                print(f"iter_count:{iter_count}")
                print(f"user_prompt:{user_prompt}")
                # pipeline.history_agent.add_user(user_prompt)
                text, image_path = pipeline.run_edit(user_prompt, iter_count)
                iter_count += 1 


            if text != None and image_path != None:
                print(f"response: {text}; image_path: {image_path}")
            elif text != None:
                print(f"response: {text}")
            elif image_path != None:
                print(f"image_path: {image_path}")
            else:
                print("Image operation failed. Please check logs for details.")
        except Exception as e:
            print(f"Error: {str(e)}")
            # save_conversation_state(conversation_state)


if __name__ == '__main__':
    main()