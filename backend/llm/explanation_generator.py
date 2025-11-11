import os
from openai import OpenAI
import time 
import numpy as np 

def get_deepseek_explanation(prompt):

    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    
    if not deepseek_api_key:
        return "DEEPSEEK_API_KEY not set in environment variables. Please set your Deepseek key."
    
    MAX_RETRIES = 3
    
    for attempt in range(MAX_RETRIES):
        try:
            client = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com"
            )

            response = client.chat.completions.create(
                model="deepseek-chat", 
                messages=[
                    {"role": "system", "content": "You are a professional teacher in teaching other LLM to play taxi_v3. Your task is to simulate your thought process based on the current state and action. Answer concisely in **English** and **strictly limit your response to 50 words**. Focus on the reasoning for choosing this specific action in the current state. Reasoning is more important than the answer we share with you, your output will be used to train other LLM's thinking process."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100, 
                temperature=0.0,
                stream=False
            )
            content = getattr(response.choices[0].message, 'content', None)
            
            if content is None:
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    print(f"Warning: Deepseek Model returned None (filtering/temp issue). Retrying in {wait_time}s (Attempt {attempt + 1}/{MAX_RETRIES}).")
                    time.sleep(wait_time)
                    continue
                else:
                    return "API_CONTENT_ERROR: Deepseek returned empty content (None) after max retries."
            
            return content.strip()
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 5
                print(f"Error: Deepseek API connection failed: {e}. Retrying in {wait_time}s (Attempt {attempt + 1}/{MAX_RETRIES}).")
                time.sleep(wait_time)
                continue
            else:
                return f"API_CONNECTION_ERROR: Deepseek Connection failed after max retries: {e}"

    return "API_UNKNOWN_ERROR: Could not get a response."

def generate_explanation_for_rl_steps(successful_episodes):

    explanations = []
    episodes_to_process = successful_episodes[:3]
    MAX_STEPS_FOR_PROMPT = 15 
    
    for episode_data in episodes_to_process:
        full_reasoning_chain = []
        steps = episode_data["steps"]
        steps_to_send = steps[:MAX_STEPS_FOR_PROMPT]
        
        for i, step in enumerate(steps_to_send):
            state_info = step["state"]       
            action = step["action"]
            
            lookahead_steps = steps[i+1 : i+4]
            lookahead_str = ""
            if lookahead_steps:
                lookahead_details = []
                for j, next_step in enumerate(lookahead_steps):
                    lookahead_details.append(f"Step {i+j+2}: State {next_step['state']} -> Action {next_step['action']}")
                lookahead_str = "Subsequent steps planned: \n" + "\n".join(lookahead_details)
   
            prompt = (
                f"Current Step: {i+1}\n"
                f"Current State: {state_info}\n"
                f"Chosen Action: {action}\n"
                
                f"{lookahead_str}\n\n" 
                
                f"Explain the reasoning for choosing this action:"
            )
            
            step_reasoning = get_deepseek_explanation(prompt)
            
            full_reasoning_chain.append(f"{i+1}: {step_reasoning}")
            
        final_explanation = "\n".join(full_reasoning_chain)

        explanations.append({
            "episode": int(episode_data["episode"]),
            "total_reward": float(episode_data["total_reward"]),
            "steps": episode_data["steps"],
            "explanation": final_explanation
        })
        
    return explanations
