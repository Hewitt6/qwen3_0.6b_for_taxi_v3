import re
import json
import time
from typing import Dict, Mapping, List, Any
from collections import deque
import gymnasium as gym
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
# New imports for API integration
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import argparse
# NEW: async helpers for WebSocket streaming
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

def parse_json_response(response_content: str) -> Dict[str, Any]:
    """
    Parses the JSON response from the model after the </think> block.
    Returns a dictionary with 'action' and 'add_wall_note'.
    """
    default_response = {'action': 0, 'add_wall_note': False}
    json_part = ""
    try:
        json_part = response_content.split('</think>')[-1].strip()
        cleaned_json_part = json_part
        if cleaned_json_part.startswith("```json"):
            cleaned_json_part = cleaned_json_part[7:].strip()
        if cleaned_json_part.endswith("```"):
            cleaned_json_part = cleaned_json_part[:-3].strip()
        data = json.loads(cleaned_json_part)
        action_str = data.get('action_to_take')
        if action_str is not None:
            action = int(action_str)
            add_note = bool(data.get('add_wall_note', False))
            if 0 <= action <= 5:
                return {'action': action, 'add_wall_note': add_note}
            else:
                print(f"Parsed action {action} out of range. Attempting fallback.")
        else:
            print(f"JSON valid but 'action_to_take' key missing. Attempting fallback.")
    except Exception as e:
        print(f"Error parsing model JSON: {e}. Attempting fallback.")

    print(f"Fallback: Searching for digit in raw string: '{json_part}'")
    all_digits = re.findall(r'([0-5])', json_part)
    if all_digits:
        action = int(all_digits[-1])
        print(f"Fallback successful: Found action {action}.")
        return {'action': action, 'add_wall_note': False}
    else:
        print(f"Error: Fallback failed. No 0-5 digit found. Using default. Response: '{response_content}'")
        return default_response

ACTION_MAP: Mapping[int, str] = {
    0: "South", 1: "North", 2: "East", 3: "West", 4: "Pickup", 5: "Dropoff"
}

LOCATION_LOOKUP: Mapping[int, Dict[str, str]] = {
    0: {"name": "Red", "coords": "(0, 0)"},
    1: {"name": "Green", "coords": "(0, 4)"},
    2: {"name": "Yellow", "coords": "(4, 0)"},
    3: {"name": "Blue", "coords": "(4, 3)"},
}

def decode_state(state: int) -> Dict[str, int]:
    dest_idx = state % 4
    state //= 4
    passenger_location = state % 5
    state //= 5
    taxi_col = state % 5
    state //= 5
    taxi_row = state
    return {
        "taxi_row": taxi_row,
        "taxi_col": taxi_col,
        "passenger_location": passenger_location,
        "destination_index": dest_idx,
    }

def describe_state_for_llm(decoded_state: Dict[str, int]) -> str:
    taxi_pos = f"({decoded_state['taxi_row']}, {decoded_state['taxi_col']})"
    destination = LOCATION_LOOKUP[decoded_state["destination_index"]]
    if decoded_state["passenger_location"] == 4:
        passenger_sentence = (
            "The passenger is already **in the taxi**."
            f"The drop-off destination is **{destination['name']}** at {destination['coords']}."
        )
    else:
        pickup = LOCATION_LOOKUP[decoded_state["passenger_location"]]
        passenger_sentence = (
            f"The passenger is **waiting at {pickup['name']}** located at {pickup['coords']}."
            f"The final destination is **{destination['name']}** at {destination['coords']}."
        )
    return f"The taxi is currently at position {taxi_pos}. {passenger_sentence}"

def build_prompt(observation: int, last_history_item: str, wall_note: str) -> List[Dict[str, str]]:
    decoded_state = decode_state(observation)
    llm_description = describe_state_for_llm(decoded_state)
    messages = [
        {
            "role": "user",
            "content": f"""You are a the driver in Taxi-v3 game，the taxi is at your position.
The positions in the Taxi-v3 game are understood as (row, column) coordinates on a 5 times 5 grid.
**Rules & Rewards:**
1.  **Objective:** 1. **Go to passenger**. 2. **Pickup (4)**. 3. **Go to destination**. 4. **Dropoff (5)**. Do this ASAP.
2. You won't move if your action hits a wall, so take note of walls after your trial.
3. You can only pickup or dropoff passenger when you and passenger are at the same position. e.g. If the taxi is at (4,3), and the passenger is at (4,0), your action should be move because you have to arrive there first
**State of this turn:** {llm_description}
**Actions you can take:** 0:One grid down South, 1: One grid up North, 2: One grid to East, 3: One grid to West, 4:Pickup at taxi's position, 5:Dropoff at taxi's position
**Last Result:** - {last_history_item}
**Wall Notes:** {wall_note}

**Your Task:**
1.  Choose the best action for this turn.
2.  Set `add_wall_note: true` ONLY IF your `Last Result` was a **move (0-3)** AND it **hit a wall**.

Think briefly with less than 50 words. After </think>, respond with **ONLY** a valid JSON:
{{"action_to_take": "NUMBER", "add_wall_note": BOOLEAN}}
"""
        }
    ]
    return messages

class SingleTaxiRunner:
    def __init__(self, pipe):
        self.pipe = pipe
        self.env = gym.make("Taxi-v3")
        self.obs, _ = self.env.reset()
        self.history = deque(maxlen=1)
        self.history.append("The game just started.")
        self.wall_note = "- No walls noted yet."
        self.steps = 0
        self.total_reward = 0
        self.invalid_actions = 0
        self.start_time = time.time()

    # New: reset for frontend
    def reset_game(self):
        self.obs, _ = self.env.reset()
        self.history.clear()
        self.history.append("The game just started.")
        self.wall_note = "- No walls noted yet."
        self.steps = 0
        self.total_reward = 0
        self.invalid_actions = 0
        self.start_time = time.time()
        return self.get_state()

    # New: single model-driven step for frontend
    def step_once(self) -> Dict[str, Any]:
        messages = build_prompt(self.obs, self.history[0], self.wall_note)
        try:
            results = self.pipe(
                [messages],
                temperature=0.6, top_p=0.95, top_k=20, min_p=0,
                do_sample=True, max_new_tokens=10000, batch_size=1
            )
        except Exception as e:
            return {"error": f"Inference error: {e}"}

        try:
            response_content = results[0][0]['generated_text'][-1]['content']
        except Exception as e:
            response_content = ""
        parsed = parse_json_response(response_content)
        action = parsed['action']
        add_wall_note = parsed['add_wall_note']
        action_name = ACTION_MAP.get(action, 'Unknown')

        prev_decoded = decode_state(self.obs)
        prev_pos = (prev_decoded['taxi_row'], prev_decoded['taxi_col'])

        obs, reward, terminated, truncated, info = self.env.step(action)
        new_decoded = decode_state(obs)
        new_pos = (new_decoded['taxi_row'], new_decoded['taxi_col'])

        self.steps += 1
        self.total_reward += reward
        if reward == -10:
            self.invalid_actions += 1

        if action <= 3:
            if new_pos == prev_pos and reward == -1:
                entry = f"Tried {action_name} ({action}) at {prev_pos}, but **hit a wall**. Position remained {new_pos}."
                if add_wall_note:
                    if "No walls noted yet." in self.wall_note:
                        self.wall_note = ""
                    note = f"- Wall hit at {prev_pos} when trying to move {action_name}."
                    if note not in self.wall_note:
                        self.wall_note += f"\n{note}"
            else:
                entry = f"Moved {action_name} ({action}). New position is {new_pos}."
        else:
            if reward == -10:
                entry = f"Tried {action_name} ({action}) at {new_pos}, but it was **illegal** (reward: -10)."
            elif terminated:
                entry = f"Successfully performed {action_name} ({action}) and **finished the mission**!"
            else:
                entry = f"Successfully performed {action_name} ({action}) at {new_pos}."

        self.history.append(entry)
        self.obs = obs

        return {
            "step": self.steps,
            "action": action,
            "action_name": action_name,
            "reward": reward,
            "total_reward": self.total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "done": bool(terminated or truncated),
            "last_history": entry,
            "wall_notes": self.wall_note,
            "observation": int(obs),
            "decoded_state": new_decoded,
            "model_raw": response_content,
        }

    # New: state snapshot for frontend
    def get_state(self) -> Dict[str, Any]:
        decoded = decode_state(self.obs)
        return {
            "step": self.steps,
            "total_reward": self.total_reward,
            "invalid_actions": self.invalid_actions,
            "last_history": self.history[0],
            "wall_notes": self.wall_note,
            "observation": int(self.obs),
            "decoded_state": decoded,
            "llm_description": describe_state_for_llm(decoded),
        }

    def run(self, max_steps: int = 100):
        print("Starting single Taxi-v3 game...")
        while self.steps < max_steps:
            messages = build_prompt(self.obs, self.history[0], self.wall_note)
            try:
                results = self.pipe(
                    [messages],
                    temperature=0.6, top_p=0.95, top_k=20, min_p=0,
                    do_sample=True, max_new_tokens=10000, batch_size=1
                )
            except Exception as e:
                print(f"Inference error: {e}")
                break

            try:
                response_content = results[0][0]['generated_text'][-1]['content']
            except Exception as e:
                print(f"Parse result error: {e}")
                response_content = ""

            parsed = parse_json_response(response_content)
            action = parsed['action']
            add_wall_note = parsed['add_wall_note']
            action_name = ACTION_MAP.get(action, 'Unknown')

            prev_decoded = decode_state(self.obs)
            prev_pos = (prev_decoded['taxi_row'], prev_decoded['taxi_col'])

            obs, reward, terminated, truncated, info = self.env.step(action)
            new_decoded = decode_state(obs)
            new_pos = (new_decoded['taxi_row'], new_decoded['taxi_col'])

            self.steps += 1
            self.total_reward += reward
            if reward == -10:
                self.invalid_actions += 1

            if action <= 3:
                if new_pos == prev_pos and reward == -1:
                    entry = f"Tried {action_name} ({action}) at {prev_pos}, but **hit a wall**. Position remained {new_pos}."
                    if add_wall_note:
                        if "No walls noted yet." in self.wall_note:
                            self.wall_note = ""
                        note = f"- Wall hit at {prev_pos} when trying to move {action_name}."
                        if note not in self.wall_note:
                            self.wall_note += f"\n{note}"
                else:
                    entry = f"Moved {action_name} ({action}). New position is {new_pos}."
            else:
                if reward == -10:
                    entry = f"Tried {action_name} ({action}) at {new_pos}, but it was **illegal** (reward: -10)."
                elif terminated:
                    entry = f"Successfully performed {action_name} ({action}) and **finished the mission**!"
                else:
                    entry = f"Successfully performed {action_name} ({action}) at {new_pos}."

            self.history.append(entry)
            print(f"Step {self.steps}: {entry} | Reward: {reward} | Total: {self.total_reward}")

            self.obs = obs
            if terminated or truncated:
                status = "Success" if terminated and reward == 20 else "Failed"
                print(f"Game over: {status} in {self.steps} steps. Total reward: {self.total_reward}.")
                break

        duration = time.time() - self.start_time
        print(f"Duration: {duration:.2f}s | Invalid actions: {self.invalid_actions}")
        print("Wall Notes:")
        print(self.wall_note if self.wall_note.strip() else "- No walls noted yet.")
        self.env.close()

def load_model_and_pipe(
    model_name: str = "Qwen/Qwen3-1.7B",
    adapter_path: str = "/Users/syz/Documents/gym_v3_qwen_play/gym-taxi-qwen3/backend/qwen3-1.7b-dpo-taxi-driver"
):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer pad_token set to eos_token.")

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )

    print(f"Loading PEFT adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapters...")
    model = model.merge_and_unload()
    print("Adapters merged.")

    print("Creating text-generation pipeline...")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype="auto"
    )
    print("Model loaded successfully.")
    return pipe

# New: FastAPI app for frontend integration
APP_LOCK = threading.Lock()
PIPE = None
RUNNER: SingleTaxiRunner | None = None

def create_app() -> FastAPI:
    app = FastAPI(title="Taxi-v3 Qwen3 Driver API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/load_model")
    def api_load_model(
        model_name: str = Body("Qwen/Qwen3-1.7B"),
        adapter_path: str = Body("/Users/syz/Documents/gym_v3_qwen_play/gym-taxi-qwen3/backend/qwen3-1.7b-dpo-taxi-driver"),
    ):
        global PIPE, RUNNER
        with APP_LOCK:
            PIPE = load_model_and_pipe(model_name=model_name, adapter_path=adapter_path)
            RUNNER = SingleTaxiRunner(PIPE)
        return {"status": "loaded"}

    @app.post("/api/start")
    def api_start():
        global PIPE, RUNNER
        with APP_LOCK:
            if PIPE is None:
                PIPE = load_model_and_pipe()
            if RUNNER is None:
                RUNNER = SingleTaxiRunner(PIPE)
            state = RUNNER.reset_game()
        return {"status": "started", "state": state}

    @app.get("/api/state")
    def api_state():
        if RUNNER is None:
            return {"error": "Game not started"}
        return RUNNER.get_state()

    @app.post("/api/next")
    def api_next():
        if RUNNER is None:
            return {"error": "Game not started"}
        result = RUNNER.step_once()
        return result

    @app.post("/api/stop")
    def api_stop():
        global RUNNER
        if RUNNER is not None:
            RUNNER.env.close()
            RUNNER = None
        return {"status": "stopped"}

    # NEW: WebSocket endpoint for live driving/animation
    @app.websocket("/ws/game")
    async def ws_game(ws: WebSocket):
        await ws.accept()
        global PIPE, RUNNER
        try:
            # Ensure model/runner are ready
            with APP_LOCK:
                if PIPE is None:
                    PIPE = load_model_and_pipe()
                if RUNNER is None:
                    RUNNER = SingleTaxiRunner(PIPE)

            await ws.send_json({"type": "ready", "state": RUNNER.get_state()})

            while True:
                msg = await ws.receive_json()
                cmd = (msg.get("cmd") or "state").lower()

                if cmd in ("start", "reset"):
                    state = RUNNER.reset_game()
                    await ws.send_json({"type": "started", "state": state})

                elif cmd == "state":
                    await ws.send_json({"type": "state", "state": RUNNER.get_state()})

                elif cmd == "step":
                    result = await run_in_threadpool(RUNNER.step_once)
                    await ws.send_json({"type": "step", **result})

                elif cmd == "auto":
                    # Auto-run until done; optional speed_ms
                    speed_ms = int(msg.get("speed_ms", 400))
                    while True:
                        result = await run_in_threadpool(RUNNER.step_once)
                        await ws.send_json({"type": "step", **result})
                        if result.get("done"):
                            break
                        await asyncio.sleep(max(0, speed_ms) / 1000.0)

                elif cmd == "stop":
                    await ws.send_json({"type": "stopped"})
                    break

                else:
                    await ws.send_json({"type": "error", "message": f"Unknown cmd: {cmd}"})

        except WebSocketDisconnect:
            # Client disconnected
            pass
        except Exception as e:
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass

    return app

def main():
    # Add CLI/Server switch for flexible usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server for frontend")
    parser.add_argument("--cli", action="store_true", help="Run CLI single game loop")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.serve or not args.cli:
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port)
        return

    # CLI mode (legacy)
    pipe = load_model_and_pipe()
    runner = SingleTaxiRunner(pipe)
    runner.run(max_steps=100)

if __name__ == "__main__":
    main()
