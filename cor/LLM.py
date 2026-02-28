import re
import time
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from openai import OpenAI
API_KEY = ""
BASE_URL =""

@dataclass
class Response:
    content: str
    reasoning_content: str = ""
    json_data: Optional[Dict] = None


class LLMClass:
    """
    A class for handling LLM inference with various model providers.
    Supports batch processing with multithreading.
    """

    def __init__(self):
        
        # Default retry settings
        self.max_retry = 3
        self.validate_json = False

    def json_check(self, content: str) -> Tuple[bool, Optional[Dict]]:
        """
        Check if the content is valid JSON and return the parsed data if it is.
        """
        try:
            parsed_json = json.loads(content)
            return True, parsed_json
        except json.JSONDecodeError:
            pass

        json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_blocks:
            for block in json_blocks:
                try:
                    parsed_json = json.loads(block)
                    return True, parsed_json
                except json.JSONDecodeError:
                    continue

        json_pattern = r'(\{[\s\S]*\})'
        matches = re.findall(json_pattern, content)
        for match in matches:
            try:
                parsed_json = json.loads(match)
                return True, parsed_json
            except json.JSONDecodeError:
                continue

        return False, None

    def extract_and_parse_json(self, content: str) -> dict:
        """
        Extract JSON content from a string and parse it into a dictionary.
        """
        if '```json' in content:
            json_blocks = re.findall(
                r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_blocks:
                content = json_blocks[0].strip()
        if content.count("{") == 1 and content.count("}") == 1:
            content = re.search(r'({.*?})', content).group(0)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        repaired = content
        repaired = re.sub(r'(?<!\\)"', r'\"', repaired)
        repaired = re.sub(r"'(.*?)'", r'"\1"', repaired)
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        stack = []
        json_candidates = []
        start_index = -1

        for i, char in enumerate(repaired):
            if char == '{':
                if not stack:
                    start_index = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_index != -1:
                        json_candidates.append(repaired[start_index:i+1])
                        start_index = -1

        if json_candidates:
            for candidate in reversed(json_candidates):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

        if json_candidates:
            try:
                return json.loads(json_candidates[0])
            except json.JSONDecodeError:
                pass

        matches = re.findall(r'\{[\s\S]*?\}', repaired)
        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue

        print(
            f"Failed to parse JSON content: {content[:200]}{'...' if len(content)>200 else ''}")
        return {}

    def call_llm(self, model: str, message: str, system_prompt: str = "", validate_json: bool = False) -> Response:
        """
        Call LLM with retry logic and JSON validation.
        """
        for attempt in range(self.max_retry):
            try:
                response = self._call_openai_compatible(
                    model, message, system_prompt)

                if self.validate_json:
                    is_json, json_data = self.json_check(response.content)
                    if is_json:
                        response.json_data = json_data
                        return response
                else:
                    return response

            except Exception as e:
                print(f"Attempt {attempt+1} failed for model {model}: {e}")
                time.sleep(1)
        return Response(content="", reasoning_content="")

    def _call_openai_compatible(self, model: str, message: str, system_prompt: str) -> Response:
        """
        Call models through an OpenAI-compatible API.
        Configure YOUR_API_KEY and YOUR_BASE_URL before use.
        """
        try:
            client = OpenAI(
                api_key=API_KEY,
                base_url=BASE_URL,
            )
            mapped_model = self.name2model.get(model, model)
            response = client.chat.completions.create(
                model=mapped_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ]
            )
            res = response.choices[0].message
            return Response(content=res.content)
        except Exception as e:
            print(f"Error in _call_openai_compatible: {e}")
            return Response(content='', reasoning_content='')

    def _extract_reasoning(self, response: Response) -> Response:
        """Extract reasoning content from models that support it"""
        try:
            reasoning_content = re.search(
                r'<think>(.*?)</think>', response.content, re.DOTALL)
            if reasoning_content:
                reasoning = reasoning_content.group(1).strip()
                content = response.content.split('</think>')[-1].strip()
                return Response(content=content, reasoning_content=reasoning, json_data=response.json_data)
        except:
            pass
        return response

    def process_batch_with_pool(self,
                                messages: List[str],
                                model: str,
                                system_prompt: str = "",
                                pool_size: int = 50,
                                validate_json: bool = False) -> List[Response]:
        """
        Process a batch of messages using a thread pool.
        """
        if pool_size == 1:
            return [self.call_llm(model, message, system_prompt, validate_json) for message in messages]
        self.validate_json = validate_json

        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            futures = [
                executor.submit(self.call_llm, model, message,
                                system_prompt, validate_json)
                for message in messages
            ]

            results = []
            for future in tqdm(futures, total=len(messages), desc="Processing requests"):
                response = future.result()

                try:
                    response = self._extract_reasoning(response)
                except:
                    pass
                results.append(response)

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the LLMClass with different models")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to use for testing")
    args = parser.parse_args()

    llm_client = LLMClass()

    system_prompt = "你是一个助手"
    test_message = "你好吗"

    print(f"=== Testing LLM Model ===")
    print(f"Model: {args.model}")
    print(f"System Prompt: {system_prompt}")
    print(f"Message: {test_message}")
    print(f"=" * 40)

    response = llm_client.call_llm(
        model=args.model,
        message=test_message,
        system_prompt=system_prompt,
        validate_json=False
    )

    print("\nTest Successful!")
    print(f"\nResponse Content:")
    print(f"{response.content}")

    if response.reasoning_content:
        print(f"\nReasoning Content:")
        print(f"{response.reasoning_content}")

    if response.json_data:
        print(f"\nJSON Data:")
        print(f"{response.json_data}")
