# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence

# import llm
import numpy as np
from funsearch import evaluator
from funsearch import programs_database
import re
import ast
import autopep8
import logging

def reformat_to_two_spaces(code: str) -> str:
    # Regular expression to match leading spaces at the beginning of each line
    pattern = r"^\s+"

    # Function to replace leading spaces with two spaces per indentation level
    def replace_with_two_spaces(match):
        space_count = len(match.group(0))
        return ' ' * (2 * (space_count // 4))  # Assumes original indentation was 4 spaces

    # Split the code into lines, reformat each line, and join back into a single string
    reformatted_lines = [
        re.sub(pattern, replace_with_two_spaces, line)
        for line in code.splitlines()
    ]
    return "\n".join(reformatted_lines)

def post_process(code: str) -> str:
    patterns = [
        r'\[/INST\]',
        r'>\[INST\]',
        r'<s>',
        r'<\s>',
        r'\[PYTHON\]\n```',
        r'```\n\[/PYTHON\]',
        r'\[PYTHON\]',
        r'\[/PYTHON\]',
    ]

    for pattern in patterns:
        code = re.sub(pattern, '', code)
    return code

def validate_python_code(code: str) -> str:
    try:
        ast.parse(code)
        return code
    except SyntaxError as e:
        logging.error(f"Syntax error in code: {e}")
        lines = code.split('\n')
        trimmed_code = '\n'.join(lines[:e.lineno - 1])
        return validate_python_code(trimmed_code)


class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int, model, log_path=None, model_type='gpt') -> None:
    self._samples_per_prompt = samples_per_prompt
    self.model = model
    self.prompt_count = 0
    self.log_path = log_path
    self.model_type = model_type

  def _draw_sample(self, prompt: str) -> str:

    if self.model_type=='gpt':
      response = self.model.prompt(prompt)
    else:
      output = self.model(
          "<s>[INST] " + prompt + " [/INST]", 
          max_tokens=4096,
          stop=["</s>"],
          echo=True
      )

      output_text = output['choices'][0]['text']
      code_start = output_text.find('```@funsearch.run\n') + 3  # Find the start of the code block
      response = output_text[code_start:]
      response = post_process(response)
      response = validate_python_code(response)
      response = autopep8.fix_code(response, options={
          'indent_size': 2  # Format to 2 spaces
      })
      with open('last_eval.txt', 'a') as file_eval:   #PVD: output for inspection what else may be required
        file_eval.write(f"FINAL RESPONSE\n{response}\n")
        file_eval.flush()  

    if response:
      self._log(prompt, response, self.prompt_count)
    self.prompt_count += 1
    return response

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

  def _log(self, prompt: str, response: str, index: int):
    if self.log_path is not None:
      with open(self.log_path / f"prompt_{index}.log", "a") as f:
        f.write(prompt)
      with open(self.log_path / f"response_{index}.log", "a") as f:
        f.write(str(response))


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      model: LLM,
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = model

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    prompt = self._database.get_prompt()
    samples = self._llm.draw_samples(prompt.code)
    # This loop can be executed in parallel on remote evaluator machines.

    with open('last_eval.txt', 'a') as file_eval:   #PVD: show when output is valid according to parser
      file_eval.write(f"SAMPLES\n{samples}\n")
      file_eval.flush()

    for sample in samples:
      chosen_evaluator = np.random.choice(self._evaluators)
      chosen_evaluator.analyse(
          sample, prompt.island_id, prompt.version_generated)
