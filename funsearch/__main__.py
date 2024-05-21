import json
import logging
import os
import pathlib
import pickle
import time
import datetime

import click
import llm
from llama_cpp import Llama

from dotenv import load_dotenv

import sys
import signal
import traceback

print(f"main - Python version: {'.'.join(map(str, sys.version_info[:3]))}")



from funsearch import config, core, sandbox, sampler, programs_database, code_manipulation, evaluator

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)


def get_all_subclasses(cls):
  all_subclasses = []

  for subclass in cls.__subclasses__():
    all_subclasses.append(subclass)
    all_subclasses.extend(get_all_subclasses(subclass))

  return all_subclasses


SANDBOX_TYPES = get_all_subclasses(sandbox.DummySandbox) + [sandbox.DummySandbox]
SANDBOX_NAMES = [c.__name__ for c in SANDBOX_TYPES]


def parse_input(filename_or_data: str):
  if len(filename_or_data) == 0:
    raise Exception("No input data specified")
  p = pathlib.Path(filename_or_data)
  if p.exists():
    if p.name.endswith(".json"):
      return json.load(open(filename_or_data, "r"))
    if p.name.endswith(".pickle"):
      return pickle.load(open(filename_or_data, "rb"))
    raise Exception("Unknown file format or filename")
  if "," not in filename_or_data:
    data = [filename_or_data]
  else:
    data = filename_or_data.split(",")
  if data[0].isnumeric():
    f = int if data[0].isdecimal() else float
    data = [f(v) for v in data]
  return data

@click.group()
@click.pass_context
def main(ctx):
  pass


@main.command()
@click.argument("spec_file", type=click.File("r"))
@click.argument('inputs')
@click.option('--model_name', default="gpt-3.5-turbo-instruct", help='LLM model') # ORIGINAL
# @click.option('--model_name', default="codellama-13b-python.Q5_K_M", help='LLM model') # ORIGINAL# 
# @click.option('--model_name', default="starcoder-newbpe-q4_0", help='LLM model') # CHANGED VERSION
@click.option('--output_path', default="./data/", type=click.Path(file_okay=False), help='path for logs and data')
@click.option('--load_backup', default=None, type=click.File("rb"), help='Use existing program database')
@click.option('--iterations', default=-1, type=click.INT, help='Max iterations per sampler')
@click.option('--samplers', default=15, type=click.INT, help='Samplers')
@click.option('--sandbox_type', default="ContainerSandbox", type=click.Choice(SANDBOX_NAMES), help='Sandbox type')
def run(spec_file, inputs, model_name, output_path, load_backup, iterations, samplers, sandbox_type):
  """ Execute function-search algorithm:

\b
  SPEC_FILE is a python module that provides the basis of the LLM prompt as
            well as the evaluation metric.
            See examples/cap_set_spec.py for an example.\n
\b
  INPUTS    input filename ending in .json or .pickle, or a comma-separated
            input data. The files are expected contain a list with at least
            one element. Elements shall be passed to the solve() method
            one by one. Examples
              8
              8,9,10
              ./examples/cap_set_input_data.json
"""

  # Modify inputs to be filesystem-friendly (replace commas with underscores)
  sanitized_inputs = inputs.replace(",", "_")  
  timestamp = str(int(time.time()))
  # Convert the string timestamp back to an integer
  timestamp_int = int(timestamp)
  dt_object = datetime.datetime.fromtimestamp(timestamp_int)
  formatted_date = dt_object.strftime('%Y-%m-%d %H:%M:%S')

  # Split the path into components
  specs = spec_file.name.split("/")
  problem_file = specs[-1]
  problem_name = problem_file.split(".")[0]

  # Extend log_path with sanitized inputs and model name
  log_path = pathlib.Path(output_path) / f"{problem_name}_{model_name}_n={sanitized_inputs}_{timestamp}"
  if model_name.startswith('gpt'):
    model_type='gpt'
    model = llm.get_model(model_name)
    model.key = model.get_key()
  else:
    model_type='llama'
    model = Llama(
        model_path=f"{model_name}.gguf",  # Download the model file first, "phi-2.Q4_K_M.gguf" is "medium, balanced quality - recommended"
        verbose=False,
        # n_ctx=16384,             # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=16,            # The number of CPU threads to use, tailor to your system and the resulting performance
        n_gpu_layers=-1          # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration, -1 is equal to all layers offloaded.
      )
  
  lm = sampler.LLM(2, model, log_path, model_type)
  
  file_run = open('all_runs.txt', 'a')
  file_run.write(f'{formatted_date} {problem_name} {inputs} {model_name}\n')
  file_run.flush()  
  file_run.close()
  
  file_eval = open('last_eval.txt', 'w')
  file_eval.write(f'{formatted_date} {problem_name} {inputs} {model_name}\n')
  file_eval.flush()  # Ensures all internal buffers associated with file are written to disk
  
  if not log_path.exists():
      log_path.mkdir(parents=True)
      logging.info(f"Writing logs to {log_path}")

  specification = spec_file.read()
  function_to_evolve, function_to_run = core._extract_function_names(specification)
  template = code_manipulation.text_to_program(specification)

  conf = config.Config(num_evaluators=1)
  database = programs_database.ProgramsDatabase(
    conf.programs_database, template, function_to_evolve, identifier=timestamp)
  if load_backup:
    database.load(load_backup)

  inputs = parse_input(inputs)

  sandbox_class = next(c for c in SANDBOX_TYPES if c.__name__ == sandbox_type)
  evaluators = [evaluator.Evaluator(
    database,
    sandbox_class(base_path=log_path),
    template,
    function_to_evolve,
    function_to_run,
    inputs,
  ) for _ in range(conf.num_evaluators)]

  # We send the initial implementation to be analysed by one of the evaluators.
  initial = template.get_function(function_to_evolve).body
  evaluators[0].analyse(initial, island_id=None, version_generated=None)
  assert len(database._islands[0]._clusters) > 0, ("Initial analysis failed. Make sure that Sandbox works! "
                                                   "See e.g. the error files under sandbox data.")

  samplers = [sampler.Sampler(database, evaluators, lm)
              for _ in range(samplers)]

  core.run(samplers, database, iterations)


@main.command()
@click.argument("db_file", type=click.File("rb"))
def ls(db_file):
  """List programs from a stored database (usually in data/backups/ )"""
  conf = config.Config(num_evaluators=1)

  # A bit silly way to list programs. This probably does not work if config has changed any way
  database = programs_database.ProgramsDatabase(
    conf.programs_database, None, "", identifier="")
  database.load(db_file)

  progs = database.get_best_programs_per_island()
  print("Found {len(progs)} programs")
  for i, (prog, score) in enumerate(progs):
    print(f"{i}: Program with score {score}")
    print(prog)
    print("\n")


#Back up mechanism:
def save_last_eval(message):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    new_backup_file_path = os.path.join(backup_folder, f'last_eval_{current_time}.txt')
    last_eval_file_path = 'last_eval.txt'

    # If last_eval.txt exists, read its contents and append to the new backup file
    if os.path.exists(last_eval_file_path):
        with open(last_eval_file_path, 'r') as last_eval_file:
            last_eval_content = last_eval_file.read()
        with open(new_backup_file_path, 'w') as new_backup_file:
            new_backup_file.write(last_eval_content)

    # Append the new message to the new backup file
    with open(new_backup_file_path, 'a') as new_backup_file:
        new_backup_file.write(message)

    # Write the new message to last_eval.txt, overwriting it
    with open(last_eval_file_path, 'w') as last_eval_file:
        last_eval_file.write(message)

def signal_handler(signal, frame):
    save_last_eval("Code execution was manually interrupted.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Ensure the backups folder exists
backup_folder = 'data/backups'
if not os.path.exists(backup_folder):
    os.makedirs(backup_folder)

if __name__ == '__main__':
  try:
    print("Running main code...")

    while True:
      main()

  except Exception as e:
    error_message = ''.join(traceback.format_exception(None, e, e.__traceback__))
    save_last_eval(f"Code crashed or interrupted with the following error:\n{error_message}")
    raise
