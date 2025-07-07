import os
import sys
import logging
import time
from datetime import datetime, timedelta
import torch
import multiprocessing as mp

# Local imports
import ape, data, llm, config
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator

# System and environment setup
mp.set_start_method("spawn", force=True)
sys.path.append(os.path.abspath(".."))
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def format_duration(seconds):
    """Convert duration in seconds to a human-readable format."""
    return str(timedelta(seconds=int(seconds)))


def load_movies(file_path, max_lines=3000, split_idx=1499):
    """Load and split movie data from file."""
    film, id_film = [], []
    with open(file_path, 'r', encoding='latin-1') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            parts = line.strip().split("::")
            if len(parts) == 3:
                movie_id, title, genres = parts
                film.append(f"{movie_id}::{title} ::{genres}")
                id_film.append(f"{movie_id}::")

    return (id_film[:split_idx], film[:split_idx]), (id_film[split_idx:], film[split_idx:])

def load_users(file_path, max_lines=6000, split_idx=2999):
    """Load and split user data from file."""
    users, id_users = [], []
    with open(file_path, 'r', encoding='latin-1') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            parts = line.strip().split("::")
            if len(parts) == 5:
                user_id, gender, age, occupation, zip_code = parts
                users.append(f"{user_id}::{gender}::{age}::{occupation}::{zip_code}")
                id_users.append(f"{user_id}::")

    return (id_users[:split_idx], users[:split_idx]), (id_users[split_idx:], users[split_idx:])

def load_ratings(file_path, max_lines=100000, split_idx=49999):
    """Load and split rating data from file."""
    ratings, id_ratings = [], []
    with open(file_path, 'r', encoding='latin-1') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            parts = line.strip().split("::")
            if len(parts) == 4:
                user_id, movie_id, rating, timestamp = parts
                ratings.append(f"{user_id}::{movie_id}::{rating}::{timestamp}")
                id_ratings.append(f"{user_id}::{movie_id}::")

    return (id_ratings[:split_idx], ratings[:split_idx]), (id_ratings[split_idx:], ratings[split_idx:])


def run():
    start_time = time.time()

    # Load and split MovieLens data
    #induce_data, test_data = load_movies('dataset/movies.dat')
    #induce_data, test_data = load_users('dataset/users.dat')
    induce_data, test_data = load_ratings('dataset/ratings.dat')

    # Split induce data into prompt_gen and eval sets
    prompt_gen_size = min(int(len(induce_data[0]) * 0.5), 100)
    prompt_gen_data, eval_data = data.create_split(induce_data, prompt_gen_size)

    # Templates for generating and evaluating prompts
    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\nOutput: [OUTPUT]"
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    prompt_gen_template = (
        """I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n""
        ""[full_DEMO]\n\n""
        ""The instruction was to process input-output pairs where:\n""
        ""- Each input consists of userId, movieId, rating and timestamp in the format 'UserID::MovieID::Rating::Timestamp'\n""
        ""Additional constraints:\n""
        ""- The output format should be: 'UserID::MovieID::Rating::Timestamp'\n\n""
        ""-Dont write python code'\n\n"" 
        ""The instruction to obtaining the output is [APE]""")

    base_config = '../experiments/configs/instruction_induction.yaml'
    conf = {
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': 'movieTest'
        }
    }
    configuration = config.update_config(conf, base_config)
    model = llm.model_from_config(configuration['generation']['model'], disable_tqdm=False)

    # Find prompts
    logging.info("Finding best prompts...")
    res = ape.find_prompts(
        eval_template=eval_template,
        prompt_gen_data=prompt_gen_data,
        eval_data=eval_data,
        conf=conf,
        base_conf=base_config,
        few_shot_data=prompt_gen_data,
        demos_template=demos_template,
        prompt_gen_template=prompt_gen_template,
        model=model
    )
    logging.info("Finished finding prompts.\n")

    # Log top 10 prompts
    prompts, scores = res.sorted()
    logging.info('Top 10 Prompts:')
    for idx, (prompt, score) in enumerate(zip(prompts, scores), start=1):
        logging.info(f'Prompt {idx}: {score}')

    # Setup test config for evaluation
    test_conf = {
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': 'movieTest',
            'num_samples': len(test_data[0])
        }
    }

    # Evaluate best prompt on test data
    logging.info("Evaluating on test data...")
    test_res = ape.evaluate_prompts(
        prompts=[prompts[0]],
        eval_template=eval_template,
        eval_data=test_data,
        few_shot_data=prompt_gen_data,
        demos_template=demos_template,
        conf=test_conf,
        base_conf=base_config,
        model=model
    )
    test_score = test_res.sorted()[1][0]
    logging.info(f'Test score: {test_score}')

    # Log execution metadata
    exec_time = time.time() - start_time
    logging.info(f"Execution completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total execution time: {format_duration(exec_time)}")

    # Save results to file
    results_filename = f"results/Llama-3B-Instruct_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    with open(results_filename, 'w') as f:
        f.write("----------------------------------------------------------------------\n")
        f.write(f"Total execution time: {format_duration(exec_time)}\n")
        f.write(f"Test score: {test_score}\n\n")
        f.write(f"conf: {conf}\n")
        f.write(f"test_conf: {test_conf}\n\n")
        f.write(f"Best Prompt:\n{prompts[0]}\n")

    logging.info(f"Results saved to: {results_filename}")


if __name__ == '__main__':
    global_start = time.time()
    run()
    global_end = time.time()
    print(f"Time elapsed: {format_duration(global_end - global_start)}")
