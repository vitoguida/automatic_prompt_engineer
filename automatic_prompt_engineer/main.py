
import sys
import os
import fire
import ape, data

import logging
from datetime import datetime
import time
from datetime import timedelta
from tqdm import tqdm

sys.path.append(os.path.abspath(".."))
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator

logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='%(message)s'
)

def run():

        # Ottieni la data/ora attuale in formato desiderato
        data_corrente = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"Inizio esecuzione, data: {data_corrente}")
        start_time = time.time()

        film = []
        idFilm = []

        with open('dataset/movies.dat', 'r', encoding='latin-1') as file:
            for i, line in enumerate(file):
                if i >= 3000:
                    break  # Ferma dopo la riga 199 (cioè la riga n. 200)
                line = line.strip()
                if line:
                    parts = line.split("::")
                    if len(parts) == 3:
                        movie_id, title, genres = parts
                        film.append(f"{movie_id}::{title} ::{genres}")
                        idFilm.append(f"{movie_id}::")

        test_idFilm = idFilm[1499:]
        idFilm = idFilm[:1499]
        test_film = film[1499:]
        film = film[:1499]

        induce_data = (idFilm, film)
        test_data = (test_idFilm, test_film)

        # Get size of the induce data
        induce_data_size = len(induce_data[0])
        prompt_gen_size = min(int(induce_data_size * 0.5), 100)
        # Induce data is split into prompt_gen_data and eval_data
        prompt_gen_data, eval_data = data.create_split(
            induce_data, prompt_gen_size)


        # Data is in the form input: single item, output: list of items
        # For prompt_gen_data, sample a single item from the output list
        #prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0]
                                               #for output in prompt_gen_data[1]]

        eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\nOutput: [OUTPUT]"
        #prompt_gen_template = "I gave a friend a instruction. Based on the instruction they produced  " \
                              #"the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]"
        prompt_gen_template = (
            "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n"
            "[full_DEMO]\n\n"
            "The instruction was to process input-output pairs where:\n"
            "- Each input consists of an integer ID followed by movie information in the format 'ID::Movie Title (Year)::Genre'\n"
            "- The output must be formatted to be compatible with a movie database import system.\n\n"
            "Additional constraints:\n"
            "- The output format should be: 'ID::Movie Title (Year) ::Genre'\n\n"
            "This instruction allows transforming structured movie data into a cleaner format for further processing.\n\n"
            "The instruction was to [APE]"
        )

        #prompt_gen_template = """you are the movielens1M dataset , i will give to you a lookup key 'MovieID' and you will provide to me more info about the item:\n\n[full_DEMO]\n\nThe instruction was to [APE]"""
        demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"

        base_config = '../experiments/configs/instruction_induction.yaml'
        conf = {
            'evaluation': {
                'method': exec_accuracy_evaluator,
                'task': 'movieTest',
                'num_samples': min(2, len(eval_data[0])),
                'num_samples_2': min(2, len(eval_data[0]))
            }
        }

        logging.info("Starting finding prompts")
        res, demo_fn = ape.find_prompts(eval_template=eval_template,
                                        prompt_gen_data=prompt_gen_data,
                                        eval_data=eval_data,
                                        conf=conf,
                                        base_conf=base_config,
                                        few_shot_data=prompt_gen_data,
                                        demos_template=demos_template,
                                        prompt_gen_template=prompt_gen_template)

        elapsed_time = time.time() - start_time
        logging.info('Finished finding prompts.')
        prompts, scores = res.sorted()
        logging.info('Prompts:')
        for prompt, score in list(zip(prompts, scores))[:10]:
            logging.info(f'  {score}: {prompt}')

        # Evaluate on test data
        logging.info('Evaluating on test data...')

        test_conf = {
            'generation': {
                'num_subsamples': 2,
                'num_demos': 5,
                'num_prompts_per_subsample': 2,
                'model': {
                    'model_config': {
                        # 'model': 'text-ada-001'
                    }
                }
            },
            'evaluation': {
                'method': exec_accuracy_evaluator,
                'task': 'movieTest',
                'num_samples': len(test_data[0]),
                'num_samples_2': 1,
                'model': {
                    'model_config': {
                        'temperature': 0.7,
                        'max_tokens': 500,
                        'top_p': 1.0,
                        'frequency_penalty': 0.0,
                        'presence_penalty': 0.0
                    },
                    'batch_size': 1
                }
            }
        }

        test_res = ape.evaluate_prompts(prompts=[prompts[0]],
                                        eval_template=eval_template,
                                        eval_data=test_data,
                                        few_shot_data=prompt_gen_data,
                                        demos_template=demos_template,
                                        conf=test_conf,
                                        base_conf=base_config)

        test_score = test_res.sorted()[1][0]
        logging.info(f'Test score: {test_score}')
        print(f'Test score: {test_score}')


        data_corrente = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        logging.info(f"Fine esecuzione, data: {data_corrente}")
        exec_time = time.time() - start_time
        # converto i secondi in hh:mm:ss
        formatted_time = str(timedelta(seconds=int(exec_time)))
        logging.info(f"Tempo totale di esecuzione: {exec_time/3600} ore")


        # Save a text file to experiments/results/instruction_induction/task.txt with the best prompt and test score
        #Llama-3-1-Nemotron-Nano-8B-v1
        #Phi-4-reasoning-plus
        #gemma-3-12b-it
        with open(f'Llama-3B-Instruct_{data_corrente}.txt', 'a') as f:
            f.write(f'----------------------------------------------------------------------\n')
            f.write(f'Tempo totale di esecuzione: {exec_time/3600} ore\n')
            f.write(f'Test score: {test_score}\n\n')
            f.write(f'conf: {conf}\n')
            f.write(f'test_conf: {test_conf}\n\n')
            f.write(f'Prompt: {prompts[0]}\n\n\n\n\n')



if __name__ == '__main__':
    fire.Fire(run())