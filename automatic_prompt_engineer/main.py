import random

import fire

from automatic_prompt_engineer import ape, data
from experiments.data.instruction_induction.load_data import load_data, tasks
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator
from llm import GPT_Forward, model_from_config
from experiments.evaluation.instruction_induction import utility
import config


def run():
    film = []
    idFilm = []

    with open('dataset/movies.dat', 'r', encoding='latin-1') as file:
        for i, line in enumerate(file):
            if i >= 1000:
                break  # Ferma dopo la riga 199 (cioè la riga n. 200)
            line = line.strip()
            if line:
                parts = line.split("::")
                if len(parts) == 3:
                    movie_id, title, genres = parts
                    film.append(f"{movie_id}::{title} ::{genres}")
                    idFilm.append(f"{movie_id}::")

    test_idFilm = idFilm[499:]
    idFilm = idFilm[:499]
    test_film = film[499:]
    film = film[:499]

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
    prompt_gen_template = "I gave a friend a instruction. Based on the instruction they produced  " \
                          "the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]"
    #prompt_gen_template = """you are the movielens1M dataset , i will give to you a lookup key 'MovieID' and you will provide to me more info about the item:\n\n[full_DEMO]\n\nThe instruction was to [APE]"""
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"

    base_config = '../experiments/configs/instruction_induction.yaml'
    conf = {
        'generation': {
            'num_subsamples': 2,
            'num_demos': 5,
            'num_prompts_per_subsample': 2,
            'model': {
                'gpt_config': {
                    # 'model': 'text-ada-001'
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': 'movietesttask',
            'num_samples': min(2, len(eval_data[0])),
            'num_samples_2': min(2, len(eval_data[0])),
            'model': {
                'gpt_config': {
                    # 'model': 'text-ada-001'
                }
            }
        }
    }

    res, demo_fn = ape.find_prompts(eval_template=eval_template,
                                    prompt_gen_data=prompt_gen_data,
                                    eval_data=eval_data,
                                    conf=conf,
                                    base_conf=base_config,
                                    few_shot_data=prompt_gen_data,
                                    demos_template=demos_template,
                                    prompt_gen_template=prompt_gen_template)

    print('Finished finding prompts.')
    prompts, scores = res.sorted()
    print('Prompts:')
    for prompt, score in list(zip(prompts, scores))[:10]:
        print(f'  {score}: {prompt}')

    # Evaluate on test data
    print('Evaluating on test data...')

    test_conf = {
        'generation': {
            'num_subsamples': 2,
            'num_demos': 5,
            'num_prompts_per_subsample': 2,
            'model': {
                'gpt_config': {
                    # 'model': 'text-ada-001'
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': 'movieTest',
            'num_samples': min(50, len(test_data[0])),
            'num_samples_2': 1,
            'model': {
                'gpt_config': {
                    # 'model': 'text-ada-001'
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
    print(f'Test score: {test_score}')

    # Save a text file to experiments/results/instruction_induction/task.txt with the best prompt and test score
    with open('movielensDetect.txt', 'w') as f:
        f.write(f'Test score: {test_score}\n')
        f.write(f'Prompt: {prompts[0]}\n')

    #value = evaluate_prompt(prompts[0], test_film, conf, base_config)

    #print(f'{value}/{len(test_film)}')

def evaluate_prompt(prompt, answers, conf, base_conf):
    configuration = config.update_config(conf, base_conf)
    model = model_from_config(configuration['evaluation']['model'])
    model_outputs = ""
    model_outputs = model.evaluate_best_prompt(prompt, 1)

    score_fn = utility.get_multi_answer_em
    value = 0
    for prediction, ans_ in zip(model_outputs, answers):
        score = score_fn(prediction, ans_)
        value += 1

    return value




if __name__ == '__main__':
    fire.Fire(run())