import json
import os
import concurrent.futures

from absl import app
from absl import flags
from absl import logging
from mathematics_dataset import generate
from tqdm import tqdm, trange

FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', None, 'Where to write output text')
flags.DEFINE_boolean('train_split', True,
                     'Whether to split training data by difficulty')
flags.mark_flag_as_required('output_dir')

def _generate_and_write(module_name, module, regime_dir, per_module):
    path = os.path.join(regime_dir, module_name + '.json')
    data = []
    for _ in range(per_module):
        problem, _ = generate.sample_from_module(module)
        data.append(
            {"question": str(problem.question), "answer": str(problem.answer)}
        )
    with open(path, 'w') as f:
        json.dump(data, f)

def main(_):
    generate.init_modules(FLAGS.train_split)
    output_dir = os.path.expanduser(FLAGS.output_dir)
    if os.path.exists(output_dir):
        logging.fatal('output dir %s already exists', output_dir)
    logging.info('Writing to %s', output_dir)
    os.makedirs(output_dir)

    for regime, flat_modules in tqdm(generate.filtered_modules.items()):
        regime_dir = os.path.join(output_dir, regime)
        os.mkdir(regime_dir)
        per_module = generate.counts[regime]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for module_name, module in tqdm(flat_modules.items(), leave=False):
                futures.append(
                    executor.submit(_generate_and_write,
                                    module_name, module, regime_dir, per_module)
                )
            for _ in tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures),
                          desc="Generating JSON files",
                          leave=False):
                pass

if __name__ == '__main__':
    app.run(main)
