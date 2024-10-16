import numpy as np
from tqdm import tqdm

from strategy.abstract_strategy import AbstractStrategy
from model.abstract_model import AbstractModel
from dataset.adaptest_dataset import AdapTestDataset


class RandomStrategy(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Random Select Strategy'

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        selection = {}
        for sid in tqdm(adaptest_data.data.keys(), "Selecting: "):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            choice = np.random.randint(len(untested_questions))
            selection[sid] = untested_questions[choice]
        return selection