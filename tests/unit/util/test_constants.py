from unittest import TestCase, main
from omegaconf import DictConfig

from src.util.dict_extraction import get_nested_dict_values


class TestAnalysisColumns(TestCase):


    def test_(self):

        config = DictConfig({
            'params': {
                'ml__datasplit': {
                    'params': {
                        'percentage_unknown_classes': 0.1
                    }
                },
                'io__import': {
                    'class': 'my_dataset_class'
                }
            }
        })

        columns2extract = [
            ['params', 'ml__datasplit', 'params', 'percentage_unknown_classes'],
            ['params', 'io__import', 'class'],
        ]

        attributes = get_nested_dict_values(columns2extract, config)
        self.assertEqual(attributes, ["0.1", 'my_dataset_class'])

        

if __name__ == '__main__':
    main()