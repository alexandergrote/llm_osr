from pathlib import Path
from typing import Optional, List
from unittest import TestCase, main

import yaml
from pydantic import BaseModel

from src.util.constants import Directory, YamlField
from src.util.dynamic_import import DynamicImport


class YamlTestCase(BaseModel):

    identifier: str
    class_name: str
    params: Optional[dict]


class TestYamlConfig(TestCase):
    def setUp(self) -> None:

        self.yaml_files = Directory.CONFIG.rglob("*.yaml")

    def _load_yaml_file(self, path: Path) -> dict:

        with open(path) as f:
            yaml_file = yaml.load(f, Loader=yaml.SafeLoader)

        return yaml_file

    def _recursive_test_case_lookup(
        self,
        identifier: str,
        required_keys: set,
        dictionary: dict,
        result: list,
    ) -> List[YamlTestCase]:

        found_keys = set(dictionary.keys())

        if len(found_keys.difference(required_keys)) == 0:

            test_case = YamlTestCase(
                identifier=identifier,
                class_name=dictionary[YamlField.CLASS_NAME.value],
                params=dictionary[YamlField.PARAMS.value],
            )

            result.append(test_case)

        for key, values in dictionary.items():

            if isinstance(values, dict):
                result = self._recursive_test_case_lookup(
                    identifier=f"{identifier}__{key}",
                    required_keys=required_keys,
                    dictionary=values,
                    result=result,
                )

        return result

    def _get_all_test_cases_from_yaml(
        self, identifier: str, dictionary: dict
    ) -> List[YamlTestCase]:
        
        result: List[YamlTestCase] = []

        required_keys = set(
            [YamlField.CLASS_NAME.value, YamlField.PARAMS.value]
        )

        result = self._recursive_test_case_lookup(
            identifier=identifier,
            required_keys=required_keys,
            dictionary=dictionary,
            result=result,
        )

        return result

    def test_yaml_files(self):

        for yaml_filepath in self.yaml_files:

            if yaml_filepath.name == "config.yaml":
                continue

            yaml_dict = self._load_yaml_file(path=yaml_filepath)

            if not yaml_dict:
                continue

            cases = self._get_all_test_cases_from_yaml(
                identifier=str(yaml_filepath), dictionary=yaml_dict
            )

            # reverse list so that sub parts of yaml are tested first
            cases.reverse()

            for case in cases:

                with self.subTest(msg=case.identifier):

                    obj_initialized = DynamicImport.init_class(
                        name=case.class_name, params=case.params
                    )

                    self.assertIsNotNone(obj_initialized)


if __name__ == "__main__":
    main()