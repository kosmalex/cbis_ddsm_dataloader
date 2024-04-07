import json
import os.path
import pandas as pd
from typing import List, Dict, Tuple

from torchvision.transforms import Compose
from transforms.patches_centered import centered_patch_transform
from transforms.patches_random import random_patch_transform
from transforms.patches_normal import normal_patch_transform_wrapper
from cbis_datasets.classification_dataset import CBISDDSMClassificationDataset
from cbis_datasets.sam_dataset import SAMDataset
from cbis_datasets.mamosam_dataset import MAMOSAMDataset


class CBISDDSMDatasetFactory:
    def __init__(self, config_path) -> None:
        self.__config = self.__read_config(config_path)
        self.__train: bool = False
        self.__test: bool = False
        self.__dataframe = None
        self.__excluded_attrs: List[str] = []
        self.__excluded_values: Dict[str, set] = {'lesion_type': {'mass', 'calcification'}}
        self.__attribute_mapped_values: Dict[str, Dict[str, str]] = {}
        self.__transform_list = []
        self.__image_transform_list = []
        self.__image_transform_list_applied_training = []
        self.__image_transform_list_applied_validation = []
        self.__plus_normal = False
        self.__patch_transform_selected = False
        self.__split_validation = False
        self.__validation_percentage = 0.0
        self.__split_cross_validation = False
        self.__cross_validation_folds = 5

        # Experimental
        self.__is_sam         = False
        self.__sam_transforms = []

    @staticmethod
    def __read_config(config_path):
        with open(config_path, 'r') as cf:
            config = json.load(cf)
        return config

    def __fetch_filter_lesions(self):
        try:
            csv_file_list = []
            if self.__train:
                csv_file_list.append(os.path.join(self.__config['download_path'], 'lesions_train.csv'))
            if self.__test:
                csv_file_list.append(os.path.join(self.__config['download_path'], 'lesions_test.csv'))

            self.__dataframe = pd.concat((pd.read_csv(f) for f in csv_file_list), ignore_index=True)
        except:
            print('Database seems not properly set up. Please (re)run setup.py or check the paths in config.json.')
            return

        self.__dataframe.drop(self.__excluded_attrs, axis=1, inplace=True)

        for attribute, value_set in self.__excluded_values.items():
            self.__dataframe = self.__dataframe.loc[~self.__dataframe[attribute].isin(list(value_set))]

        for attribute, mapping in self.__attribute_mapped_values.items():
            for v1, v2 in mapping.items():
                self.__dataframe[attribute].replace(v1, v2, inplace=True)

        join_f = lambda x : os.path.join(self.__config['download_path'], x)
        self.__dataframe["image_path"] = self.__dataframe["image_path"].apply(join_f)
        self.__dataframe["patch_path"] = self.__dataframe["patch_path"].apply(join_f)
        self.__dataframe["mask_path"]  = self.__dataframe["mask_path"].apply(join_f)

    def add_masses(self):
        self.__excluded_values['lesion_type'].remove('mass')
        return self

    def add_calcifications(self):
        self.__excluded_values['lesion_type'].remove('calcification')
        return self

    def train(self):
        self.__train = True
        return self

    def test(self):
        self.__test = True
        return self

    def train_test_merge(self):
        self.__train = True
        self.__test = True
        return self

    def sam(self):
      self.__is_sam = True
      return self

    def drop_attribute_values(self, attribute: str, *value_list: str):
        value_set = self.__excluded_values.get(attribute, set())
        for v in value_list:
            value_set.add(v)
        return self

    def map_attribute_value(self, attribute: str, mapping: Dict[str, str]):
        attribute_mapping = self.__attribute_mapped_values.get(attribute, dict())
        attribute_mapping.update(mapping)
        self.__attribute_mapped_values[attribute] = attribute_mapping
        return self

    def drop_attributes(self, *attribute_list: str):
        self.__excluded_attrs.extend(attribute_list)
        return self

    def show_counts(self):
        self.__fetch_filter_lesions()
        df = self.__dataframe
        labels_list = ["lesion_type", "type1", "type2", "pathology", "assessment", "breast_density", "subtlety"]
        print(os.linesep)
        for label in labels_list:
            if label in df.columns:
                print(df[label].value_counts(sort=True, ascending=False))
                print(os.linesep)

        return self

    def lesion_patches_centered(self, shape: Tuple[int] = (1024, 1024)):
        if self.__patch_transform_selected:
            raise Exception('Patch transform already selected!')
        self.__transform_list.append(centered_patch_transform(shape))
        self.__patch_transform_selected = True
        return self

    def lesion_patches_random(self, shape: Tuple[int] = (1024, 1024), min_overlap=0.9, normal_probability=0.0):
        if self.__patch_transform_selected:
            raise Exception('Patch transform already selected!')
        patch_transform = random_patch_transform(shape, min_overlap)
        if normal_probability > 0:
            self.__plus_normal = True
            patch_transform = normal_patch_transform_wrapper(patch_transform, normal_probability, shape,
                                                             1 - min_overlap)
        self.__transform_list.append(patch_transform)
        self.__patch_transform_selected = True
        return self

    def add_image_transforms(self, transform_list: List, for_train: bool = True, for_val: bool = True):
        self.__image_transform_list.extend(transform_list)
        self.__image_transform_list_applied_training.extend([for_train]*len(transform_list))
        self.__image_transform_list_applied_validation.extend([for_val]*len(transform_list))
        return self

    def split_train_val(self, validation_percentage: float):
        self.__split_cross_validation = False
        self.__split_validation = True
        self.__validation_percentage = validation_percentage
        return self

    def split_cross_validation(self, k_folds=5):
        self.__split_validation = False
        self.__split_cross_validation = True
        self.__cross_validation_folds = k_folds
        return self

    def create_classification(self, attribute: str, mask_input: bool = False):
        self.__fetch_filter_lesions()
        label_list = self.__dataframe[attribute].unique().tolist()
        if self.__plus_normal:
            label_list.append('NORMAL')

        if self.__split_validation:
            shuffled_dataframe = self.__dataframe.sample(frac=1)
            num_samples = len(shuffled_dataframe.index)
            num_validation = int(num_samples * self.__validation_percentage)

            train_dataframe = shuffled_dataframe.iloc[num_validation:, :]

            train_image_transforms = [trans for trans, ft in
                                      zip(self.__image_transform_list, self.__image_transform_list_applied_training) if
                                      ft]
            val_transforms = [trans for trans, fv in
                              zip(self.__image_transform_list, self.__image_transform_list_applied_validation) if fv]

            train_dataset = CBISDDSMClassificationDataset(train_dataframe, self.__config['download_path'], attribute, label_list,
                                                          masks=mask_input, transform=Compose(self.__transform_list),
                                                          train_image_transform=Compose(train_image_transforms),
                                                          test_image_transform=Compose(val_transforms))

            val_dataframe = shuffled_dataframe.iloc[:num_validation, :]
            val_dataset = CBISDDSMClassificationDataset(val_dataframe, self.__config['download_path'], attribute,
                                                        label_list,
                                                        masks=mask_input, transform=Compose(self.__transform_list),
                                                        train_image_transform=Compose(train_image_transforms),
                                                        test_image_transform=Compose(val_transforms))

            if self.__is_sam:
              return ( 
                MAMOSAMDataset(train_dataframe, Compose(self.__transform_list), Compose(self.__image_transform_list)),
                MAMOSAMDataset(val_dataframe  , Compose(self.__transform_list), Compose(self.__image_transform_list)),
              )

            return train_dataset.train_mode(), val_dataset.test_mode()

        elif self.__split_cross_validation:
            shuffled_dataframe = self.__dataframe.sample(frac=1)
            num_samples = len(shuffled_dataframe.index)
            num_sample_per_fold = int(num_samples / self.__cross_validation_folds)
            train_image_transforms = [trans for trans, ft in
                                      zip(self.__image_transform_list, self.__image_transform_list_applied_training)
                                      if ft]
            val_transforms = [trans for trans, fv in
                              zip(self.__image_transform_list, self.__image_transform_list_applied_validation) if
                              fv]

            fold_dataframes = []
            for i in range(self.__cross_validation_folds):
                start_i = i * num_sample_per_fold
                end_i = (i + 1) * num_sample_per_fold
                dataframe = shuffled_dataframe.iloc[start_i:end_i, :]
                fold_dataframes.append(dataframe)

            datasets = []
            for i in range(self.__cross_validation_folds):
                train_dataframe = pd.concat(list(d for ind, d in enumerate(fold_dataframes) if ind != i),
                                            ignore_index=True)
                train_dataset = CBISDDSMClassificationDataset(train_dataframe, self.__config['download_path'],
                                                              attribute, label_list,
                                                              masks=mask_input,
                                                              transform=Compose(self.__transform_list),
                                                              train_image_transform=Compose(train_image_transforms),
                                                              test_image_transform=Compose(val_transforms))

                val_dataframe = fold_dataframes[i]
                val_dataset = CBISDDSMClassificationDataset(val_dataframe, self.__config['download_path'], attribute,
                                                            label_list,
                                                            masks=mask_input, transform=Compose(self.__transform_list),
                                                            train_image_transform=Compose(train_image_transforms),
                                                            test_image_transform=Compose(val_transforms))
                if self.__is_sam:
                  datasets.append((
                    MAMOSAMDataset(train_dataframe, Compose(self.__transform_list), Compose(self.__image_transform_list)),
                    MAMOSAMDataset(val_dataframe  , Compose(self.__transform_list), Compose(self.__image_transform_list))
                  ))
                else:
                  datasets.append((train_dataset.train_mode(), val_dataset.test_mode()))
                        
            return datasets

        else:
            train_dataset = CBISDDSMClassificationDataset(self.__dataframe, self.__config['download_path'], attribute,
                                                          label_list,
                                                          masks=mask_input, transform=Compose(self.__transform_list),
                                                          train_image_transform=Compose(self.__image_transform_list),
                                                          test_image_transform=Compose(self.__image_transform_list))
            if self.__is_sam:
              return MAMOSAMDataset(self.__dataframe, Compose(self.__transform_list), Compose(self.__image_transform_list)), None
            else:
              return train_dataset, None


if __name__ == "__main__":
    dataset = CBISDDSMDatasetFactory('./config.json') \
        .train() \
        .add_masses() \
        .drop_attributes("assessment", "breast_density", "subtlety") \
        .map_attribute_value('pathology', {'BENIGN_WITHOUT_CALLBACK': 'BENIGN'}) \
        .show_counts() \
        .lesion_patches_random(normal_probability=0.8) \
        .create_classification('pathology', mask_input=True)
    dataset.visualize()
    pass
