#!/usr/bin/env python
# coding: utf-8

import featuretools as ft
# import composeml as cp
import pandas
import os
import time
# from pathlib import Path
from yaspin import yaspin
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import dask.array as array
import argparse

# os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask
# import modin.pandas as pandas

# from pdb import set_trace


def gen_feature_matrix(entityset, features_only=False, feature_matrix_encode=False,
                       saved_features=None):
    '''A function compute and return (feature_matrix, feature_defs) from an featuretools EntitySet

    entityset: the EntitySet to compute features from
    features_only: only return feature_defs, do not actually compute the feature_matrix
    feature_matrix_encode: whether return encoded feature_matrix (Categorical variable one-hot)
    saved_features: load a pre defined feature file and compute feature_matrix based on it
    '''

    if 'goldstandard' in entityset.entity_dict.keys():
        goldstandard_exist = True
        goldstandard_id = 'goldstandard'
    else:
        goldstandard_exist = False
        goldstandard_id = None
    ##FIX manual partition by person_id does NOT improve Dask computing performance
    # ignore 'partition' columns in every entity when building features
    # ignore_variables = dict()
    # for entity in entityset.entities:
    #     if 'partition' in [v.name for v in entity.variables]:
    #         ignore_variables[entity.id] = ['partition']

    ##CAUTION when the entityset is backed by Dask dataframes, only limited set of primitives are supported
    # agg_primitives_all=['avg_time_between', 'count', 'all', 'entropy', 'last', 'num_unique', 'n_most_common',
    #             'min', 'std', 'median', 'mean', 'percent_true', 'trend', 'sum', 'time_since_last', 'any',
    #             'num_true', 'time_since_first', 'first', 'max', 'mode', 'skew']
    # agg_primitives_dask=['count', 'all', 'num_unique', #'n_most_common',
    #               'min', 'std', 'mean', 'percent_true', 'sum', 'any',
    #               'num_true', 'max']

    ## define features per entity(table)
    agg_primitives = ['mean', 'max', 'min', 'std', 'last', 'skew', 'time_since_last'] # 'trend' # trend takes extremely long time to compute
    include_variables = {'measurement': ['measurement_datetime', 'value_as_number', 'measurement_concept_id'],
                         'observation':['observation_concept_id', 'observation_datetime', 'value_as_number']}
    agg_primitives_device_exposure = ['count', 'avg_time_between', 'time_since_first']
    include_entities_device_exposure = ['device_exposure']

    trans_primitives = ['age']
    groupby_trans_primitives = []
    include_entities = ['person']
    primitive_options = {tuple(trans_primitives): {'include_entities': include_entities},
                         tuple(agg_primitives): {'include_variables': include_variables},
                         tuple(agg_primitives_device_exposure): {'include_entities': include_entities_device_exposure},
                        }
    ignore_entities = [goldstandard_id, 'condition_occurrence', 'drug_exposure',
                       'observation_period', 'procedure_occurrence', 'visit_occurrence']
    ignore_variables = {}
    where_primitives = agg_primitives
    entityset['measurement']['measurement_concept_id'].interesting_values = entityset[
                                                        'measurement'].df['measurement_concept_id'].unique()
    entityset['observation']['observation_concept_id'].interesting_values = entityset[
                                                        'observation'].df['observation_concept_id'].unique()
    # if isinstance(entityset.entities[0].df, pandas.DataFrame):
    #     agg_primitives = agg_primitives_all
    # else:
    #     agg_primitives = agg_primitives_dask

    # build features
    if saved_features is None:
        with yaspin(color="yellow") as spinner:
            spinner.write("No features definition file specified, calculating feature matrix from ground zero ... ")
            feature_defs = ft.dfs(entityset=entityset, target_entity="person",
                                  features_only=True,
                                  agg_primitives=agg_primitives+agg_primitives_device_exposure,
                                  trans_primitives=trans_primitives,
                                  groupby_trans_primitives=groupby_trans_primitives,
                                  primitive_options=primitive_options,
                                  ignore_entities=ignore_entities,
                                  ignore_variables=ignore_variables,
                                  where_primitives=where_primitives,
                                  max_depth=2)
            spinner.write("> generated {} features".format(len(feature_defs)))
            if features_only:
                return feature_defs

            tic = time.perf_counter()
            feature_matrix = ft.calculate_feature_matrix(feature_defs, entityset)
            if isinstance(entityset.entities[0].df, dd.DataFrame):
                feature_matrix = feature_matrix.compute()
            toc = time.perf_counter()
            spinner.write(f"> feature matrix calculate completed in {toc - tic:0.4f} seconds")
            if feature_matrix_encode:
                feature_matrix_enc, features_enc = ft.encode_features(feature_matrix, feature_defs)
                spinner.write("> generated {} encoded features and the feature matrix".format(len(features_enc)))
            spinner.ok("Done")
    else:
        with yaspin(color="yellow") as spinner:
            spinner.write("Using saved features from {} ... ".format(saved_features))
            feature_defs = ft.load_features(saved_features)
            spinner.write("> {} features loaded from {}".format(len(feature_defs), saved_features))

            tic = time.perf_counter()
            feature_matrix = ft.calculate_feature_matrix(feature_defs, entityset)
            if isinstance(entityset.entities[0].df, dd.DataFrame):
                feature_matrix = feature_matrix.compute()
            toc = time.perf_counter()
            spinner.write(f"> feature matrix calculate complete in {toc - tic:0.4f} seconds")
            spinner.ok("Done")

    if goldstandard_exist:
        if isinstance(entityset.entities[0].df, dd.DataFrame):
            goldstandard = entityset['goldstandard'].df.compute()
        else:
            goldstandard = entityset['goldstandard'].df
    if feature_matrix_encode:
        feature_matrix = feature_matrix_enc
    if goldstandard_exist:
        feature_matrix = feature_matrix.merge(goldstandard, on='person_id', how='right')

    return feature_matrix, feature_defs
