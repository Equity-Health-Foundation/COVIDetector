#!/usr/bin/env python
# coding: utf-8

import featuretools as ft
import composeml as cp
import pandas
import os
import time
from pathlib import Path
from yaspin import yaspin
from dask.distributed import Client
import dask.dataframe as dd
import argparse


from pdb import set_trace


inputdir = 'release_07-06-2020/evaluation/'
outputdir = '/home/tom/Documents/_ml_data_cache/DREAM-challenge/output2'


def start_dask(n_workers=4, threads_per_worker=1):
    try:
        client.close()
    except:
        pass
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker)
    return client


def csv_to_entityset(input_dir, entityset_id='covid', dask_client=None, blocksize='40MB'):
    # convert the DREAM COVID competition dataset into Featuretools EntitySet for feature engineering
    # Parameters:
    ## input_dir: where the competition dataset CSV files are stored
    ## entityset_id: the EntitySet id to save
    ## dask_client: the Dask client object created by start_dask
    ## blocksize: the blocksize used by dask.dataframe read_csv

    # read all CSV files
    if dask_client is not None:
        assert dask_client.status == 'running', "Error: the Dask client is not running!"
        with yaspin(color="yellow") as spinner:
            spinner.text = "Loading raw data csv files as Dask DataFrames ... "
            condition_occurrence = dd.read_csv(os.path.join(input_dir, "condition_occurrence.csv"), blocksize=blocksize)
            device_exposure = dd.read_csv(os.path.join(input_dir, "device_exposure.csv"), blocksize=blocksize)
            drug_exposure = dd.read_csv(os.path.join(input_dir, "drug_exposure.csv"), blocksize=blocksize)
            goldstandard = dd.read_csv(os.path.join(input_dir, "goldstandard.csv"), blocksize=blocksize)
            measurement = dd.read_csv(os.path.join(input_dir, "measurement.csv"), blocksize=blocksize)
            observation = dd.read_csv(os.path.join(input_dir, "observation.csv"), blocksize=blocksize)
            observation_period = dd.read_csv(os.path.join(input_dir, "observation_period.csv"), blocksize=blocksize)
            person = dd.read_csv(os.path.join(input_dir, "person.csv"), blocksize=blocksize)
            procedure_occurrence = dd.read_csv(os.path.join(input_dir, "procedure_occurrence.csv"), blocksize=blocksize)
            visit_occurrence = dd.read_csv(os.path.join(input_dir, "visit_occurrence.csv"), blocksize=blocksize)
            spinner.ok("Done")
    else:
        with yaspin(color="yellow") as spinner:
            spinner.text = "Loading raw data csv files as pandas DataFrames ... "
            condition_occurrence = pandas.read_csv(os.path.join(input_dir, 'condition_occurrence.csv')).dropna(how='all', axis='columns')
            device_exposure = pandas.read_csv(os.path.join(input_dir, 'device_exposure.csv')).dropna(how='all', axis='columns')
            drug_exposure = pandas.read_csv(os.path.join(input_dir, 'drug_exposure.csv')).dropna(how='all', axis='columns')
            goldstandard = pandas.read_csv(os.path.join(input_dir, "goldstandard.csv"))
            measurement = pandas.read_csv(os.path.join(input_dir, 'measurement.csv')).dropna(how='all', axis='columns')
            observation = pandas.read_csv(os.path.join(input_dir, 'observation.csv')).dropna(how='all', axis='columns')
            observation_period = pandas.read_csv(os.path.join(input_dir, 'observation_period.csv')).dropna(how='all', axis='columns')
            person = pandas.read_csv(os.path.join(input_dir, 'person.csv')).dropna(how='all', axis='columns')
            procedure_occurrence = pandas.read_csv(os.path.join(input_dir, 'procedure_occurrence.csv')).dropna(how='all', axis='columns')
            visit_occurrence = pandas.read_csv(os.path.join(input_dir, 'visit_occurrence.csv')).dropna(how='all', axis='columns')
            spinner.ok("Done")

    with yaspin(color="yellow") as spinner:
        spinner.text = "Bulding EntitySet ... "
        # select only the columns with data
        condition_occurrence = condition_occurrence[['condition_occurrence_id', 'person_id', 'condition_start_datetime',
               'condition_end_datetime', 'condition_concept_id',
               'condition_type_concept_id', 'condition_source_concept_id',
               'condition_status_source_value', 'condition_status_concept_id']]
        device_exposure = device_exposure[['device_exposure_id', 'person_id', 'device_exposure_start_datetime',
                                           'device_exposure_end_datetime']]
        drug_exposure = drug_exposure[['drug_exposure_id', 'person_id', 'drug_concept_id', 'drug_exposure_start_datetime',
               'drug_exposure_end_datetime', 'stop_reason', 'refills', 'quantity', 'days_supply',
               'drug_type_concept_id', 'drug_source_concept_id', 'route_source_value', 'dose_unit_source_value']]
        measurement = measurement[['measurement_id', 'person_id', 'measurement_datetime',
               'value_as_number', 'range_low', 'range_high', 'value_source_value',
               'measurement_concept_id', 'measurement_type_concept_id',
               'operator_concept_id', 'value_as_concept_id', 'unit_concept_id',
               'measurement_source_concept_id', 'unit_source_value']]
        person = person[['person_id', 'birth_datetime', 'gender_concept_id', 'race_concept_id',
               'ethnicity_concept_id', 'location_id', 'gender_source_value',
               'race_source_concept_id', 'ethnicity_source_concept_id']]
        procedure_occurrence = procedure_occurrence[['procedure_occurrence_id', 'person_id', 'procedure_datetime',
               'procedure_concept_id', 'procedure_type_concept_id', 'procedure_source_concept_id']]
        observation = observation[['observation_id', 'person_id', 'observation_concept_id', 'observation_datetime',
               'observation_type_concept_id', 'value_as_number', 'value_as_string', 'value_as_concept_id', 'unit_concept_id',
               'observation_source_concept_id', 'unit_source_value']]
        observation_period = observation_period[['observation_period_id', 'person_id', 'observation_period_start_date',
               'observation_period_end_date']]
        visit_occurrence = visit_occurrence[['person_id', 'visit_concept_id', 'visit_start_datetime',
                                        'visit_end_datetime', 'visit_source_concept_id']]

        # build EntitySet
        covid = ft.EntitySet(id=entityset_id)
        goldstandard = covid.entity_from_dataframe(entity_id="goldstandard",
                                            dataframe=goldstandard,
                                            index='person_id',
                                            variable_types={"status": ft.variable_types.variable.Categorical})
        del(goldstandard)
        covid = covid.entity_from_dataframe(entity_id="condition_occurrence",
                                            dataframe=condition_occurrence,
                                            index='condition_occurrence_id',
                                            time_index='condition_start_datetime',
                                            secondary_time_index={'condition_end_datetime': ['condition_end_datetime']},
                                            variable_types={"condition_occurrence_id": ft.variable_types.variable.Index,
                                                            "person_id": ft.variable_types.variable.Id,
                                                            "condition_start_datetime": ft.variable_types.variable.DatetimeTimeIndex,
                                                            "condition_end_datetime": ft.variable_types.variable.Datetime,
                                                            "condition_concept_id": ft.variable_types.variable.Categorical,
                                                            "condition_type_concept_id": ft.variable_types.variable.Categorical,
                                                            "condition_source_concept_id": ft.variable_types.variable.Categorical,
                                                            "condition_status_source_value": ft.variable_types.variable.Categorical,
                                                            "condition_status_concept_id": ft.variable_types.variable.Categorical
                                                           })
        del(condition_occurrence)
        covid = covid.entity_from_dataframe(entity_id="device_exposure",
                                            dataframe=device_exposure,
                                            index='device_exposure_id',
                                            time_index='device_exposure_start_datetime',
                                            secondary_time_index={'device_exposure_end_datetime': ['device_exposure_end_datetime']},
                                            variable_types={'device_exposure_id': ft.variable_types.Index,
                                                            "person_id": ft.variable_types.Id,
                                                            "device_exposure_start_datetime": ft.variable_types.variable.DatetimeTimeIndex,
                                                            "device_exposure_end_datetime": ft.variable_types.Datetime})
        del(device_exposure)
        covid = covid.entity_from_dataframe(entity_id="drug_exposure",
                                            dataframe=drug_exposure,
                                            index='drug_exposure_id',
                                            time_index='drug_exposure_start_datetime',
                                            secondary_time_index={'drug_exposure_end_datetime': ['drug_exposure_end_datetime',
                                                                  "refills","quantity","days_supply","stop_reason"]},
                                            variable_types={"drug_exposure_id": ft.variable_types.Index,
                                                            "person_id": ft.variable_types.Id,
                                                            "drug_exposure_start_datetime": ft.variable_types.DatetimeTimeIndex,
                                                            "drug_exposure_end_datetime": ft.variable_types.Datetime,
                                                            "refills": ft.variable_types.Numeric,
                                                            "quantity": ft.variable_types.Numeric,
                                                            "days_supply": ft.variable_types.Numeric,
                                                            "drug_concept_id": ft.variable_types.Categorical,
                                                            "drug_type_concept_id": ft.variable_types.Categorical,
                                                            "stop_reason": ft.variable_types.Categorical,
                                                            "drug_source_concept_id": ft.variable_types.Categorical,
                                                            "route_source_value": ft.variable_types.Categorical,
                                                            "dose_unit_source_value": ft.variable_types.Categorical
                                                           })
        del(drug_exposure)
        covid = covid.entity_from_dataframe(entity_id="measurement",
                                            dataframe=measurement,
                                            index='measurement_id',
                                            time_index='measurement_datetime',
                                            variable_types={"measurement_id": ft.variable_types.Index,
                                                            "person_id": ft.variable_types.Id,
                                                            "measurement_datetime": ft.variable_types.DatetimeTimeIndex,
                                                            "value_as_number": ft.variable_types.Numeric,
                                                            "range_low": ft.variable_types.Numeric,
                                                            "range_high": ft.variable_types.Numeric,
                                                            "value_source_value": ft.variable_types.Numeric,
                                                            "measurement_concept_id": ft.variable_types.Categorical,
                                                            "measurement_type_concept_id": ft.variable_types.Categorical,
                                                            "operator_concept_id": ft.variable_types.Categorical,
                                                            "value_as_concept_id": ft.variable_types.Categorical,
                                                            "unit_concept_id": ft.variable_types.Categorical,
                                                            "measurement_source_concept_id": ft.variable_types.Categorical,
                                                            "unit_source_value": ft.variable_types.Categorical
                                                           })
        del(measurement)
        covid = covid.entity_from_dataframe(entity_id="observation",
                                            dataframe=observation,
                                            index='observation_id',
                                            time_index='observation_datetime',
                                            variable_types={"observation_id": ft.variable_types.Categorical,
                                                            "person_id": ft.variable_types.Id,
                                                            "observation_concept_id": ft.variable_types.Categorical,
                                                            "observation_datetime": ft.variable_types.DatetimeTimeIndex,
                                                            "observation_type_concept_id": ft.variable_types.Categorical,
                                                            "value_as_number": ft.variable_types.Numeric,
                                                            "value_as_string": ft.variable_types.Categorical,
                                                            "value_as_concept_id": ft.variable_types.Categorical,
                                                            "observation_source_concept_id": ft.variable_types.Categorical,
                                                            "unit_source_value": ft.variable_types.Categorical,
                                                            "unit_concept_id": ft.variable_types.Categorical,
                                                            "unit_source_value": ft.variable_types.Categorical
                                                           })
        del(observation)
        covid = covid.entity_from_dataframe(entity_id="observation_period",
                                            dataframe=observation_period,
                                            index='observation_period_id',
                                            time_index='observation_period_start_date',
                                            secondary_time_index={'observation_period_end_date': ['observation_period_end_date']},
                                            variable_types={"person_id": ft.variable_types.Id,
                                                            "observation_period_start_date": ft.variable_types.DatetimeTimeIndex,
                                                            "observation_period_end_date": ft.variable_types.Datetime
                                                           })
        del(observation_period)
        covid = covid.entity_from_dataframe(entity_id="person",
                                            dataframe=person,
                                            index='person_id',
                                            time_index='birth_datetime',
                                            variable_types={"person_id": ft.variable_types.variable.Index,
                                                            "birth_datetime": ft.variable_types.variable.DateOfBirth,
                                                            "gender_concept_id": ft.variable_types.variable.Categorical,
                                                            "race_concept_id": ft.variable_types.variable.Categorical,
                                                            "ethnicity_concept_id": ft.variable_types.variable.Categorical,
                                                            "location_id": ft.variable_types.variable.Id,
                                                            "gender_source_value": ft.variable_types.variable.Categorical,
                                                            "race_source_concept_id": ft.variable_types.variable.Categorical,
                                                            "ethnicity_source_concept_id": ft.variable_types.variable.Categorical
                                                           })
        del(person)
        covid = covid.entity_from_dataframe(entity_id="procedure_occurrence",
                                            dataframe=procedure_occurrence,
                                            index='procedure_occurrence_id',
                                            time_index='procedure_datetime',
                                            variable_types={"procedure_occurrence_id": ft.variable_types.Index,
                                                            "person_id": ft.variable_types.Id,
                                                            "procedure_datetime": ft.variable_types.DatetimeTimeIndex,
                                                            "procedure_concept_id": ft.variable_types.Categorical,
                                                           "procedure_type_concept_id": ft.variable_types.Categorical,
                                                           "procedure_source_concept_id": ft.variable_types.Categorical
                                                           })
        del(procedure_occurrence)
        covid = covid.entity_from_dataframe(entity_id="visit_occurrence",
                                            dataframe=visit_occurrence,
                                            index='visit_occurrence_id',
                                            time_index='visit_start_datetime',
                                            secondary_time_index={'visit_end_datetime': ['visit_end_datetime']},
                                            variable_types={"person_id": ft.variable_types.Id,
                                                            "visit_start_datetime": ft.variable_types.DatetimeTimeIndex,
                                                            "visit_end_datetime": ft.variable_types.Datetime,
                                                            "visit_source_concept_id": ft.variable_types.Categorical,
                                                            "visit_concept_id": ft.variable_types.Categorical
                                                           })
        del(visit_occurrence)
        covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['condition_occurrence']['person_id']))
        covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['device_exposure']['person_id']))
        covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['drug_exposure']['person_id']))
        covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['measurement']['person_id']))
        covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['observation']['person_id']))
        covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['procedure_occurrence']['person_id']))
        covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['observation_period']['person_id']))
        covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['visit_occurrence']['person_id']))
        spinner.ok("Done")

    return covid


def entityset_to_parquet(entityset, output):
    # Save a Featuretools EntitySet as parquet files
    # they can be loaded with Featuretools's read_entityset function later
    with yaspin(color="yellow") as spinner:
        spinner.text = "Saving EntitySet {} to parquet files at {} ... ".format(entityset.id, output)
        entityset.to_parquet(output, engine='pyarrow')
        spinner.ok("Done")


def gen_feature_matrix(entityset, features_only=False, outputdir='.',
                    out_feature_matrix_file=None, saved_features=None, saved_encoded_features=None):
    # build features
    if saved_features is None and saved_encoded_features is None:
        with yaspin(color="yellow") as spinner:
            spinner.text = "No features definition file specified, calculating feature matrix from ground zero ... "
            feature_defs = ft.dfs(entityset=entityset, target_entity="person",
                                              # agg_primitives=['avg_time_between', 'count', 'all', 'entropy', 'last', 'num_unique', #'n_most_common',
                                              #                 'min', 'std', 'median', 'mean', 'percent_true', 'trend', 'sum', 'time_since_last', 'any',
                                              #                 'num_true', 'time_since_first', 'first', 'max', 'mode', 'skew'],
                                              # use only the agg_primitives currently supported by Dask
                                              agg_primitives=['count', 'all', 'num_unique', #'n_most_common',
                                                              'min', 'std', 'mean', 'percent_true', 'sum', 'any',
                                                              'num_true', 'max'],
                                              trans_primitives=[],
                                              features_only=True)
            spinner.write("> generated {} features".format(len(feature_defs)))
            if features_only:
                return feature_defs

            tic = time.perf_counter()
            feature_matrix = ft.calculate_feature_matrix(feature_defs, entityset)
            try:
                feature_matrix = feature_matrix.compute()
            except:
                pass
            toc = time.perf_counter()
            spinner.write(f"> feature matrix calculate completed in {toc - tic:0.4f} seconds")
            # set_trace()
            feature_matrix_enc, features_enc = ft.encode_features(feature_matrix, feature_defs)
            spinner.write("> generated {} encoded features and the feature matrix".format(len(features_enc)))

            ft.save_features(feature_defs, os.path.join(outputdir, "feature_defs.json"))
            spinner.write("> saved features to definition file {}".format("feature_defs.json"))

            ft.save_features(features_enc, os.path.join(outputdir, "features_enc.json"))
            spinner.write("> saved encoded features to definition file {}".format("feature_enc.json"))
            spinner.ok("Done")
    else:
        if saved_features is None:
            with yaspin(text="Using saved encoded features from {} ... ".format(saved_encoded_features), color="yellow") as spinner:
                features_enc = ft.load_features(saved_encoded_features)
                spinner.write("> {} encoded features loaded from {}".format(len(features_enc), saved_encoded_features))
                tic = time.perf_counter()
                feature_matrix_enc = ft.calculate_feature_matrix(features_enc, entityset)
                try:
                    feature_matrix_enc = feature_matrix_enc.compute()
                except:
                    pass
                toc = time.perf_counter()
                spinner.write(f"> encoded feature matrix calculate completed in {toc - tic:0.4f} seconds")
                spinner.ok("Done")
        else:
            with yaspin(text="Using saved features from {} ... ".format(saved_features), color="yellow") as spinner:
                feature_defs = ft.load_features(saved_features)
                spinner.write("> {} features loaded from {}".format(len(feature_defs), saved_features))

                tic = time.perf_counter()
                feature_matrix = ft.calculate_feature_matrix(feature_defs, entityset)
                try:
                    feature_matrix = feature_matrix.compute()
                except:
                    pass
                toc = time.perf_counter()
                spinner.write(f"> feature matrix calculate complete in {toc - tic:0.4f} seconds")

                tic = time.perf_counter()
                feature_matrix_enc, features_enc = ft.encode_features(feature_matrix, feature_defs)
                toc = time.perf_counter()
                spinner.write(f"> encoded feature matrix calculate completed in {toc - tic:0.4f} seconds")
                spinner.ok("Done")

    try:
        goldstandard = entityset['goldstandard'].df.compute()
    except:
        goldstandard = entityset['goldstandard'].df
    feature_matrix_enc = feature_matrix_enc.merge(goldstandard, on='person_id')
    if out_feature_matrix_file is not None:
        with yaspin(text="Saving feature matrix to {} ... ".format(out_feature_matrix_file), color="yellow") as spinner:
            feature_matrix_enc.to_csv(os.path.join(outputdir, out_feature_matrix_file), index=False)
            spinner.ok("Done")
    return feature_matrix_enc, features_enc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''This is a data processing and
                        feature enginering tool for the DREAM-COVID challenge dataset''')
    parser.add_argument('-i', '--input_dir', help="where the dataset CSV files are stored")
    parser.add_argument('-e', '--entityset_id', default='covid',
                        help="the featuretools EntitySet id to save")
    parser.add_argument('-o', '--output_entityset', default=None,
                        help="the path to save the newly generated EntitySet files")

    parser.add_argument('-p', '--input_es_dir', help="where the EntitySet parquet files are stored")
    parser.add_argument('-f', '--feature_file', default=None,
                        help="the path to save the engineered feature files")
    parser.add_argument('-m', '--feature_matrix_path', default=None,
                        help="the path to save the engineered feature files and one-hot encoded feature matrix file")

    parser.add_argument('-d', '--dask', action='store_true', default=False,
                        help="use Dask to process the DataFrames")
    # parser.add_argument('--no-dask', dest='dask', action='store_false')
    parser.add_argument('-n', '--n_workers', type=int, default=4,
                        help="the number of Dask workers (processes) to launch")
    parser.add_argument('-t', '--threads_per_worker', type=int, default=1,
                        help="the number of threads per Dask worker")

    p = parser.parse_args()

    if p.input_es_dir is None:
        if p.dask:
            client = start_dask(n_workers=p.n_workers, threads_per_worker=p.threads_per_worker)
            es = csv_to_entityset(input_dir=p.input_dir, entityset_id=p.entityset_id, dask_client=client)
        else:
            es = csv_to_entityset(input_dir=p.input_dir, entityset_id=p.entityset_id)
        if p.output_entityset is not None:
            entityset_to_parquet(es, output=p.output_entityset)
    else:
        if p.dask:
            client = start_dask(n_workers=p.n_workers, threads_per_worker=p.threads_per_worker)
        es = ft.read_entityset(p.input_es_dir)
        if p.output_entityset is not None:
            error('input entityset path detected, no need to output entityset!')

    if p.feature_matrix_path is not None:
        # outputdir = os.path.dirname(p.feature_matrix_path)
        # if os.path.isfile(p.feature_matrix_path):
        #     feature_matrix_file = os.path.basename(p.feature_matrix_path)
        # else:
        feature_matrix_file = "feature_matrix_enc.csv"
        Path(p.feature_matrix_path).mkdir(parents=True, exist_ok=True)
        gen_feature_matrix(es, outputdir=p.feature_matrix_path, out_feature_matrix_file=feature_matrix_file, saved_features=p.feature_file)
