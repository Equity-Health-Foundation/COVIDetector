#!/usr/bin/env python
# coding: utf-8

import featuretools as ft
# import composeml as cp
import pandas
import os
import time
# from pathlib import Path
from yaspin import yaspin
from dask.distributed import Client
import dask.dataframe as dd
import dask.array as array
import argparse

# os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask
# import modin.pandas as pandas

# from pdb import set_trace


def start_dask(n_workers=4, threads_per_worker=1):
    try:
        client.close()
    except:
        pass
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker)
    return client


def csv_to_entityset(input_dir, entityset_id='covid', goldstandard_exist=True):
    # convert the DREAM COVID competition dataset into Featuretools EntitySet for feature engineering
    # Parameters:
    ## input_dir: where the competition dataset CSV files are stored
    ## entityset_id: the EntitySet id to save

    with yaspin(color="yellow") as spinner:
        spinner.write("Loading raw data csv files as pandas DataFrames ... ")
        # build EntitySet while read csvs
        covid = ft.EntitySet(id=entityset_id)
        condition_occurrence = pandas.read_csv(os.path.join(input_dir, 'condition_occurrence.csv')).dropna(how='all', axis='columns')
        condition_occurrence = condition_occurrence[['condition_occurrence_id', 'person_id', 'condition_start_datetime',
               'condition_end_datetime', 'condition_concept_id',
               'condition_type_concept_id', 'condition_source_concept_id',
               'condition_status_source_value', 'condition_status_concept_id']]
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

        device_exposure = pandas.read_csv(os.path.join(input_dir, 'device_exposure.csv')).dropna(how='all', axis='columns')
        device_exposure = device_exposure[['device_exposure_id', 'person_id', 'device_exposure_start_datetime',
                                           'device_exposure_end_datetime']]
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

        drug_exposure = pandas.read_csv(os.path.join(input_dir, 'drug_exposure.csv')).dropna(how='all', axis='columns')
        drug_exposure = drug_exposure[['drug_exposure_id', 'person_id', 'drug_concept_id', 'drug_exposure_start_datetime',
               'drug_exposure_end_datetime', 'stop_reason', 'refills', 'quantity', 'days_supply',
               'drug_type_concept_id', 'drug_source_concept_id', 'route_source_value', 'dose_unit_source_value']]
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

        if goldstandard_exist:
            goldstandard = pandas.read_csv(os.path.join(input_dir, "goldstandard.csv"))
            goldstandard = covid.entity_from_dataframe(entity_id="goldstandard",
                                                dataframe=goldstandard,
                                                index='person_id',
                                                variable_types={"status": ft.variable_types.variable.Categorical})
            del(goldstandard)

        measurement = pandas.read_csv(os.path.join(input_dir, 'measurement.csv')).dropna(how='all', axis='columns')
        measurement = measurement[['measurement_id', 'person_id', 'measurement_datetime',
               'value_as_number', 'range_low', 'range_high', 'value_source_value',
               'measurement_concept_id', 'measurement_type_concept_id',
               'operator_concept_id', 'value_as_concept_id', 'unit_concept_id',
               'measurement_source_concept_id', 'unit_source_value']]
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

        observation = pandas.read_csv(os.path.join(input_dir, 'observation.csv')).dropna(how='all', axis='columns')
        observation = observation[['observation_id', 'person_id', 'observation_concept_id', 'observation_datetime',
               'observation_type_concept_id', 'value_as_number', 'value_as_string', 'value_as_concept_id', 'unit_concept_id',
               'observation_source_concept_id', 'unit_source_value']]
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

        observation_period = pandas.read_csv(os.path.join(input_dir, 'observation_period.csv')).dropna(how='all', axis='columns')
        observation_period = observation_period[['observation_period_id', 'person_id', 'observation_period_start_date',
               'observation_period_end_date']]
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

        person = pandas.read_csv(os.path.join(input_dir, 'person.csv')).dropna(how='all', axis='columns')
        person = person[['person_id', 'birth_datetime', 'gender_concept_id', 'race_concept_id',
               'ethnicity_concept_id', 'location_id', 'gender_source_value',
               'race_source_concept_id', 'ethnicity_source_concept_id']]
        covid = covid.entity_from_dataframe(entity_id="person",
                                            dataframe=person,
                                            index='person_id',
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

        procedure_occurrence = pandas.read_csv(os.path.join(input_dir, 'procedure_occurrence.csv')).dropna(how='all', axis='columns')
        procedure_occurrence = procedure_occurrence[['procedure_occurrence_id', 'person_id', 'procedure_datetime',
               'procedure_concept_id', 'procedure_type_concept_id', 'procedure_source_concept_id']]
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

        visit_occurrence = pandas.read_csv(os.path.join(input_dir, 'visit_occurrence.csv')).dropna(how='all', axis='columns')
        visit_occurrence = visit_occurrence[['person_id', 'visit_concept_id', 'visit_start_datetime',
                                        'visit_end_datetime', 'visit_source_concept_id']]
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

        # add relationships
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


def csv_to_dask_entityset(input_dir, entityset_id='covid', dask_client=None,
                          blocksize='40MB', partition_on=None, n_partition=None):
    # convert the DREAM COVID competition dataset into Featuretools EntitySet backed by Dask for feature engineering
    # Parameters:
    ## input_dir: where the competition dataset CSV files are stored
    ## entityset_id: the EntitySet id to save
    ## dask_client: the Dask client object created by start_dask
    ## blocksize: the blocksize used by dask.dataframe read_csv

    # read all CSV files
    if dask_client is not None:
        assert dask_client.status == 'running', "Error: the Dask client is not running!"
    else:
        error('Error: Need a running Dask client!')

    with yaspin(color="yellow") as spinner:
        spinner.write("Loading raw data csv files as Dask DataFrames ... ")
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

    # filter columns with information
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
    # set_trace()
    if (partition_on is not None) and (n_partition is not None):
        partition_col = person[['person_id']].compute()
        partition_col['partition'] = array.random.randint(n_partition, size=len(person))

        condition_occurrence = condition_occurrence.merge(partition_col, 'left', on='person_id')
        device_exposure = device_exposure.merge(partition_col, 'left', on='person_id')
        drug_exposure = drug_exposure.merge(partition_col, 'left', on='person_id')
        goldstandard = goldstandard.merge(partition_col, 'left', on='person_id')
        measurement = measurement.merge(partition_col, 'left', on='person_id')
        observation = observation.merge(partition_col, 'left', on='person_id')
        observation_period = observation_period.merge(partition_col, 'left', on='person_id')
        person = person.merge(partition_col, 'left', on='person_id')
        procedure_occurrence = procedure_occurrence.merge(partition_col, 'left', on='person_id')
        visit_occurrence = visit_occurrence.merge(partition_col, 'left', on='person_id')

        # build EntitySet
        covid = ft.EntitySet(id=entityset_id)
        goldstandard = covid.entity_from_dataframe(entity_id="goldstandard",
                                            dataframe=goldstandard,
                                            index='person_id',
                                            variable_types={"status": ft.variable_types.variable.Categorical,
                                                            "partition": ft.variable_types.variable.Id})
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
                                                            "condition_status_concept_id": ft.variable_types.variable.Categorical,
                                                            "partition": ft.variable_types.variable.Id
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
                                                            "device_exposure_end_datetime": ft.variable_types.Datetime,
                                                            "partition": ft.variable_types.variable.Id})
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
                                                            "dose_unit_source_value": ft.variable_types.Categorical,
                                                            "partition": ft.variable_types.variable.Id
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
                                                            "unit_source_value": ft.variable_types.Categorical,
                                                            "partition": ft.variable_types.variable.Id
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
                                                            "unit_source_value": ft.variable_types.Categorical,
                                                            "partition": ft.variable_types.variable.Id
                                                           })
        del(observation)
        covid = covid.entity_from_dataframe(entity_id="observation_period",
                                            dataframe=observation_period,
                                            index='observation_period_id',
                                            time_index='observation_period_start_date',
                                            secondary_time_index={'observation_period_end_date': ['observation_period_end_date']},
                                            variable_types={"person_id": ft.variable_types.Id,
                                                            "observation_period_start_date": ft.variable_types.DatetimeTimeIndex,
                                                            "observation_period_end_date": ft.variable_types.Datetime,
                                                            "partition": ft.variable_types.variable.Id
                                                           })
        del(observation_period)
        covid = covid.entity_from_dataframe(entity_id="person",
                                            dataframe=person,
                                            index='person_id',
                                            variable_types={"person_id": ft.variable_types.variable.Index,
                                                            "birth_datetime": ft.variable_types.variable.DateOfBirth,
                                                            "gender_concept_id": ft.variable_types.variable.Categorical,
                                                            "race_concept_id": ft.variable_types.variable.Categorical,
                                                            "ethnicity_concept_id": ft.variable_types.variable.Categorical,
                                                            "location_id": ft.variable_types.variable.Id,
                                                            "gender_source_value": ft.variable_types.variable.Categorical,
                                                            "race_source_concept_id": ft.variable_types.variable.Categorical,
                                                            "ethnicity_source_concept_id": ft.variable_types.variable.Categorical,
                                                            "partition": ft.variable_types.variable.Id
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
                                                            "procedure_source_concept_id": ft.variable_types.Categorical,
                                                            "partition": ft.variable_types.variable.Id
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
                                                            "visit_concept_id": ft.variable_types.Categorical,
                                                            "partition": ft.variable_types.variable.Id
                                                           })
        del(visit_occurrence)
    else:
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

    # add relationships
    covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['condition_occurrence']['person_id']))
    covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['device_exposure']['person_id']))
    covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['drug_exposure']['person_id']))
    covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['measurement']['person_id']))
    covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['observation']['person_id']))
    covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['procedure_occurrence']['person_id']))
    covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['observation_period']['person_id']))
    covid = covid.add_relationship(ft.Relationship(covid['person']['person_id'], covid['visit_occurrence']['person_id']))

    return covid


def entityset_to_parquet(entityset, output, partition_on=None):
    # Save a Featuretools EntitySet as parquet files
    # they can be loaded with Featuretools's read_entityset function later
    with yaspin(color="yellow") as spinner:
        spinner.write("Saving EntitySet {} to parquet files at {} ... ".format(entityset.id, output))
        if partition_on is not None:
            entityset.to_parquet(output, engine='pyarrow', partition_on=partition_on)
        else:
            entityset.to_parquet(output, engine='pyarrow')
        spinner.ok("Done")


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
    # agg_primitives_all=['avg_time_between', 'count', 'all', 'entropy', 'last', 'num_unique', #'n_most_common',
    #             'min', 'std', 'median', 'mean', 'percent_true', 'trend', 'sum', 'time_since_last', 'any',
    #             'num_true', 'time_since_first', 'first', 'max', 'mode', 'skew']
    # agg_primitives_dask=['count', 'all', 'num_unique', #'n_most_common',
    #               'min', 'std', 'mean', 'percent_true', 'sum', 'any',
    #               'num_true', 'max']

    agg_primitives = ['mean', 'max', 'min', 'std']#'last', 'skew', 'time_since_last', 'trend'] # trend takes extremely long time to compute
    trans_primitives = []#['age']
    groupby_trans_primitives = []
    include_variables = {'measurement': ['measurement_datetime', 'value_as_number',
                                         'measurement_concept_id'],
                         'person': ['birth_datetime']
                        }
    include_entities = ['person']
    primitive_options = {#tuple(trans_primitives): {'include_entities': include_entities},
                         tuple(agg_primitives): {'include_variables': include_variables}
                        }
    ignore_entities = [goldstandard_id, 'condition_occurrence', 'device_exposure', 'drug_exposure',
                       'observation', 'observation_period', 'procedure_occurrence', 'visit_occurrence']
    ignore_variables = {}
    where_primitives = agg_primitives
    entityset['measurement']['measurement_concept_id'].interesting_values = entityset[
                                                        'measurement'].df['measurement_concept_id'].unique()

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
                                  agg_primitives=agg_primitives,
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
            spinner.write("Using saved features from {} ... ".format(saved_encoded_features))
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
