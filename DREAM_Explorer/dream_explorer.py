#!/usr/bin/env python
# coding: utf-8

# Run this app with `python DREAM_explorer.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import plotly.express as px
import pandas
import featuretools as ft
import os
import argparse
# import pdb


app = dash.Dash(external_stylesheets=[dbc.themes.MINTY])


input_groups = html.Div([
        dbc.Row([
            dbc.Col(html.Div(
                dbc.InputGroup([
                        dbc.InputGroupAddon("input the patient id", addon_type="prepend"),
                        dbc.Input(id="patient_id_input", type="number", value=1, debounce=True,
                                    style={'width': 1})
                ], className="mb-3")
            ), width=3)
        ])
])

header = dbc.Alert([
        html.H4("DREAM COVID Explorer", className="alert-heading"),
        html.P(),
        html.Hr(),
        input_groups,
    ])

app.layout = html.Div(children=[
    header,
    dcc.Loading(
        id="loading-1",
        type="graph",
        fullscreen=True,
        children=[
            dbc.Table(id='patient_info'),
            dcc.Graph(id='patient_timelines'),
            dcc.Graph(id='patient_measurement')
        ]),
])


@app.callback([Output("patient_info", "children"),
                Output('patient_timelines', 'figure'),
                Output('patient_measurement', 'figure')],
              [Input("patient_id_input", "value")])
def input_triggers_spinner(patient_id):
    table = update_patient_info(patient_id)
    fig1 = update_figure_patient_timelines(patient_id)
    fig2 = update_figure_patient_measurement(patient_id)
    return table, fig1, fig2


def update_patient_info(patient_id):
    df_person = entityset['person'].df
    df = df_person[df_person['person_id'] == patient_id].compute()
    table = dbc.Table.from_dataframe(df, striped=True, bordered=False, borderless=True,
                                    hover=True, size='sm')
    return table


def update_figure_patient_timelines(patient_id):
    df_condition_occurrence = entityset['condition_occurrence'].df
    condition_occurrence = df_condition_occurrence[df_condition_occurrence['person_id'] == patient_id].compute().merge(
            data_dict, 'left', left_on='condition_concept_id', right_on='concept_id')[[
                'condition_start_datetime', 'condition_end_datetime', 'condition_status_source_value', 'concept_name']]
    condition_occurrence.rename(columns={'condition_start_datetime':'start_datetime', 'condition_end_datetime':'end_datetime',
                                    'condition_status_source_value':'Y'}, inplace=True)
    condition_occurrence['facet']='condition_occurrence'

    df_drug_exposure = entityset['drug_exposure'].df
    drug_exposure = df_drug_exposure[df_drug_exposure['person_id']== patient_id].compute().merge(
            data_dict, 'left', left_on='drug_concept_id', right_on='concept_id')[[
                'drug_exposure_start_datetime', 'drug_exposure_end_datetime', 'refills', 'quantity', 'days_supply',
                'stop_reason', 'route_source_value', 'dose_unit_source_value', 'concept_name']]
    drug_exposure.rename(columns={'drug_exposure_start_datetime':'start_datetime', 'drug_exposure_end_datetime':'end_datetime',
                             'route_source_value':'Y'}, inplace=True)
    drug_exposure['facet']='drug_exposure'

    df_visit_occurrence = entityset['visit_occurrence'].df
    visit_occurrence = df_visit_occurrence[df_visit_occurrence['person_id']== patient_id].compute().merge(
            data_dict, 'left', left_on='visit_concept_id', right_on='concept_id')[[
                'visit_start_datetime', 'visit_end_datetime', 'concept_name']]
    visit_occurrence.rename(columns={'visit_start_datetime':'start_datetime', 'visit_end_datetime':'end_datetime'
                             }, inplace=True)
    visit_occurrence['Y'] =  visit_occurrence['concept_name']
    visit_occurrence['facet']='visit_occurrence'

    unified_df = pandas.concat([condition_occurrence, drug_exposure, visit_occurrence])
    unified_df.loc[unified_df.concept_name.isna(), 'concept_name'] = 'NA'

    fig = px.timeline(unified_df, x_start="start_datetime", x_end="end_datetime",
                  y="Y", facet_row='facet', color='concept_name', hover_name='concept_name',
                  range_x=['2010-01-01','2020-06-30'])
    fig.update_layout(showlegend=False, height=1000)
    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(matches=None, autorange="reversed") # otherwise tasks are listed from the bottom up

    return fig


def update_figure_patient_measurement(patient_id):
    df_measurement = entityset['measurement'].df
    measurement = df_measurement[df_measurement['person_id']== patient_id].compute().merge(
        data_dict, 'left', left_on='measurement_concept_id', right_on='concept_id')[['measurement_datetime', 'range_low',
            'range_high', 'value_as_number', 'unit_source_value', 'concept_name']]
    measurement['value_percentile'] = (measurement['value_as_number']-measurement['range_low'])/(
            measurement['range_high']-measurement['range_low'])
    measurement.rename(columns={'measurement_datetime':'datetime'}, inplace=True)
    measurement['facet'] = 'measurement'

    df_observation = entityset['observation'].df
    observation = df_observation[df_observation['person_id']== patient_id].compute().merge(
            data_dict, 'left', left_on='observation_concept_id', right_on='concept_id')[['observation_datetime',
                'value_as_number', 'value_as_string', 'unit_source_value', 'concept_name']]
    observation['value_percentile'] = (observation['value_as_number']-90)/10
    observation.loc[observation.value_as_string == 'Yes', 'value_percentile'] = 1
    observation.loc[observation.value_as_string == 'No', 'value_percentile'] = 0
    observation.loc[observation.value_as_string == 'Never', 'value_percentile'] = 0
    observation.rename(columns={'observation_datetime':'datetime'}, inplace=True)
    observation['facet']='observation'

    unified_df = pandas.concat([measurement, observation])
    unified_df.loc[unified_df.concept_name.isna(), 'concept_name'] = 'NA'

    fig = px.scatter(unified_df, x="datetime", y="value_percentile", color="concept_name",
                hover_name='concept_name', hover_data=['concept_name', 'value_percentile',
                    'value_as_number', 'range_low', 'range_high'],
                facet_row='facet')
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(matches=None)

    return fig



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is a patient history explorer for DREAM-COVID challenge dataset')
    # parser = argparse.ArgumentParser(version='1.0')
    parser.add_argument('-d', '--datadict', help="the path to the data_dictionary.csv")
    parser.add_argument('-e', '--entityset', help="the path to the EntitySet files generated by feature_engineering.py")
    args = parser.parse_args()

    data_dict = pandas.read_csv(args.datadict)
    entityset = ft.read_entityset(args.entityset)
    # data_dict = pandas.read_csv('/media/tom/Data6T/datasets/DREAM-challenge/data_dictionary.csv')
    # entityset = ft.read_entityset('/home/tom/Documents/_ml_data_cache/DREAM-challenge/covid_EntitySet_cleaned.parquet')
    app.run_server(port=8050, host='0.0.0.0')
