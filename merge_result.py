import pandas as pd
import config
import os

pd.set_option('display.max_columns', None)

if os.path.exists(config.output_age_file and config.output_gender_file and config.output_education_file):
    df_age = pd.read_csv(config.output_age_file)
    df_gender = pd.read_csv(config.output_gender_file)
    df_education = pd.read_csv(config.output_education_file)

    # df_result = pd.concat([df_age, df_gender, df_education], axis=1)
    df_age_gender = pd.merge(df_age, df_gender, on='ID')
    df_result = pd.merge(df_age_gender, df_education, on='ID')
    print(df_result.shape)
    print(df_result.head())
    df_result.to_csv(config.output + '/result.csv', index=False)
