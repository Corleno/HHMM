import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle


def Load_pos_Z_test():
    with open("model_res/p_post.pickle", "rb") as res:
        Z_pos = pickle.load(res)
    Z_pos = np.array(Z_pos)
    return Z_pos


def Load_data():
    data_location = '../../data/survey_data/'
    with open(data_location + "processed_survey_data.pkl", "rb") as f:
        survey_data = pickle.load(f)
    with open(data_location + "mcmcPatientIDs", "rb") as f:
        ID_data = pickle.load(f)
    ID_data = np.array(ID_data)
    return ID_data, survey_data


if __name__ == "__main__":
    ID_data, survey_data = Load_data()
    Z_pos = Load_pos_Z_test()

    survey_ID = survey_data['ID'].values.reshape(-1)
    survey_X = survey_data['X']

    # common IDs
    common_IDs = np.array(list(set(ID_data).intersection(set(survey_ID))))
    ID2z_pos = {id:z_pos for id, z_pos in zip(ID_data, Z_pos)}
    Z_pos_reg = pd.DataFrame(data=np.array([ID2z_pos[id] for id in common_IDs]), columns=['Z_pos'])
    index = np.array([np.where(survey_ID == id)[0][0] for id in common_IDs])
    survey_X_reg = survey_X.iloc[index,:].reset_index(drop=True).copy()
    common_IDs_df = pd.DataFrame(data=common_IDs, columns=['ID'])
    survey = pd.concat([common_IDs_df, survey_X_reg, Z_pos_reg], axis=1)

    # survey.columns
    # 'marital_status', 'school', 'smoke_py', 'snus', 'drink', 'sixdrinks',
    #        'num_preg', 'hormon_contr', 'morn_contr', 'condom', 'sex',
    #        'age_partner', 'youngpartner', 'chlamydia', 'herpes', 'trichomonas',
    #        'gonorrhoea', 'gw', 'vaccine', 'x_drink_a', 'Z_pos'

    f = 'Z_pos ~ C(marital_status) + C(school) + C(smoke_py) + C(snus) + C(drink) + C(sixdrinks) + C(num_preg) + C(hormon_contr) + C(morn_contr) + C(condom) + C(sex) + C(age_partner) + C(youngpartner) + C(chlamydia) + C(herpes) + C(trichomonas) + C(gonorrhoea) + C(gw) + C(vaccine) + x_drink_a'
    res = smf.logit(formula=f, data=survey).fit()
    print(res.summary())

    data_location = '../../data/survey_data/'
    with open(data_location + "processed_survey_result.pkl", "wb") as f:
        pickle.dump(survey, f)
    breakpoint()


