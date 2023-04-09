import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from toy_data import generate_toy_data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import  matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random
from plot import draw_pie


def load_dataset(name='tadpole', seed=42, pi=2):
    if name == 'tadpole':
        return load_tadpole(seed)
    elif name == 'av45':
        return load_tadpole_AV45(seed)
    elif name == 'adult':
        return load_adult(seed, smaller=True)
    elif name == 'toy':
        return load_toy_test()
    elif name == 'toy_new':
        return load_toy_new(seed, pi)
    elif name == 'toy_3':
        return load_toy_three_group(seed)
    else:
        print('dataset not exist')
        return -1


def load_tadpole_AV45(seed=42, version=1, verbose=False):

    if seed == 0:
        verbose = True
    df_tadpole = pd.read_csv('./datasets/tadpole/TADPOLE_D1_D2.csv')
    df_tadpole_base = df_tadpole[df_tadpole['VISCODE'] == 'bl']
    feature_keywords = ['CAUDALMIDDLEFRONTAL', 'FRONTALPOLE', 'LATERALORBITOFRONTAL',
                        'MEDIALORBITOFRONTAL', 'PARSOPERCULARIS', 'PARSORBITALIS',
                        'PARSTRIANGULARIS', 'ROSTRALMIDDLEFRONTAL', 'SUPERIORFRONTAL',
                        'CAUDALANTERIORCINGULATE', 'ISTHMUSCINGULATE', 'POSTERIORCINGULATE',
                        'ROSTRALANTERIORCINGULATE', 'INFERIORPARIETAL', 'PRECUNEUS',
                        'SUPERIORPARIETAL', 'SUPRAMARGINAL', 'BANKSSTS',
                        'ENTORHINAL', 'FUSIFORM', 'INFERIORTEMPORAL', 'LINGUAL',
                        'MIDDLETEMPORAL', 'PARAHIPPOCAMPAL', 'SUPERIORTEMPORAL',
                        'TEMPORALPOLE', 'TRANSVERSETEMPORAL', 'CUNEUS',
                        'LATERALOCCIPITAL', 'PERICALCARINE', 'PARACENTRAL',
                        'POSTCENTRAL', 'PRECENTRAL']
    prefix1 = 'CTX_LH_'
    prefix2 = 'CTX_RH_'
    size_surfix = '_SIZE_UCBERKELEYAV45_10_17_16'
    cortical_full_name = [prefix1 + i + size_surfix for i in feature_keywords] \
                         + [prefix2 + i + size_surfix for i in feature_keywords]
    demographic = ['PTGENDER']
    label = ['DX_bl']
    df_cort = df_tadpole_base[demographic + cortical_full_name + label]
    df_cort = df_cort.replace(r'^\s*$', np.nan, regex=True)
    df_cort_clean = df_cort.dropna().copy()

    if verbose:
        print('number of all:', len(df_cort_clean))
        print('number of males:', sum(df_cort_clean['PTGENDER'] == 'Male'))
        print('number of females:', sum(df_cort_clean['PTGENDER'] == 'Female'))
        print('number of AD', sum(df_cort_clean['DX_bl'] == 'AD'))
        print('number of MCI', sum(df_cort_clean['DX_bl'].isin(['EMCI', 'LMCI'])))
        print('number of CN', sum(df_cort_clean['DX_bl'].isin(['CN', 'SMC'])))

        count_list = df_cort_clean['PTGENDER'].value_counts()
        draw_pie(count_list, count_list.index)

    y = df_cort_clean['DX_bl'].copy()
    y[y == 'EMCI'] = 'MCI'
    y[y == 'LMCI'] = 'MCI'
    y[y == 'SMC'] = 'CN'

    if verbose:
        count_list = y.value_counts()
        draw_pie(count_list, count_list.index)

        for disease in ['CN', 'MCI', 'AD']:
            idx = y[y == disease].index
            count_list = df_cort_clean.loc[idx]['PTGENDER'].value_counts()
            draw_pie(count_list, count_list.index, disease)

    group = df_cort_clean['PTGENDER']
    pi = 1
    random.seed(seed)
    if version == 0:  # CN vs MCI
        print("AV45 dataset preprocessing ... version 0: CN vs MCI")
        drop_index = y[y == 'AD'].index
        df_cort_clean.drop(drop_index, inplace=True)
        y.drop(drop_index, inplace=True)

        MCI_index = y[y == 'MCI'].index
        male_in_MCI_index = group[group == 'Male'].index & MCI_index
        # drop half of male in MCI
        drop_male_in_MCI_index = random.sample(list(male_in_MCI_index), int(len(male_in_MCI_index) / 2))
        df_cort_clean.drop(drop_male_in_MCI_index, inplace=True)
        y.drop(drop_male_in_MCI_index, inplace=True)
        pi_MCI = len(y[(y == 'MCI') & (group == 'Female')]) / len(y[(y == 'MCI') & (group == 'Male')])

        CN_index = y[y == 'CN'].index
        female_in_CN_index = group[group == 'Female'].index & CN_index
        # drop half of female in CN
        drop_female_in_CN_index = random.sample(list(female_in_CN_index), int(len(female_in_CN_index) / 2))
        df_cort_clean.drop(drop_female_in_CN_index, inplace=True)
        y.drop(drop_female_in_CN_index, inplace=True)
        pi_CN = len(y[(y == 'CN') & (group == 'Male')]) / len(y[(y == 'CN') & (group == 'Female')])

        pi = pi_MCI

        if verbose:
            count_list = y.value_counts()
            draw_pie(count_list, count_list.index, 'processed AV45')

            for disease in ['CN', 'MCI']:
                idx = y[y == disease].index
                count_list = df_cort_clean.loc[idx]['PTGENDER'].value_counts()
                draw_pie(count_list, count_list.index, 'processed ' + disease)

        y[y=='CN'] = 0
        y[y=='MCI'] = 1
    elif version == 1:  # CN vs AD
        print("AV45 dataset preprocessing ... version 1: CN vs AD")
        drop_index = y[y == 'MCI'].index
        df_cort_clean.drop(drop_index, inplace=True)
        y.drop(drop_index, inplace=True)

        AD_index = y[y == 'AD'].index
        male_in_AD_index = group[group == 'Male'].index & AD_index
        # drop half of male in AD
        drop_male_in_AD_index = random.sample(list(male_in_AD_index), int(len(male_in_AD_index) / 2))
        df_cort_clean.drop(drop_male_in_AD_index, inplace=True)
        y.drop(drop_male_in_AD_index, inplace=True)

        CN_index = y[y == 'CN'].index
        female_in_CN_index = group[group == 'Female'].index & CN_index
        # drop half of female in CN
        drop_female_in_CN_index = random.sample(list(female_in_CN_index), int(len(female_in_CN_index) / 2))
        df_cort_clean.drop(drop_female_in_CN_index, inplace=True)
        y.drop(drop_female_in_CN_index, inplace=True)

        pi_AD = len(y[(y == 'AD') & (group == 'Female')]) / len(y[(y == 'AD') & (group == 'Male')])
        pi_CN = len(y[(y == 'CN') & (group == 'Male')]) / len(y[(y == 'CN') & (group == 'Female')])
        pi = pi_AD

        if verbose:
            count_list = y.value_counts()
            draw_pie(count_list, count_list.index, 'processed AV45')
            for disease in ['CN', 'AD']:
                idx = y[y == disease].index
                count_list = df_cort_clean.loc[idx]['PTGENDER'].value_counts()
                draw_pie(count_list, count_list.index, 'processed ' + disease)

        y[y=='CN'] = 0
        y[y=='AD'] = 1
    elif version == 2:  # MCI vs AD
        print("AV45 dataset preprocessing ... version 2: MCI vs AD")
        drop_index = y[y == 'CN'].index
        df_cort_clean.drop(drop_index, inplace=True)
        y.drop(drop_index, inplace=True)

        AD_index = y[y == 'AD'].index
        male_in_AD_index = group[group == 'Male'].index & AD_index
        # drop half of male in AD
        drop_male_in_AD_index = random.sample(list(male_in_AD_index), int(len(male_in_AD_index) / 2))
        df_cort_clean.drop(drop_male_in_AD_index, inplace=True)
        y.drop(drop_male_in_AD_index, inplace=True)
        pi_AD = len(y[(y == 'AD') & (group == 'Female')]) / len(y[(y == 'AD') & (group == 'Male')])

        MCI_index = y[y == 'MCI'].index
        female_in_MCI_index = group[group == 'Female'].index & MCI_index
        # drop half of female in MCI
        drop_female_in_MCI_index = random.sample(list(female_in_MCI_index), int(len(female_in_MCI_index) / 2))
        df_cort_clean.drop(drop_female_in_MCI_index, inplace=True)
        y.drop(drop_female_in_MCI_index, inplace=True)
        pi_MCI = len(y[(y == 'MCI') & (group == 'Male')]) / len(y[(y == 'MCI') & (group == 'Female')])

        pi = pi_AD

        if verbose:
            count_list = y.value_counts()
            draw_pie(count_list, count_list.index, 'processed AV45')
            for disease in ['MCI', 'AD']:
                idx = y[y == disease].index
                count_list = df_cort_clean.loc[idx]['PTGENDER'].value_counts()
                draw_pie(count_list, count_list.index, 'processed ' + disease)

        y[y=='MCI'] = 0
        y[y=='AD'] = 1

    X = df_cort_clean[demographic + cortical_full_name].copy()
    X.loc[X['PTGENDER'] == 'Male', 'PTGENDER'] = 0
    X.loc[X['PTGENDER'] == 'Female', 'PTGENDER'] = 1
    X = StandardScaler().fit_transform(X)
    # X = np.concatenate([group.to_numpy().reshape(-1,1), X], axis=1)
    y = y.astype('float64').to_numpy()
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    sensible_feature_idx = 0

    return X_train, X_test, y_train, y_test, sensible_feature_idx, pi


def load_tadpole(seed=42):
    # DXCHANGE: 1=Stable: NL to NL; 2=Stable: MCI to MCI; 3=Stable: Dementia to Dementia;
    # 4=Conversion: NL to MCI; 5=Conversion: MCI to Dementia; 6=Conversion: NL to Dementia;
    # 7=Reversion: MCI to NL; 8=Reversion: Dementia to MCI; 9=Reversion: Dementia to NL。
    # MCI: DXCHANGE should be 2, 4, 8; AD: DXCHANGE should be 3, 5, 6
    print("Tadpole dataset preprocessing ...")

    race = 'PTRACCAT'  # Am Indian/Alaskan, Asian, Black, Hawaiian/Other PI, More than one, Unknown, White
    gender = 'PTGENDER'  # Male, Female
    features = ['CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate',
                'Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp',
                'FDG', 'AV45', 'ABETA_UPENNBIOMK9_04_19_17',
                'TAU_UPENNBIOMK9_04_19_17', 'PTAU_UPENNBIOMK9_04_19_17',
                'APOE4', 'AGE', 'ADAS13', 'Ventricles']

    df_tadpole = pd.read_csv('datasets/tadpole/TADPOLE_D1_D2.csv')
    df_MCI = df_tadpole[(df_tadpole.DXCHANGE == 2) & (df_tadpole.VISCODE == 'bl')]  # we only pick the baseline visits
    df_AD = df_tadpole[(df_tadpole.DXCHANGE == 3) & (df_tadpole.VISCODE == 'bl')]

    # len_AD = int(1 / 2 * len(df_AD))
    len_AD = int(1 * len(df_AD))
    df_MCIAD = pd.concat([df_MCI, df_AD[:len_AD]])  # select part of AD to make more imbalanced data
    group = df_MCIAD[gender]
    print(f"Grouped Info:\n  {group.value_counts()}")
    group[group == 'Male'] = 0
    group[group == 'Female'] = 1
    print(f"Grouped Info after processing:\n {df_MCIAD[gender].value_counts()}")

    X = df_MCIAD[features]
    X = X.apply(pd.to_numeric, errors='coerce')  # fill all the blank cells with NaN
    X = X.dropna(axis=1, how='all')
    X.fillna(X.mean(), inplace=True)
    X = X.to_numpy()
    X = StandardScaler().fit_transform(X)
    X = np.concatenate([group.to_numpy().reshape(-1,1), X], axis=1)   # put the group vector in the first column
    sensible_feature_idx = 0
    pi = 1
    y = np.concatenate([np.zeros(len(df_MCI)), np.ones(len_AD)])  # class 0 is MCI, class 1 is AD
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    return X_train, X_test, y_train, y_test, sensible_feature_idx, pi


def load_toy_new(seed=42, pi=2):
    print("Toy_new dataset preprocessing ...")
    # pi = pi
    n_samples_low = 200  # number of males
    n_samples = pi * n_samples_low  # number of females
    n_dimensions = 2

    np.random.seed(0)
    varA = 0.8
    aveApos = [-1.0] * n_dimensions
    aveAneg = [1.0] * n_dimensions
    varB = 0.5
    aveBpos = [0.5] * int(n_dimensions / 2) + [-0.5] * int(n_dimensions / 2 + n_dimensions % 2)
    aveBneg = [0.5] * n_dimensions

    X = np.random.multivariate_normal(aveApos, np.diag([varA] * n_dimensions), n_samples)
    X = np.vstack([X, np.random.multivariate_normal(aveAneg, np.diag([varA] * n_dimensions), n_samples_low)])
    X = np.vstack([X, np.random.multivariate_normal(aveBpos, np.diag([varB] * n_dimensions), n_samples_low)])
    X = np.vstack([X, np.random.multivariate_normal(aveBneg, np.diag([varB] * n_dimensions), n_samples)])
    sensible_feature = [1] * (n_samples + n_samples_low) + [-1] * (n_samples_low + n_samples)
    sensible_feature = np.array(sensible_feature)
    sensible_feature.shape = (len(sensible_feature), 1)
    X = np.hstack([X, sensible_feature])
    y = [1] * n_samples + [-1] * n_samples_low + [1] * n_samples_low + [-1] * n_samples
    y = np.array(y)
    sensible_feature_id = len(X[1, :]) - 1
    # idx_A = list(range(0, n_samples+n_samples_low))
    # idx_B = list(range(n_samples+n_samples_low, n_samples*2+n_samples_low*2))

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    return X_train, X_test, y_train, y_test, sensible_feature_id, pi


def load_toy_three_group(seed=42):
    print("Toy_3 dataset preprocessing ...")
    pi_list = [1, 1]
    n_samples_1 = 100  # number of group1
    n_samples_2 = pi_list[0] * n_samples_1  # number of group2
    n_samples_3 = pi_list[1] * n_samples_1  # number of group3
    n_dimensions = 2

    np.random.seed(0)
    var1 = 0.8
    ave1pos = [-1.0] * n_dimensions
    ave1neg = [1.0] * n_dimensions
    var2 = 0.5
    ave2pos = [0.5] * int(n_dimensions / 2) + [-0.5] * int(n_dimensions / 2 + n_dimensions % 2)
    ave2neg = [0.5] * n_dimensions
    var3 = 0.5
    ave3pos = [0.8] * int(n_dimensions / 2) + [-0.8] * int(n_dimensions / 2 + n_dimensions % 2)
    ave3neg = [0.8] * n_dimensions

    X = np.random.multivariate_normal(ave1pos, np.diag([var1] * n_dimensions), n_samples_1)
    X = np.vstack([X, np.random.multivariate_normal(ave1neg, np.diag([var1] * n_dimensions), n_samples_1)])

    X = np.vstack([X, np.random.multivariate_normal(ave2pos, np.diag([var2] * n_dimensions), n_samples_2)])
    X = np.vstack([X, np.random.multivariate_normal(ave2neg, np.diag([var2] * n_dimensions), n_samples_2)])

    X = np.vstack([X, np.random.multivariate_normal(ave3pos, np.diag([var3] * n_dimensions), n_samples_3)])
    X = np.vstack([X, np.random.multivariate_normal(ave3neg, np.diag([var3] * n_dimensions), n_samples_3)])

    sensible_feature = [1] * n_samples_1 * 2 + [2] * n_samples_2 * 2 + [3] * n_samples_3 * 2
    sensible_feature = np.array(sensible_feature)
    sensible_feature.shape = (len(sensible_feature), 1)
    X = np.hstack([X, sensible_feature])
    y = [1] * n_samples_1 + [-1] * n_samples_1 + [1] * n_samples_2 + \
        [-1] * n_samples_2 + [1] * n_samples_3 + [-1] * n_samples_3
    y = np.array(y)
    sensible_feature_id = len(X[1, :]) - 1
    # idx_A = list(range(0, n_samples_2+n_samples_1))
    # idx_B = list(range(n_samples_2+n_samples_1, n_samples_2*2+n_samples_1*2))

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    return X_train, X_test, y_train, y_test, sensible_feature_id, pi_list


def load_adult(seed=42, smaller=False, scaler=True):
    '''
    :param smaller: selecting this flag it is possible to generate a smaller version of the training and test sets.
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.

    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    '''
    print("Adult dataset preprocessing ...")
    data = pd.read_csv(
        "./datasets/adult/adult.data",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"]
            )
    len_train = len(data.values[:, -1])
    data_test = pd.read_csv(
        "./datasets/adult/adult.test",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"]
    )
    data = pd.concat([data, data_test])
    # Considering the relative low portion of missing data, we discard rows with missing data
    domanda = data["workclass"][4].values[1]
    data = data[data["workclass"] != domanda]
    data = data[data["occupation"] != domanda]
    data = data[data["native-country"] != domanda]
    # Here we apply discretisation on column marital_status
    data.replace(['Divorced', 'Married-AF-spouse',
                  'Married-civ-spouse', 'Married-spouse-absent',
                  'Never-married', 'Separated', 'Widowed'],
                 ['not married', 'married', 'married', 'married',
                  'not married', 'not married', 'not married'], inplace=True)
    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
    datamat = data.values
    datamat = shuffle(datamat, random_state=seed)
    target = np.array([-1.0 if val == 0 else 1.0 for val in np.array(datamat)[:, -1]])
    datamat = datamat[:, :-1]
    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)
    if smaller:
        print('A smaller version of the dataset is loaded...')
        data = namedtuple('_', 'data, target')(datamat[:len_train // 20, :-1], target[:len_train // 20])
        data_test = namedtuple('_', 'data, target')(datamat[len_train:, :-1], target[len_train:])
    else:
        print('The dataset is loaded...')
        data = namedtuple('_', 'data, target')(datamat[:len_train, :-1], target[:len_train])
        data_test = namedtuple('_', 'data, target')(datamat[len_train:, :-1], target[len_train:])

    sensible_feature_idx = 9
    pi = 1
    X_train = data.data
    y_train = data.target
    X_test = data_test.data
    y_test = data_test.target

    return X_train, X_test, y_train, y_test, sensible_feature_idx, pi


def load_toy_test():
    # Load toy test
    n_samples = 100 * 2
    n_samples_low = 20 * 2
    n_dimensions = 10
    X, y, sensible_feature_id, _, _ = generate_toy_data(n_samples=n_samples,
                                                        n_samples_low=n_samples_low,
                                                        n_dimensions=n_dimensions)
    data = namedtuple('_', 'data, target')(X, y)
    return data, data


