import streamlit as st
import pandas as pd
from pandas import read_csv
import numpy as np
import random
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Heart Attack Complication Prediction Program",
    page_icon=":heart:",
    layout="wide",
    menu_items={
        'About': "# Heart Attack Complication Prediction Program \n *Author*: Christopher Nuzum"
    }
)

st.markdown(
    """
    <style>
        code {
            font-family: monospace;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

predict_tab, graphs_tab, build_tab = st.tabs(["Predictor", "Descriptive Methods", "Build Process"])


def clean_data(dataset_):
    dataset = dataset_.copy()

    with build_tab:
        st.write(dataset.head())
        st.write("Is 'ID' column unique? ", dataset['ID'].is_unique)
        st.write("Setting 'ID' column as index...")
        dataset.set_index('ID', inplace=True)
        st.write(dataset.head())
        st.write("duplicate rows: ", dataset.duplicated().sum())
        st.write("dataset shape (rows, cols): ", dataset.shape)

        row_count = dataset.shape[0]
        null_entries = dataset.isna().sum()
        st.write("count of missing values per column: ", null_entries)
        st.write("removing cols missing 50% or more values...")
        removed_cols = []
        for col, null_count in null_entries.items():
            if null_count > row_count * 0.5:
                removed_cols.append([col, null_count])
                dataset.drop(columns=col, inplace=True)
        r_cols = pd.DataFrame(data=removed_cols, columns=['Column', 'Missing Values'])
        r_cols.set_index('Column', inplace=True)
        st.write(r_cols)
        st.write("dataset shape (rows, cols): ", dataset.shape)

        # converting the fatality/reason column to simply survived (0)/died (1)
        replace_dict = {'LET_IS': {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}}
        dataset.replace(to_replace=replace_dict, inplace=True)

        dataset.fillna(-1, inplace=True)

    return dataset


def show_model_performance(x_, y_):
    class Scores:
        score = -1
        tns = 0
        fps = 0
        fns = 0
        tps = 0

        def __init__(self):
            self.class_accuracy_scores = [0 for _ in range(12)]

        def cm_accuracy(self):
            return (self.tps + self.tns) / (self.tps + self.fps + self.tns + self.fns)

        def cm_recall(self):
            if self.tps + self.fns == 0:
                return 0
            else:
                return self.tps / (self.tps + self.fns)

        def cm_precision(self):
            if self.tps + self.fps == 0:
                return 0
            else:
                return self.tps / (self.tps + self.fps)

        def cm_f1(self):
            if self.cm_precision() + self.cm_recall() == 0:
                return 0
            else:
                return 2 * (self.cm_precision() * self.cm_recall()) / (self.cm_precision() + self.cm_recall())

    random_vals = range(5)
    scores = [Scores() for _ in range(len(random_vals))]

    model = DecisionTreeClassifier(max_depth=39, random_state=1)

    for rv in random_vals:
        x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.2, random_state=rv)
        model.fit(x_train, y_train)

        scores[rv].score = model.score(x_test, y_test)
        y_predict = model.predict(x_test)

        for _c_index, _class in enumerate(y_predict.T):
            scores[rv].class_accuracy_scores[_c_index] += accuracy_score(y_test.T[_c_index], y_predict.T[_c_index])

        cm = multilabel_confusion_matrix(y_test, y_predict)
        for _c_index, _class in enumerate(cm):
            # confusion matrix format:
            # [ TN, FP ]
            # [ FN, TP ]
            scores[rv].tns += _class[0][0]
            scores[rv].fps += _class[0][1]
            scores[rv].fns += _class[1][0]
            scores[rv].tps += _class[1][1]

    results_string = ""
    _tot_score = 0
    _tot_tns = 0
    _tot_fps = 0
    _tot_fns = 0
    _tot_tps = 0
    _tot_cm_acc = 0
    _tot_cm_rec = 0
    _tot_cm_prec = 0
    _tot_cm_f1 = 0
    _tot_class_accuracies = [0 for _ in range(12)]
    for _rv in scores:
        _tot_score += _rv.score
        _tot_tns += _rv.tns
        _tot_fps += _rv.fps
        _tot_fns += _rv.fns
        _tot_tps += _rv.tps
        _tot_cm_acc += _rv.cm_accuracy()
        _tot_cm_rec += _rv.cm_recall()
        _tot_cm_prec += _rv.cm_precision()
        _tot_cm_f1 += _rv.cm_f1()
        for _c_index, _class_score in enumerate(_rv.class_accuracy_scores):
            _tot_class_accuracies[_c_index] += _class_score
    _avg_score = _tot_score / len(scores)
    _avg_tns = int(_tot_tns / len(scores))
    _avg_fps = int(_tot_fps / len(scores))
    _avg_fns = int(_tot_fns / len(scores))
    _avg_tps = int(_tot_tps / len(scores))
    _avg_cm_acc = _tot_cm_acc / len(scores)
    _avg_cm_rec = _tot_cm_rec / len(scores)
    _avg_cm_prec = _tot_cm_prec / len(scores)
    _avg_cm_f1 = _tot_cm_f1 / len(scores)
    _avg_class_accuracies = [0 for _ in range(12)]
    for _class_index, _class_score in enumerate(_tot_class_accuracies):
        _avg_class_accuracies[_class_index] = _class_score / len(scores)

    results_string += f"max_depth 39: avg - {_avg_score:.2f} cm_acc - {_avg_cm_acc:.3f} cm_recall {_avg_cm_rec:.3f} cm_precision {_avg_cm_prec:.3f} cm_f1 {_avg_cm_f1:.3f}   ---   tps {_avg_tps:3.0f} fps {_avg_fps:3.0f} \
tns {_avg_tns:4.0f} fns {_avg_fns:3.0f}\n"
    results_string += f"       accuracy scores for each class: "
    for _index, _avg in enumerate(_avg_class_accuracies):
        results_string += f"{_index} - {_avg:.2f}   "
    results_string += f"\n"

    with build_tab:
        st.write("Decision Tree")
        st.code(results_string)

        _avg_class_acc_df = pd.DataFrame([clean_dataset.columns[-12:], _avg_class_accuracies], index=["Class", "Accuracy"])
        line_fig = px.line(_avg_class_acc_df.T, x="Class", y="Accuracy", title="Accuracies for Each Class", markers=True)
        st.plotly_chart(line_fig)


if __name__ == '__main__':
    raw_dataset = read_csv("mi_data.csv")
    clean_dataset = clean_data(raw_dataset)
    array = clean_dataset.values

    x = array[:, :-12]  # all columns except last 12
    y = array[:, -12:]  # last 12 columns

    show_model_performance(x, y)

    ml_model = DecisionTreeClassifier(max_depth=39, random_state=1)
    ml_model.fit(x, y)

    with graphs_tab:
        complication_names = []
        complication_counts = []
        for _col_index, _col in enumerate(y.T):
            complication_names.append(clean_dataset.columns[-12 + _col_index])
            complication_counts.append(_col.sum())
        no_complications_patients = 0
        for _row in y:
            if _row.sum() == 0:
                no_complications_patients += 1
        complication_names.append("None")
        complication_counts.append(no_complications_patients)

        _col1, _col2 = st.columns([2, 1])
        with _col1:
            st.subheader("Frequency of Complications")
            st.plotly_chart(px.pie(names=complication_names, values=complication_counts))
            comps = pd.DataFrame(y, columns=clean_dataset.columns[-12:])
            comps["Sex"] = x.T[1]
            comps.replace(to_replace={'Sex': {0: 'female', 1: 'male'}}, inplace=True)
            st.subheader("Frequency of Complications By Gender")
            fig = px.histogram(comps, x='Sex', y=["FIBR_PREDS", "PREDS_TAH", "JELUD_TAH", "FIBR_JELUD", "A_V_BLOK", "OTEK_LANC", "RAZRIV", "DRESSLER", "ZSN", "REC_IM", "P_IM_STEN", "LET_IS"], barmode='group', height=400)
            st.plotly_chart(fig)
        with _col2:
            complication_descriptions = pd.DataFrame([["FIBR_PREDS", "Atrial fibrillation"], ["PREDS_TAH", "Supraventricular tachycardia"], ["JELUD_TAH", "Ventricular tachycardia"],
                                                      ["FIBR_JELUD", "Ventricular fibrillation"], ["A_V_BLOK", "Third-degree AV block"], ["OTEK_LANC", "Pulmonary edema"], ["RAZRIV", "Myocardial rupture"],
                                                      ["DRESSLER", "Dressler syndrome"], ["ZSN", "Chronic heart failure"], ["REC_IM", "Relapse of the myocardial infarction"], ["P_IM_STEN", "Post-infarction angina"],
                                                      ["LET_IS", "Lethal outcome"]], columns=["Variable Name", "Complication Description"])
            complication_descriptions.set_index("Variable Name", inplace=True)
            for _ in range(13):
                st.write(" ")
            st.dataframe(complication_descriptions, height=458, width=380)

        st.subheader("Importance of Features in Predicting Complications")
        feature_importance = pd.DataFrame(data=ml_model.feature_importances_, index=clean_dataset.columns[:-12])
        st.bar_chart(feature_importance, height=700)

        st.write("10 Most Important Features")
        feature_importance_with_desc = feature_importance.copy()
        feature_importance_with_desc["Description"] = ["Age", "Gender", "Quantity of myocardial infarctions in the anamnesis", "Exertional angina pectoris in the anamnesis",
                                                       "Functional class (FC) of angina pectoris in the last year", "Coronary heart disease (CHD) in recent weeks, days before admission to hospital",
                                                       "Presence of an essential hypertension", "Symptomatic hypertension", "Duration of arterial hypertension", "Presence of chronic Heart failure (HF) in the anamnesis",
                                                       "Observing of arrhythmia in the anamnesis", "Premature atrial contractions in the anamnesis", "Premature ventricular contractions in the anamnesis",
                                                       "Paroxysms of atrial fibrillation in the anamnesis", "A persistent form of atrial fibrillation in the anamnesis", "Ventricular fibrillation in the anamnesis",
                                                       "Ventricular paroxysmal tachycardia in the anamnesis", "First-degree AV block in the anamnesis", "Third-degree AV block in the anamnesis",
                                                       "LBBB (anterior branch) in the anamnesis", "Incomplete LBBB in the anamnesis", "Complete LBBB in the anamnesis", "Incomplete RBBB in the anamnesis",
                                                       "Complete RBBB in the anamnesis", "Diabetes mellitus in the anamnesis", "Obesity in the anamnesis", "Thyrotoxicosis in the anamnesis",
                                                       "Chronic bronchitis in the anamnesis", "Obstructive chronic bronchitis in the anamnesis", "Bronchial asthma in the anamnesis", "Chronic pneumonia in the anamnesis",
                                                       "Pulmonary tuberculosis in the anamnesis", "Systolic blood pressure according to intensive care unit", "Diastolic blood pressure according to intensive care unit",
                                                       "Pulmonary edema at the time of admission to intensive care unit", "Cardiogenic shock at the time of admission to intensive care unit",
                                                       "Paroxysms of atrial fibrillation at the time of admission to intensive care unit, (or at a pre-hospital stage)",
                                                       "Paroxysms of supraventricular tachycardia at the time of admission to intensive care unit, (or at a pre-hospital stage)",
                                                       "Paroxysms of ventricular tachycardia at the time of admission to intensive care unit, (or at a pre-hospital stage)",
                                                       "Ventricular fibrillation at the time of admission to intensive care unit, (or at a pre-hospital stage)",
                                                       "Presence of an anterior myocardial infarction (left ventricular) (ECG changes in leads V1 – V4 )",
                                                       "Presence of a lateral myocardial infarction (left ventricular) (ECG changes in leads V5 – V6 , I, AVL)",
                                                       "Presence of an inferior myocardial infarction (left ventricular) (ECG changes in leads III, AVF, II).",
                                                       "Presence of a posterior myocardial infarction (left ventricular) (ECG changes in V7 – V9, reciprocity changes in leads V1 – V3)",
                                                       "Presence of a right ventricular myocardial infarction", "ECG rhythm at the time of admission to hospital – sinus (with a heart rate 60-90)",
                                                       "ECG rhythm at the time of admission to hospital – atrial fibrillation", "ECG rhythm at the time of admission to hospital – atrial",
                                                       "ECG rhythm at the time of admission to hospital – idioventricular",
                                                       "ECG rhythm at the time of admission to hospital – sinus with a heart rate above 90 (tachycardia)",
                                                       "ECG rhythm at the time of admission to hospital – sinus with a heart rate below 60 (bradycardia)",
                                                       "Premature atrial contractions on ECG at the time of admission to hospital", "Frequent premature atrial contractions on ECG at the time of admission to hospital",
                                                       "Premature ventricular contractions on ECG at the time of admission to hospital",
                                                       "Frequent premature ventricular contractions on ECG at the time of admission to hospital",
                                                       "Paroxysms of atrial fibrillation on ECG at the time of admission to hospital", "Persistent form of atrial fibrillation on ECG at the time of admission to hospital",
                                                       "Paroxysms of supraventricular tachycardia on ECG at the time of admission to hospital",
                                                       "Paroxysms of ventricular tachycardia on ECG at the time of admission to hospital", "Ventricular fibrillation on ECG at the time of admission to hospital",
                                                       "Sinoatrial block on ECG at the time of admission to hospital", "First-degree AV block on ECG at the time of admission to hospital",
                                                       "Type 1 Second-degree AV block (Mobitz I/Wenckebach) on ECG at the time of admission to hospital",
                                                       "Type 2 Second-degree AV block (Mobitz II/Hay) on ECG at the time of admission to hospital", "Third-degree AV block on ECG at the time of admission to hospital",
                                                       "LBBB (anterior branch) on ECG at the time of admission to hospital", "LBBB (posterior branch) on ECG at the time of admission to hospital",
                                                       "Incomplete LBBB on ECG at the time of admission to hospital", "Complete LBBB on ECG at the time of admission to hospital",
                                                       "Incomplete RBBB on ECG at the time of admission to hospital", "Complete RBBB on ECG at the time of admission to hospital",
                                                       "Fibrinolytic therapy by Сеliasum 750k IU", "Fibrinolytic therapy by Сеliasum 1m IU", "Fibrinolytic therapy by Сеliasum 3m IU", "Fibrinolytic therapy by Streptase",
                                                       "Fibrinolytic therapy by Сеliasum 500k IU", "Fibrinolytic therapy by Сеliasum 250k IU", "Fibrinolytic therapy by Streptodecase 1.5m IU", "Hypokalemia ( < 4 mmol/L)",
                                                       "Serum potassium content", "Increase of sodium in serum (more than 150 mmol/L)", "Serum sodium content", "Serum AlAT content", "Serum AsAT content",
                                                       "White blood cell count", "ESR (Erythrocyte sedimentation rate)", "Time elapsed from the beginning of the attack of CHD to the hospital",
                                                       "Relapse of the pain in the first 24 hours of the hospital period", "Relapse of the pain in the second day of the hospital period",
                                                       "Relapse of the pain in the third day of the hospital period", "Use of opioid drugs by the Emergency Cardiology Team",
                                                       "Use of NSAIDs by the Emergency Cardiology Team", "Use of lidocaine by the Emergency Cardiology Team", "Use of liquid nitrates in the ICU",
                                                       "Use of opioid drugs in the ICU in the first 24 hours of the hospital period", "Use of opioid drugs in the ICU in the second day of the hospital period",
                                                       "Use of opioid drugs in the ICU in the third day of the hospital period", "Use of NSAIDs in the ICU in the first 24 hours of the hospital period",
                                                       "Use of NSAIDs in the ICU in the second day of the hospital period", "Use of NSAIDs in the ICU in the third day of the hospital period",
                                                       "Use of lidocaine in the ICU", "Use of beta-blockers in the ICU", "Use of calcium channel blockers in the ICU", "Use of а anticoagulants (heparin) in the ICU",
                                                       "Use of acetylsalicylic acid in the ICU", "Use of Ticlid in the ICU", "Use of Trental in the ICU"]
        st.write(feature_importance_with_desc.sort_values(0, ascending=False)[:10])

    int_number_input_keys = ['age_input', 's_ad_orit_input', 'd_ad_orit_input', 'na_blood_input', 'roe_input']
    float_number_input_keys = ['k_blood_input', 'alt_blood_input', 'ast_blood_input', 'l_blood_input']

    for key in int_number_input_keys:
        if key not in st.session_state:
            st.session_state[key] = 0

    for key in float_number_input_keys:
        if key not in st.session_state:
            st.session_state[key] = 0.0

    def clear_inputs():
        st.session_state.age_input = 0
        st.session_state.s_ad_orit_input = 0
        st.session_state.d_ad_orit_input = 0
        st.session_state.k_blood_input = 0.0
        st.session_state.na_blood_input = 0
        st.session_state.alt_blood_input = 0.0
        st.session_state.ast_blood_input = 0.0
        st.session_state.l_blood_input = 0.0
        st.session_state.roe_input = 0

    def randomize_inputs():
        st.session_state.age_input = random.randrange(26, 93)
        st.session_state.s_ad_orit_input = random.randrange(0, 260)
        st.session_state.d_ad_orit_input = random.randrange(0, 190)
        st.session_state.k_blood_input = random.uniform(2.3, 8.2)
        st.session_state.na_blood_input = random.randrange(117, 169)
        st.session_state.alt_blood_input = random.uniform(0.03, 3.00)
        st.session_state.ast_blood_input = random.uniform(0.04, 2.15)
        st.session_state.l_blood_input = random.uniform(2.0, 27.9)
        st.session_state.roe_input = random.randrange(1, 140)

    with predict_tab:
        _c1, _c2 = st.columns([1, 5])
        with _c1:
            clear = st.button("Clear Numeric Fields", on_click=clear_inputs)
        with _c2:
            st.button("Randomize Numeric Fields", on_click=randomize_inputs)

        with st.form("my_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Use the 'Randomize Numeric Inputs' button to automatically fill in random data to the 9 numeric input fields.")
                AGE = st.number_input("Age (years)", max_value=130, key='age_input')
                S_AD_ORIT = st.number_input("Systolic blood pressure according to intensive care unit (mmHg)", max_value=300, key='s_ad_orit_input')
                D_AD_ORIT = st.number_input("Diastolic blood pressure according to intensive care unit (mmHg)", max_value=300, key='d_ad_orit_input')
                K_BLOOD = st.number_input("Serum potassium content (mmol/L)", max_value=10.0, step=0.1, format="%.1f", key='k_blood_input')
                NA_BLOOD = st.number_input("Serum sodium content (mmol/L)", max_value=200, key='na_blood_input')
                ALT_BLOOD = st.number_input("Serum AlAT content (IU/L)", max_value=4.00, key='alt_blood_input')
                AST_BLOOD = st.number_input("Serum AsAT content (IU/L)", max_value=3.00, key='ast_blood_input')
                L_BLOOD = st.number_input("White blood cell count (billions per liter)", max_value=34.0, step=0.1, format="%.1f", key='l_blood_input')
                ROE = st.number_input("ESR (Erythrocyte sedimentation rate) (mm/hr)", max_value=200, key='roe_input')
                SEX = st.selectbox(label="Gender", options=('0 – female', '1 – male'), format_func=lambda e: e[4:])[0]
                INF_ANAM = st.selectbox(label="Quantity of myocardial infarctions in the anamnesis", options=('0 – zero', '1 – one', '2 – two', '3 – three and more'),
                                        format_func=lambda e: e[4:])[0]
                STENOK_AN = st.selectbox(label="Exertional angina pectoris in the anamnesis", options=(
                    '0 – never', '1 – during the last year', '2 – one year ago', '3 – two years ago', '4 – three years ago', '5 – 4-5 years ago',
                    '6 – more than 5 years ago'), format_func=lambda e: e[4:])[0]
                FK_STENOK = st.selectbox(label="Functional class (FC) of angina pectoris in the last year",
                                         options=('0 – there is no angina pectoris', '1 – I FC', '2 – II FC', '3 – III FC', '4 – IV FC'), format_func=lambda e: e[4:])[0]
                IBS_POST = st.selectbox(label="Coronary heart disease (CHD) in recent weeks, days before admission to hospital",
                                        options=('0 – there was no СHD', '1 – exertional angina pectoris', '2 – unstable angina pectoris'),
                                        format_func=lambda e: e[4:])[0]
                GB = st.selectbox(label="Presence of an essential hypertension", options=('0 – there is no essential hypertension', '1 – Stage 1', '2 – Stage 2', '3 – Stage 3'),
                                  format_func=lambda e: e[4:])[0]
                SIM_GIPERT = st.selectbox(label="Symptomatic hypertension", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                DLIT_AG = st.selectbox(label="Duration of arterial hypertension", options=(
                    '0 – there was no arterial hypertension', '1 – one year', '2 – two years', '3 – three years', '4 – four years', '5 – five years', '6 – 6-10 years',
                    '7 – more than 10 years'), format_func=lambda e: e[4:])[0]
                ZSN_A = st.selectbox(label="Presence of chronic Heart failure (HF) in the anamnesis", options=(
                    '0 – there is no chronic heart failure', '1 – I stage', '2 – IIА stage (heart failure due to right ventricular systolic dysfunction)',
                    '3 – IIА stage (heart failure due to left ventricular systolic dysfunction)',
                    '4 – IIB stage (heart failure due to left and right ventricular systolic dysfunction)'), format_func=lambda e: e[4:])[0]
                nr_11 = st.selectbox(label="Observing of arrhythmia in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                nr_01 = st.selectbox(label="Premature atrial contractions in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                nr_02 = st.selectbox(label="Premature ventricular contractions in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                nr_03 = st.selectbox(label="Paroxysms of atrial fibrillation in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                nr_04 = st.selectbox(label="A persistent form of atrial fibrillation in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                nr_07 = st.selectbox(label="Ventricular fibrillation in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                nr_08 = st.selectbox(label="Ventricular paroxysmal tachycardia in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                np_01 = st.selectbox(label="First-degree AV block in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                np_04 = st.selectbox(label="Third-degree AV block in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                np_05 = st.selectbox(label="LBBB (anterior branch) in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                np_07 = st.selectbox(label="Incomplete LBBB in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                np_08 = st.selectbox(label="Complete LBBB in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                np_09 = st.selectbox(label="Incomplete RBBB in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                np_10 = st.selectbox(label="Complete RBBB in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                endocr_01 = st.selectbox(label="Diabetes mellitus in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                endocr_02 = st.selectbox(label="Obesity in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                endocr_03 = st.selectbox(label="Thyrotoxicosis in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]

            with col2:
                zab_leg_01 = st.selectbox(label="Chronic bronchitis in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                zab_leg_02 = st.selectbox(label="Obstructive chronic bronchitis in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                zab_leg_03 = st.selectbox(label="Bronchial asthma in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                zab_leg_04 = st.selectbox(label="Chronic pneumonia in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                zab_leg_06 = st.selectbox(label="Pulmonary tuberculosis in the anamnesis", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                O_L_POST = \
                    st.selectbox(label="Pulmonary edema at the time of admission to intensive care unit", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                K_SH_POST = \
                    st.selectbox(label="Cardiogenic shock at the time of admission to intensive care unit", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                MP_TP_POST = st.selectbox(label="Paroxysms of atrial fibrillation at the time of admission to intensive care unit, (or at a pre-hospital stage)",
                                          options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                SVT_POST = st.selectbox(label="Paroxysms of supraventricular tachycardia at the time of admission to intensive care unit, (or at a pre-hospital stage)",
                                        options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                GT_POST = st.selectbox(label="Paroxysms of ventricular tachycardia at the time of admission to intensive care unit, (or at a pre-hospital stage)",
                                       options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                FIB_G_POST = st.selectbox(label="Ventricular fibrillation at the time of admission to intensive care unit, (or at a pre-hospital stage)",
                                          options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                ant_im = st.selectbox(label="Presence of an anterior myocardial infarction (left ventricular) (ECG changes in leads V1 – V4 )", options=(
                    '0 – there is no infarct in this location', '1 – QRS has no changes', '2 – QRS is like QR-complex', '3 – QRS is like Qr-complex',
                    '4 – QRS is like QS-complex'), format_func=lambda e: e[4:])[0]
                lat_im = st.selectbox(label="Presence of a lateral myocardial infarction (left ventricular) (ECG changes in leads V5 – V6 , I, AVL)", options=(
                    '0 – there is no infarct in this location', '1 – QRS has no changes', '2 – QRS is like QR-complex', '3 – QRS is like Qr-complex',
                    '4 – QRS is like QS-complex'), format_func=lambda e: e[4:])[0]
                inf_im = st.selectbox(label="Presence of an inferior myocardial infarction (left ventricular) (ECG changes in leads III, AVF, II).", options=(
                    '0 – there is no infarct in this location', '1 – QRS has no changes', '2 – QRS is like QR-complex', '3 – QRS is like Qr-complex',
                    '4 – QRS is like QS-complex'), format_func=lambda e: e[4:])[0]
                post_im = st.selectbox(label="Presence of a posterior myocardial infarction (left ventricular) (ECG changes in V7 – V9, reciprocity changes in leads V1 – V3)",
                                       options=('0 – there is no infarct in this location', '1 – QRS has no changes', '2 – QRS is like QR-complex', '3 – QRS is like Qr-complex',
                                                '4 – QRS is like QS-complex'), format_func=lambda e: e[4:])[0]
                IM_PG_P = st.selectbox(label="Presence of a right ventricular myocardial infarction", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                ritm_ecg_p_01 = st.selectbox(label="ECG rhythm at the time of admission to hospital – sinus (with a heart rate 60-90)", options=('0 – no', '1 – yes'),
                                             format_func=lambda e: e[4:])[0]
                ritm_ecg_p_02 = st.selectbox(label="ECG rhythm at the time of admission to hospital – atrial fibrillation", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                ritm_ecg_p_04 = st.selectbox(label="ECG rhythm at the time of admission to hospital – atrial", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                ritm_ecg_p_06 = st.selectbox(label="ECG rhythm at the time of admission to hospital – idioventricular", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                ritm_ecg_p_07 = st.selectbox(label="ECG rhythm at the time of admission to hospital – sinus with a heart rate above 90 (tachycardia)",
                                             options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                ritm_ecg_p_08 = st.selectbox(label="ECG rhythm at the time of admission to hospital – sinus with a heart rate below 60 (bradycardia)",
                                             options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                n_r_ecg_p_01 = \
                    st.selectbox(label="Premature atrial contractions on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'),
                                 format_func=lambda e: e[4:])[0]
                n_r_ecg_p_02 = st.selectbox(label="Frequent premature atrial contractions on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'),
                                            format_func=lambda e: e[4:])[0]
                n_r_ecg_p_03 = st.selectbox(label="Premature ventricular contractions on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'),
                                            format_func=lambda e: e[4:])[0]
                n_r_ecg_p_04 = \
                    st.selectbox(label="Frequent premature ventricular contractions on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'),
                                 format_func=lambda e: e[4:])[0]
                n_r_ecg_p_05 = \
                    st.selectbox(label="Paroxysms of atrial fibrillation on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'),
                                 format_func=lambda e: e[4:])[0]
                n_r_ecg_p_06 = st.selectbox(label="Persistent form of atrial fibrillation on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'),
                                            format_func=lambda e: e[4:])[0]
                n_r_ecg_p_08 = \
                    st.selectbox(label="Paroxysms of supraventricular tachycardia on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'),
                                 format_func=lambda e: e[4:])[0]
                n_r_ecg_p_09 = st.selectbox(label="Paroxysms of ventricular tachycardia on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'),
                                            format_func=lambda e: e[4:])[0]
                n_r_ecg_p_10 = \
                    st.selectbox(label="Ventricular fibrillation on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'),
                                 format_func=lambda e: e[4:])[0]
                n_p_ecg_p_01 = \
                    st.selectbox(label="Sinoatrial block on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                n_p_ecg_p_03 = \
                    st.selectbox(label="First-degree AV block on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                n_p_ecg_p_04 = \
                    st.selectbox(label="Type 1 Second-degree AV block (Mobitz I/Wenckebach) on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'),
                                 format_func=lambda e: e[4:])[0]
                n_p_ecg_p_05 = \
                    st.selectbox(label="Type 2 Second-degree AV block (Mobitz II/Hay) on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'),
                                 format_func=lambda e: e[4:])[0]

            with col3:
                n_p_ecg_p_06 = \
                    st.selectbox(label="Third-degree AV block on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                n_p_ecg_p_07 = \
                    st.selectbox(label="LBBB (anterior branch) on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                n_p_ecg_p_08 = \
                    st.selectbox(label="LBBB (posterior branch) on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                n_p_ecg_p_09 = \
                    st.selectbox(label="Incomplete LBBB on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                n_p_ecg_p_10 = \
                    st.selectbox(label="Complete LBBB on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                n_p_ecg_p_11 = \
                    st.selectbox(label="Incomplete RBBB on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                n_p_ecg_p_12 = \
                    st.selectbox(label="Complete RBBB on ECG at the time of admission to hospital", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                fibr_ter_01 = st.selectbox(label="Fibrinolytic therapy by Сеliasum 750k IU", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                fibr_ter_02 = st.selectbox(label="Fibrinolytic therapy by Сеliasum 1m IU", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                fibr_ter_03 = st.selectbox(label="Fibrinolytic therapy by Сеliasum 3m IU", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                fibr_ter_05 = st.selectbox(label="Fibrinolytic therapy by Streptase", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                fibr_ter_06 = st.selectbox(label="Fibrinolytic therapy by Сеliasum 500k IU", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                fibr_ter_07 = st.selectbox(label="Fibrinolytic therapy by Сеliasum 250k IU", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                fibr_ter_08 = st.selectbox(label="Fibrinolytic therapy by Streptodecase 1.5m IU", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                GIPO_K = st.selectbox(label="Hypokalemia ( < 4 mmol/L)", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                GIPER_NA = st.selectbox(label="Increase of sodium in serum (more than 150 mmol/L)", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                TIME_B_S = st.selectbox(label="Time elapsed from the beginning of the attack of CHD to the hospital", options=(
                    '1 – less than 2 hours', '2 – 2-4 hours', '3 – 4-6 hours', '4 – 6-8 hours', '5 – 8-12 hours', '6 – 12-24 hours', '7 – more than 1 days',
                    '8 – more than 2 days', '9 – more than 3 days'), format_func=lambda e: e[4:])[0]
                R_AB_1_n = st.selectbox(label="Relapse of the pain in the first 24 hours of the hospital period",
                                        options=('0 – there is no relapse', '1 – only one', '2 – 2 times', '3 – 3 or more times'), format_func=lambda e: e[4:])[0]
                R_AB_2_n = st.selectbox(label="Relapse of the pain in the second day of the hospital period",
                                        options=('0 – there is no relapse', '1 – only one', '2 – 2 times', '3 – 3 or more times'), format_func=lambda e: e[4:])[0]
                R_AB_3_n = st.selectbox(label="Relapse of the pain in the third day of the hospital period",
                                        options=('0 – there is no relapse', '1 – only one', '2 – 2 times', '3 – 3 or more times'), format_func=lambda e: e[4:])[0]
                NA_KB = st.selectbox(label="Use of opioid drugs by the Emergency Cardiology Team", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                NOT_NA_KB = st.selectbox(label="Use of NSAIDs by the Emergency Cardiology Team", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                LID_KB = st.selectbox(label="Use of lidocaine by the Emergency Cardiology Team", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                NITR_S = st.selectbox(label="Use of liquid nitrates in the ICU", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                NA_R_1_n = st.selectbox(label="Use of opioid drugs in the ICU in the first 24 hours of the hospital period",
                                        options=('0 – no', '1 – once', '2 – twice', '3 – three times', '4 – four times'), format_func=lambda e: e[4:])[0]
                NA_R_2_n = st.selectbox(label="Use of opioid drugs in the ICU in the second day of the hospital period",
                                        options=('0 – no', '1 – once', '2 – twice', '3 – three times'), format_func=lambda e: e[4:])[0]
                NA_R_3_n = st.selectbox(label="Use of opioid drugs in the ICU in the third day of the hospital period", options=('0 – no', '1 – once', '2 – twice'),
                                        format_func=lambda e: e[4:])[0]
                NOT_NA_1_n = st.selectbox(label="Use of NSAIDs in the ICU in the first 24 hours of the hospital period",
                                          options=('0 – no', '1 – once', '2 – twice', '3 – three times', '4 – four or more times'), format_func=lambda e: e[4:])[0]
                NOT_NA_2_n = st.selectbox(label="Use of NSAIDs in the ICU in the second day of the hospital period",
                                          options=('0 – no', '1 – once', '2 – twice', '3 – three times'), format_func=lambda e: e[4:])[0]
                NOT_NA_3_n = st.selectbox(label="Use of NSAIDs in the ICU in the third day of the hospital period", options=('0 – no', '1 – once', '2 – twice'),
                                          format_func=lambda e: e[4:])[0]
                LID_S_n = st.selectbox(label="Use of lidocaine in the ICU", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                B_BLOK_S_n = st.selectbox(label="Use of beta-blockers in the ICU", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                ANT_CA_S_n = st.selectbox(label="Use of calcium channel blockers in the ICU", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                GEPAR_S_n = st.selectbox(label="Use of а anticoagulants (heparin) in the ICU", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                ASP_S_n = st.selectbox(label="Use of acetylsalicylic acid in the ICU", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                TIKL_S_n = st.selectbox(label="Use of Ticlid in the ICU", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
                TRENT_S_n = st.selectbox(label="Use of Trental in the ICU", options=('0 – no', '1 – yes'), format_func=lambda e: e[4:])[0]
            submitted = st.form_submit_button("Submit")
            if submitted:
                x_predict = np.array([AGE, SEX, INF_ANAM, STENOK_AN, FK_STENOK, IBS_POST, GB, SIM_GIPERT, DLIT_AG, ZSN_A, nr_11, nr_01, nr_02, nr_03, nr_04, nr_07, nr_08, np_01, np_04, np_05, np_07, np_08, np_09, np_10,
                                      endocr_01, endocr_02, endocr_03, zab_leg_01, zab_leg_02, zab_leg_03, zab_leg_04, zab_leg_06, S_AD_ORIT, D_AD_ORIT, O_L_POST, K_SH_POST, MP_TP_POST, SVT_POST, GT_POST, FIB_G_POST,
                                      ant_im, lat_im, inf_im, post_im, IM_PG_P, ritm_ecg_p_01, ritm_ecg_p_02, ritm_ecg_p_04, ritm_ecg_p_06, ritm_ecg_p_07, ritm_ecg_p_08, n_r_ecg_p_01, n_r_ecg_p_02, n_r_ecg_p_03,
                                      n_r_ecg_p_04, n_r_ecg_p_05, n_r_ecg_p_06, n_r_ecg_p_08, n_r_ecg_p_09, n_r_ecg_p_10, n_p_ecg_p_01, n_p_ecg_p_03, n_p_ecg_p_04, n_p_ecg_p_05, n_p_ecg_p_06, n_p_ecg_p_07, n_p_ecg_p_08,
                                      n_p_ecg_p_09, n_p_ecg_p_10, n_p_ecg_p_11, n_p_ecg_p_12, fibr_ter_01, fibr_ter_02, fibr_ter_03, fibr_ter_05, fibr_ter_06, fibr_ter_07, fibr_ter_08, GIPO_K, K_BLOOD, GIPER_NA,
                                      NA_BLOOD, ALT_BLOOD, AST_BLOOD, L_BLOOD, ROE, TIME_B_S, R_AB_1_n, R_AB_2_n, R_AB_3_n, NA_KB, NOT_NA_KB, LID_KB, NITR_S, NA_R_1_n, NA_R_2_n, NA_R_3_n, NOT_NA_1_n, NOT_NA_2_n,
                                      NOT_NA_3_n, LID_S_n, B_BLOK_S_n, ANT_CA_S_n, GEPAR_S_n, ASP_S_n, TIKL_S_n, TRENT_S_n])
                complication_strings = ["Atrial fibrillation", "Supraventricular tachycardia", "Ventricular tachycardia", "Ventricular fibrillation", "Third-degree AV block", "Pulmonary edema", "Myocardial rupture",
                                        "Dressler syndrome", "Chronic heart failure", "Relapse of the myocardial infarction", "Post-infarction angina", "Lethal outcome"]

                complications_predicted = []
                for p_index, prediction in enumerate(ml_model.predict([x_predict])[0]):
                    if prediction == 1:
                        complications_predicted.append(complication_strings[p_index])
                if not complications_predicted:
                    st.subheader("No complications are predicted.")
                else:
                    st.subheader("Complications predicted:")
                    for _c in complications_predicted:
                        st.write(_c)
