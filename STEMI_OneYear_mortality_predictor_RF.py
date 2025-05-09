#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load(r'D:\PycharmProjects\AMIP\STEMI2019OneYearWithoutCRP\RF.pkl')

print(model.feature_names_in_)

# 定义特征名称
feature_names = ['DBP','CREA', 'NTproBNP', 'LVEF', 'CS']

# Streamlit 用户界面
st.title("STEMI One-Year Mortality Predictor")

# 用户输入
CS = st.selectbox("Cardiogenic Shock (0=No, 1=Yes):", options=[0, 1],
                  format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
CREA = st.number_input("creatinine,umol/L:", min_value=1, max_value=2000, value=72)
NTproBNP = st.number_input("NT-proBNP,pg/mL:", min_value=0.0, max_value=35000.0, value=400.0)
DBP = st.number_input("diastolic blood pressure,mmHg:", min_value=0, max_value=200, value=70)
LVEF = st.number_input("LVEF,%:", min_value=5, max_value=80, value=50)

# 处理输入并进行预测
feature_values = [DBP, CREA, NTproBNP, LVEF, CS]

# 将 feature_values 转换为 Pandas DataFrame
features = pd.DataFrame([feature_values], columns=feature_names)


if st.button("Predict"):
    # 预测概率
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[1] * 100  # 高风险概率

    # 风险分级
    if predicted_proba[1] >= 0.5:
        grade = "High Risk"
    else:
        grade = "Low Risk"

    # 将结果存储到 predata 中
    new_row = {
        'CS': CS,
        'CREA': CREA,
        'NTproBNP': NTproBNP,
        'LVEF': LVEF,
        'DBP': DBP
    }
    predata = pd.concat([features, pd.DataFrame([new_row])], ignore_index=True)

    # 显示预测结果
    st.write(f"**Risk Stratification:** {grade}")
    st.write(f"**Predicted Probability of Risk:** {probability:.1f}%")

    # SHAP 值的计算和可视化
    explainer = shap.TreeExplainer(model)  # 使用TreeExplainer来解释随机森林模型
    shap_values = explainer.shap_values(features.iloc[0, :])

    shap_plot = shap.plots.force(
        explainer.expected_value[1],
        shap_values[:, 1].flatten(),
        predata.iloc[0, :],
        show=False,
        matplotlib=True)

    for text in plt.gca().texts:  # 遍历当前坐标轴的所有文本对象
        if "=" in text.get_text():
            text.set_rotation(-15)  # 设置旋转角度，修改为你希望的角度
            text.set_va('top')
        text.set_bbox(dict(facecolor='none', alpha=0.5, edgecolor='none'))

    plt.tight_layout()
    st.pyplot(plt.gcf())