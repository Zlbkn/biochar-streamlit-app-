# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import os
import plotly.express as px
import time

# ===============================
# 页面配置
# ===============================
st.set_page_config(page_title="🔥 生物炭预测仪表盘 🔥", layout="wide")

st.markdown("<h1 style='text-align:center;color:green;'>生物炭产率及产物预测仪表盘</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>输入原料和工艺参数，点击 <b>开始预测</b> 查看结果</p >", unsafe_allow_html=True)
st.markdown("---")

# ===============================
# 载入模型
# ===============================
model_path = os.path.join(os.path.dirname(__file__), "woods-model.pkl")
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("模型文件未找到，请确认 'woods-model.pkl' 与 app.py 在同一目录下")
    st.stop()

# ===============================
# 输入参数
# ===============================
st.header("⚙️ 输入参数")

col1, col2 = st.columns(2)

with col1:
    with st.expander("🌿 原料组成参数", expanded=True):
        water = st.number_input('水分/wt%-ar', value=10.0)
        ash = st.number_input('灰分/wt%-dry', value=5.0)
        vm = st.number_input('挥发分/wt%-dry', value=70.0)
        fc = st.number_input('固定碳/wt%-dry', value=25.0)
        C = st.number_input('C/wt%-dry', value=50.0)
        H = st.number_input('H/wt%-dry', value=6.0)
        O = st.number_input('O/wt%-dry', value=40.0)
        N = st.number_input('N/wt%-dry', value=0.5)
        S = st.number_input('S/wt%-dry', value=0.1)
        HHV = st.number_input('HHVMilne MJ/kg-dry', value=18.0)
        cellulose = st.number_input('纤维素/wt%-dry', value=40.0)
        hemicellulose = st.number_input('半纤维素/wt%-dry', value=25.0)
        lignin = st.number_input('褐色素/wt%-dry', value=20.0)

with col2:
    with st.expander("🔥 工艺参数", expanded=True):
        reaction_time = st.number_input('反应时间（min）', value=60)
        heating_rate = st.number_input('加热速率(°C/min)', value=10)
        feed_rate = st.number_input('进料速率(g/min)', value=5)
        n2_flow = st.number_input('氮气流量（L/min）', value=100)
        temperature = st.number_input('热解温度(°C)', value=500)

st.markdown("---")

# ===============================
# 预测按钮 + 动态进度条
# ===============================
if st.button("🚀 开始预测"):
    progress_text = "模型计算中，请稍候..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(0, 101, 10):
        time.sleep(0.1)
        my_bar.progress(percent_complete, text=progress_text)

    # 构建 DataFrame
    input_df = pd.DataFrame([[
        water, ash, vm, fc, C, H, O, N, S, HHV,
        cellulose, hemicellulose, lignin,
        reaction_time, heating_rate, feed_rate, n2_flow, temperature
    ]], columns=[
        '水分/wt%-ar','灰分/wt%-dry','挥发分/wt%-dry','固定碳/wt%-dry',
        'C/wt%-dry','H/wt%-dry','O/wt%-dry','N/wt%-dry','S/wt%-dry','HHVMilne MJ/kg-dry',
        '纤维素/wt%-dry','半纤维素/wt%-dry','褐色素/wt%-dry',
        '反应时间（min）','加热速率(°C/min)','进料速率(g/min)','氮气流量（L/min）','热解温度(°C)'
    ])

    # 特征工程
    input_df['C_H'] = input_df['C/wt%-dry'] / input_df['H/wt%-dry']
    input_df['C_O'] = input_df['C/wt%-dry'] / input_df['O/wt%-dry']
    input_df['H_C'] = input_df['H/wt%-dry'] / input_df['C/wt%-dry']
    input_df['VM_FC'] = input_df['挥发分/wt%-dry'] / input_df['固定碳/wt%-dry']
    input_df['FC_Ash'] = input_df['固定碳/wt%-dry'] / input_df['灰分/wt%-dry']
    input_df['EnergyDensity'] = input_df['HHVMilne MJ/kg-dry'] / (
            input_df['C/wt%-dry'] + input_df['H/wt%-dry'] + input_df['O/wt%-dry'] +
            input_df['N/wt%-dry'] + input_df['S/wt%-dry']
        )

    input_df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    input_df.fillna(input_df.median(), inplace=True)

    feature_cols = [
        '水分/wt%-ar', '灰分/wt%-dry', '挥发分/wt%-dry', '固定碳/wt%-dry',
        'C/wt%-dry', 'H/wt%-dry', 'O/wt%-dry', 'N/wt%-dry', 'S/wt%-dry',
        '纤维素/wt%-dry', '半纤维素/wt%-dry', '褐色素/wt%-dry',
        'HHVMilne MJ/kg-dry', '反应时间（min）', '加热速率(°C/min)',
        '进料速率(g/min)', '氮气流量（L/min）', '热解温度(°C)',
        'C_H', 'C_O', 'H_C', 'VM_FC', 'FC_Ash','EnergyDensity'
    ]

    # MinMax归一化
    scaler = MinMaxScaler()
    input_scaled = pd.DataFrame(scaler.fit_transform(input_df[feature_cols]), columns=feature_cols)

    # 模型预测
    pred = model.predict(input_scaled)
    Y_cols = [
        '生物炭产率(%)','生物油产率(%)','合成气产率(%)',
        '生物碳固定碳/wt%-dry','生物碳灰分/wt%-dry',
        '生物碳C/wt%-dry','生物炭HHV/MJ/kg-dry'
    ]
    pred_df = pd.DataFrame(pred, columns=Y_cols)

    # ===============================
    # 预测结果展示 - 仪表盘风格
    # ===============================
    st.header("📊 预测结果仪表盘")

    # 1️⃣ 关键产率指标 - 三列 Metric
    st.subheader("关键产率指标")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("生物炭产率(%)", f"{pred_df['生物炭产率(%)'][0]:.2f}")
    col_b.metric("生物油产率(%)", f"{pred_df['生物油产率(%)'][0]:.2f}")
    col_c.metric("合成气产率(%)", f"{pred_df['合成气产率(%)'][0]:.2f}")

    # 2️⃣ 产物比例 - 饼图
    st.subheader("产物比例可视化")
    prod_df = pd.DataFrame({
        '产物': ['生物炭','生物油','合成气'],
        '百分比': [
            pred_df['生物炭产率(%)'][0],
            pred_df['生物油产率(%)'][0],
            pred_df['合成气产率(%)'][0]
        ]
    })
    fig = px.pie(prod_df, names='产物', values='百分比', color='产物',
                 color_discrete_map={'生物炭':'#2ca02c','生物油':'#ff7f0e','合成气':'#1f77b4'},
                 title="产物比例")
    st.plotly_chart(fig, use_container_width=True)

    # 3️⃣ 生物炭关键成分 - 动态柱状图
    st.subheader("生物炭关键成分")
    comp_df = pd.DataFrame({
        '成分': ['固定碳','灰分','C','HHV'],
        '值': [
            pred_df['生物碳固定碳/wt%-dry'][0],
            pred_df['生物碳灰分/wt%-dry'][0],
            pred_df['生物碳C/wt%-dry'][0],
            pred_df['生物炭HHV/MJ/kg-dry'][0]
        ]
    })
    fig2 = px.bar(comp_df, x='成分', y='值', color='成分',
                  color_discrete_map={'固定碳':'#2ca02c','灰分':'#ff7f0e','C':'#1f77b4','HHV':'#d62728'},
                  title="生物炭关键成分柱状图", text='值')
    st.plotly_chart(fig2, use_container_width=True)
