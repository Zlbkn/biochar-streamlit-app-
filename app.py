# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import os

# ===============================
# 1️⃣ 页面标题
# ===============================
st.title("生物炭产率及产物预测")
st.write("输入原料性质和工艺参数，预测生物炭产率、生物油产率、合成气产率及生物炭关键成分")

# ===============================
# 2️⃣ 载入模型（使用相对路径）
# ===============================
model_path = os.path.join(os.path.dirname(__file__), "woods-model.pkl")
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("模型文件未找到，请确认 'woods-model.pkl' 与 app.py 在同一目录下")
    st.stop()

# ===============================
# 3️⃣ 用户输入
# ===============================
st.header("输入原料和工艺参数")

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

reaction_time = st.number_input('反应时间（min）', value=60)
heating_rate = st.number_input('加热速率(°C/min)', value=10)
feed_rate = st.number_input('进料速率(g/min)', value=5)
n2_flow = st.number_input('氮气流量（L/min）', value=100)
temperature = st.number_input('热解温度(°C)', value=500)

# ===============================
# 4️⃣ 构建输入 DataFrame（列名与 feature_cols 对齐）
# ===============================
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

# ===============================
# 5️⃣ 特征工程
# ===============================
input_df['C_H'] = input_df['C/wt%-dry'] / input_df['H/wt%-dry']
input_df['C_O'] = input_df['C/wt%-dry'] / input_df['O/wt%-dry']
input_df['H_C'] = input_df['H/wt%-dry'] / input_df['C/wt%-dry']
input_df['VM_FC'] = input_df['挥发分/wt%-dry'] / input_df['固定碳/wt%-dry']
input_df['FC_Ash'] = input_df['固定碳/wt%-dry'] / input_df['灰分/wt%-dry']

if 'HHVMilne MJ/kg-dry' in input_df.columns:
    input_df['EnergyDensity'] = input_df['HHVMilne MJ/kg-dry'] / (
        input_df['C/wt%-dry'] + input_df['H/wt%-dry'] + input_df['O/wt%-dry'] +
        input_df['N/wt%-dry'] + input_df['S/wt%-dry']
    )

input_df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
input_df.fillna(input_df.median(), inplace=True)

# ===============================
# 6️⃣ 特征列顺序
# ===============================
feature_cols = [
    '水分/wt%-ar','灰分/wt%-dry','挥发分/wt%-dry','固定碳/wt%-dry',
    'C/wt%-dry','H/wt%-dry','O/wt%-dry','N/wt%-dry','S/wt%-dry',
    '纤维素/wt%-dry','半纤维素/wt%-dry','褐色素/wt%-dry',
    'HHVMilne MJ/kg-dry','反应时间（min）','加热速率(°C/min)',
    '进料速率(g/min)','氮气流量（L/min)','热解温度(°C)',
    'C_H','C_O','H_C','VM_FC','FC_Ash'
]

if 'EnergyDensity' in input_df.columns:
    feature_cols.append('EnergyDensity')

# ===============================
# 7️⃣ 检查缺失列，防止 KeyError
# ===============================
missing_cols = [c for c in feature_cols if c not in input_df.columns]
if missing_cols:
    st.error(f"以下特征列在输入中不存在: {missing_cols}")
    st.stop()

# ===============================
# 8️⃣ Min-Max 归一化
# ===============================
scaler = MinMaxScaler()
input_scaled = pd.DataFrame(scaler.fit_transform(input_df[feature_cols]), columns=feature_cols)

# ===============================
# 9️⃣ 模型预测
# ===============================
pred = model.predict(input_scaled)

Y_cols = [
    '生物炭产率(%)','生物油产率(%)','合成气产率(%)',
    '生物碳固定碳/wt%-dry','生物碳灰分/wt%-dry',
    '生物碳C/wt%-dry','生物炭HHV/MJ/kg-dry'
]
pred_df = pd.DataFrame(pred, columns=Y_cols)

# ===============================
# 10️⃣ 显示结果
# ===============================
st.header("预测结果")
st.dataframe(pred_df)