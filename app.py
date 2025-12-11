import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. 配置與模型路徑 ---
MODEL_PATH = './best_ai_detector_model' 
# 必須與 fine_tuning_script.py 儲存模型時的路徑一致！

# --- 2. 模型載入函數 (使用 Streamlit 快取) ---
@st.cache_resource
def load_transformer_model(path):
    """
    使用 st.cache_resource 裝飾器，確保模型只在第一次運行時載入一次，
    大幅減少應用重啟或重新整理時的延遲。
    """
    try:
        # 載入 Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path)
        
        # 載入模型
        # map_location='cpu' 確保在沒有 GPU 的 Streamlit Cloud 環境也能運行
        model = AutoModelForSequenceClassification.from_pretrained(path, map_location=torch.device('cpu'))
        
        # 將模型切換到評估模式
        model.eval() 
        
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ 載入 Transformer 模型失敗。請檢查模型路徑和檔案是否完整。錯誤: {e}")
        st.stop()

# 載入模型和 Tokenizer
tokenizer, model = load_transformer_model(MODEL_PATH)


# --- 3. 預測函數 ---
def predict_ai_vs_human(text, tokenizer, model):
    """
    接受文本輸入，返回 Human 和 AI 的機率。
    """
    # 對輸入文本進行 Tokenization
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    )
    
    # 進行預測
    with torch.no_grad():
        # 模型推理 (推理)
        outputs = model(**inputs)
    
    # 取得 Logits (模型的原始輸出)
    logits = outputs.logits
    
    # 使用 Softmax 將 Logits 轉換為機率
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
    
    # 假設標籤 [0, 1] 對應 [Human, AI]
    # 如果只有一個輸入，probabilities 是一個包含兩個元素的列表 [P(Human), P(AI)]
    human_prob = probabilities[0]
    ai_prob = probabilities[1]
    
    return human_prob, ai_prob


# --- 4. Streamlit UI 設置 ---
st.set_page_config(page_title="🤖 AI/Human 文章偵測器", layout="wide")

st.title("🤖 AI/Human 文章偵測器 (基於 Transformers)")
st.markdown("使用您微調的 Transformer 模型來判斷文本的來源 (AI 或 Human)。")

st.subheader("📝 文本輸入區")
user_input = st.text_area(
    "請輸入一段文本：",
    placeholder="例如：The development of general artificial intelligence requires significant computational resources.",
    height=200
)

# 偵測按鈕
if st.button("🚨 立即判斷", type="primary"):
    if user_input:
        # 執行預測
        human_prob, ai_prob = predict_ai_vs_human(user_input, tokenizer, model)
        
        # 格式化百分比
        human_perc = f"{human_prob * 100:.2f}%"
        ai_perc = f"{ai_prob * 100:.2f}%"
        
        st.subheader("💡 判斷結果")
        
        # 顯示主要判斷結果
        if ai_prob > human_prob:
            st.warning(f"**主要判斷:** 該文本**較可能**由 **AI** 生成。")
        else:
            st.success(f"**主要判斷:** 該文本**較可能**由 **Human** 生成。")
            
        st.divider()

        # 顯示 AI% / Human%
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🧑🏻 Human (%)", value=human_perc, delta_color="normal")
        with col2:
            st.metric(label="🤖 AI (%)", value=ai_perc, delta_color="normal")
            
        # 可視化 (可選)
        st.subheader("📊 機率分佈")
        
        # 準備用於 Streamlit 圖表的資料框
        chart_data = pd.DataFrame({
            '類別': ['Human (人類)', 'AI (人工智慧)'],
            '機率': [human_prob, ai_prob]
        })
        
        # 使用 Streamlit 內建圖表進行可視化
        st.bar_chart(chart_data, x='類別', y='機率', color="#FF4B4B") # 使用 Streamlit 的預設警示色
        
    else:
        st.error("請輸入文本以進行偵測。")

st.caption("基於 RoBERTa-base 或您選擇的 Transformer 模型微調結果。")
