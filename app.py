import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

# Page config
st.set_page_config(
    page_title="Sales RAG Bot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 My Sales RAG Bot")
st.markdown("**Apna sales data upload karo aur sawaal poochho!**")

# API Key
API_KEY = st.sidebar.text_input(
    "🔑 OpenRouter API Key:", 
    type="password",
    help="sk-or-v1-... format mein"
)

# Model load
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# File upload
st.sidebar.markdown("### 📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Excel file upload karo",
    type=['xlsx', 'xls']
)

if uploaded_file:
    # Data load
    df = pd.read_excel(uploaded_file, header=1)
    df = df.fillna('N/A')
    
    st.sidebar.success(f"✅ {len(df)} rows loaded!")
    
    # Tabs banao
    tab1, tab2, tab3 = st.tabs([
        "💬 Chatbot", 
        "📊 Dashboard", 
        "💰 Payment Pending"
    ])
    
    # Documents banao
    @st.cache_data
    def prepare_documents(file_name):
        def row_to_text(row):
            return (f"Partner: {row['Partner Name']} | "
                   f"SKU: {row['SKU']} | "
                   f"Qty: {row['Qty']} | "
                   f"Location: {row['Locations']} | "
                   f"Status: {row['PO/Delivery Status']} | "
                   f"Current: {row['Current Status']} | "
                   f"Pending Days: {row['Pending Days']} | "
                   f"Courier: {row['Courier']}")
        
        docs = df.head(500).apply(row_to_text, axis=1).tolist()
        embeddings = model.encode(docs, show_progress_bar=False)
        return docs, embeddings
    
    with st.spinner("🔄 Embeddings ban rahi hain..."):
        documents, db_embeddings = prepare_documents(
            uploaded_file.name
        )
    
    # AI Function
    def ask_ai(prompt):
        if not API_KEY:
            return "⚠️ API Key daalo sidebar mein!"
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "google/gemma-3-4b-it:free",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            res = response.json()
            if "choices" in res:
                return res["choices"][0]["message"]["content"]
            return f"Error: {res}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # TAB 1: CHATBOT
    with tab1:
        st.markdown("### 💬 Sales Chatbot")
        
        col1, col2 = st.columns([3,1])
        with col1:
            sawaal = st.text_input(
                "❓ Apna sawaal likho:",
                placeholder="Jaise: LINK TELECOM ke pending orders?"
            )
        with col2:
            filter_word = st.text_input(
                "🔍 Filter (optional):",
                placeholder="Partner ya status"
            )
        
        if st.button("🚀 Poochho!", type="primary"):
            if sawaal:
                with st.spinner("🤔 Soch raha hoon..."):
                    # Relevant data dhundo
                    if filter_word:
                        filtered = [d for d in documents 
                                  if filter_word.lower() in d.lower()]
                        relevant = filtered[:5]
                        st.info(f"✅ '{filter_word}' → {len(filtered)} records mile")
                    else:
                        sawaal_emb = model.encode([sawaal])
                        scores = cosine_similarity(
                            sawaal_emb, db_embeddings
                        )[0]
                        top_idx = np.argsort(scores)[::-1][:5]
                        relevant = [documents[i] for i in top_idx]
                    
                    if relevant:
                        # AI se jawab lo
                        prompt = f"""
Sales Records:
{chr(10).join(relevant)}

Sawaal: {sawaal}
Concise Hindi mein jawab do.
"""
                        jawab = ask_ai(prompt)
                        
                        st.success("✅ AI ka Jawab:")
                        st.write(jawab)
                        
                        with st.expander("🔍 Relevant Records dekho"):
                            for i, rec in enumerate(relevant[:3]):
                                st.text(f"{i+1}. {rec}")
                    else:
                        st.warning("Koi record nahi mila!")
    
    # TAB 2: DASHBOARD
    with tab2:
        st.markdown("### 📊 Sales Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Orders", f"{len(df):,}")
        with col2:
            delivered = len(df[df['PO/Delivery Status']=='Delivered'])
            st.metric("Delivered", f"{delivered:,}")
        with col3:
            pending = len(df[df['PO/Delivery Status']=='Payment Pending'])
            st.metric("Payment Pending", f"{pending:,}", delta="-urgent")
        with col4:
            partners = df['Partner Name'].nunique()
            st.metric("Total Partners", f"{partners:,}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📦 Top 10 Partners")
            top_partners = df['Partner Name'].value_counts().head(10)
            st.bar_chart(top_partners)
        
        with col2:
            st.markdown("#### 🚚 Courier Performance")
            courier = df['Courier'].value_counts().head(6)
            st.bar_chart(courier)
        
        st.markdown("#### 📍 Location Wise Orders")
        location = df['Locations'].value_counts().head(10)
        st.bar_chart(location)
    
    # TAB 3: PAYMENT PENDING
    with tab3:
        st.markdown("### 💰 Payment Pending Report")
        
        pending_df = df[
            df['PO/Delivery Status'] == 'Payment Pending'
        ]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Pending Orders", 
                pending_df['OMS Order'].nunique()
            )
        with col2:
            st.metric(
                "Total Pending Qty", 
                f"{int(pending_df['Qty'].sum()):,}"
            )
        with col3:
            st.metric(
                "Partners Affected", 
                pending_df['Partner Name'].nunique()
            )
        
        # Partner wise table
        result = pending_df.groupby('Partner Name').agg(
            Unique_Orders=('OMS Order', 'nunique'),
            Total_SKUs=('SKU', 'count'),
            Total_Qty=('Qty', 'sum')
        ).reset_index().sort_values(
            'Total_Qty', ascending=False
        )
        
        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = result.to_csv(index=False)
        st.download_button(
            "📥 Report Download Karo",
            csv,
            "payment_pending.csv",
            "text/csv"
        )

else:
    st.info("👈 Sidebar mein Excel file upload karo!")
    st.markdown("""
    ### Yeh Bot Kar Sakta Hai:
    - 💬 **Chatbot**: Koi bhi sawaal poochho
    - 📊 **Dashboard**: Complete sales overview  
    - 💰 **Payment Pending**: Partner wise report
    """)
