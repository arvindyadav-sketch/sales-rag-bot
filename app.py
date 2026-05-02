import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Sales RAG Bot",
    page_icon="🤖",
    layout="wide"
)

st.sidebar.title("⚙️ Settings")
API_KEY = st.sidebar.text_input(
    "🔑 OpenRouter API Key:",
    type="password"
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Excel file upload karo",
    type=['xlsx', 'xls']
)

st.title("🤖 My Sales RAG Bot")
st.markdown("**Apna sales data upload karo aur sawaal poochho!**")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def ask_ai(prompt, api_key):
    if not api_key:
        return "⚠️ API Key daalo sidebar mein!"
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
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
        return f"API Error: {res}"
    except Exception as e:
        return f"Error: {str(e)}"

def search_data(filter_word, documents, df, top_k=5):
    fw = filter_word.strip().lower()
    results = []
    for d in documents:
        if fw in d.lower():
            results.append(d)
    if not results:
        for _, row in df.iterrows():
            text = (
                f"Partner: {row['Partner Name']} | "
                f"SKU: {row['SKU']} | "
                f"Qty: {row['Qty']} | "
                f"Location: {row['Locations']} | "
                f"Status: {row['PO/Delivery Status']} | "
                f"Current: {row['Current Status']} | "
                f"Courier: {row['Courier']}"
            )
            if fw in text.lower():
                results.append(text)
            if len(results) >= top_k:
                break
    return results[:top_k]

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, header=1)
    df = df.fillna('N/A')
    st.sidebar.success(f"✅ {len(df):,} rows loaded!")

    @st.cache_data
    def prepare_docs(fname):
        def row_to_text(row):
            return (
                f"Partner: {row['Partner Name']} | "
                f"SKU: {row['SKU']} | "
                f"Qty: {row['Qty']} | "
                f"Location: {row['Locations']} | "
                f"Status: {row['PO/Delivery Status']} | "
                f"Current: {row['Current Status']} | "
                f"Pending Days: {row['Pending Days']} | "
                f"Courier: {row['Courier']} | "
                f"OMS Order: {row['OMS Order']}"
            )
        docs = df.head(500).apply(row_to_text, axis=1).tolist()
        embs = model.encode(docs, show_progress_bar=False)
        return docs, embs

    with st.spinner("🔄 Data prepare ho raha hai..."):
        documents, db_embeddings = prepare_docs(uploaded_file.name)

    tab1, tab2, tab3 = st.tabs([
        "💬 Chatbot",
        "📊 Dashboard",
        "💰 Payment Pending"
    ])

    with tab1:
        st.markdown("### 💬 Sales Chatbot")
        sawaal = st.text_input(
            "❓ Apna sawaal likho:",
            placeholder="Jaise: LINK TELECOM ke orders ka status?"
        )
        filter_word = st.text_input(
            "🔍 Filter word (optional):",
            placeholder="Jaise: InTransit, Delhi, Evolution Inc"
        )
        if st.button("🚀 Poochho!", type="primary"):
            if sawaal:
                with st.spinner("🤔 Dhundh raha hoon..."):
                    if filter_word.strip():
                        relevant = search_data(
                            filter_word, documents, df
                        )
                        st.info(
                            f"✅ '{filter_word}' → "
                            f"{len(relevant)} records mile"
                        )
                    else:
                        sawaal_emb = model.encode([sawaal])
                        scores = cosine_similarity(
                            sawaal_emb, db_embeddings
                        )[0]
                        top_idx = np.argsort(scores)[::-1][:5]
                        relevant = [documents[i] for i in top_idx]
                        st.info(f"✅ Top {len(relevant)} records mile")
                    if relevant:
                        prompt = (
                            f"Sales Records:\n"
                            f"{chr(10).join(relevant)}\n\n"
                            f"Sawaal: {sawaal}\n"
                            f"Hindi mein seedha jawab do. "
                            f"Numbers include karo."
                        )
                        jawab = ask_ai(prompt, API_KEY)
                        st.success("✅ AI ka Jawab:")
                        st.write(jawab)
                        with st.expander("🔍 Records dekho"):
                            for i, rec in enumerate(relevant):
                                st.text(f"{i+1}. {rec}")
                    else:
                        st.warning("⚠️ Koi record nahi mila!")

        st.markdown("---")
        st.markdown("#### ⚡ Quick Buttons:")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💰 Pending Summary"):
                p = df[df['PO/Delivery Status'] == 'Payment Pending']
                st.metric("Pending Orders", int(p['OMS Order'].nunique()))
                st.metric("Total Qty", f"{int(p['Qty'].sum()):,}")
                st.metric("Partners", int(p['Partner Name'].nunique()))
        with col2:
            if st.button("📦 Top Partners"):
                st.dataframe(df['Partner Name'].value_counts().head(5))
        with col3:
            if st.button("🚚 Courier Stats"):
                st.dataframe(df['Courier'].value_counts().head(5))

    with tab2:
        st.markdown("### 📊 Complete Sales Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_orders = df['OMS Order'].nunique()
            st.metric("📦 Total Orders", f"{total_orders:,}")
        with col2:
            d = df[df['PO/Delivery Status'] == 'Delivered']['OMS Order'].nunique()
            st.metric("✅ Delivered", f"{d:,}")
        with col3:
            p = df[df['PO/Delivery Status'] == 'Payment Pending']['OMS Order'].nunique()
            st.metric("🔴 Payment Pending", f"{p:,}")
        with col4:
            st.metric("🤝 Partners", f"{df['Partner Name'].nunique():,}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📦 Top 10 Partners (Unique Orders)")
            top_p = df.groupby('Partner Name')['OMS Order'].nunique().sort_values(ascending=False).head(10)
            st.bar_chart(top_p)
        with col2:
            st.markdown("#### 📊 Status Wise (Unique Orders)")
            status_chart = df.groupby('PO/Delivery Status')['OMS Order'].nunique().sort_values(ascending=False)
            st.bar_chart(status_chart)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📍 Top Locations (Unique Orders)")
            locs = df.groupby('Locations')['OMS Order'].nunique().sort_values(ascending=False).head(10)
            st.bar_chart(locs)
        with col2:
            st.markdown("#### 🚚 Courier Performance (Unique Orders)")
            cour = df.groupby('Courier')['OMS Order'].nunique().reset_index()
cour.columns = ['Courier', 'Orders']
cour = cour.sort_values('Orders', ascending=False).head(6).set_index('Courier')
            st.bar_chart(cour)

        st.markdown("#### 📋 Complete Status Breakdown (Unique Orders)")
        status_df = df.groupby('PO/Delivery Status').agg(
            Unique_Orders=('OMS Order', 'nunique'),
            Total_SKUs=('SKU', 'count'),
            Total_Qty=('Qty', 'sum')
        ).reset_index().sort_values('Unique_Orders', ascending=False).reset_index(drop=True)
        status_df.index = status_df.index + 1
        st.dataframe(status_df, use_container_width=True)

    with tab3:
        st.markdown("### 💰 Payment Pending Report")
        pending_df = df[df['PO/Delivery Status'] == 'Payment Pending'].copy()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Orders", int(pending_df['OMS Order'].nunique()))
        with col2:
            st.metric("Total Qty", f"{int(pending_df['Qty'].sum()):,}")
        with col3:
            st.metric("Partners", int(pending_df['Partner Name'].nunique()))

        st.markdown("---")
        rows = []
        for partner, grp in pending_df.groupby('Partner Name'):
            rows.append({
                'Partner Name': partner,
                'Unique Orders': int(grp['OMS Order'].nunique()),
                'Total SKUs': int(len(grp)),
                'Total Qty': int(grp['Qty'].sum())
            })
        result = pd.DataFrame(rows).sort_values(
            'Total Qty', ascending=False
        ).reset_index(drop=True)
        result.index = result.index + 1
        st.markdown("#### 📋 Partner Wise Breakdown")
        st.dataframe(result, use_container_width=True)

        csv = result.to_csv(index=False)
        st.download_button(
            "📥 Download Report (CSV)",
            csv,
            "payment_pending.csv",
            "text/csv",
            type="primary"
        )

        st.markdown("---")
        st.markdown("#### 🔍 Partner Detail")
        selected = st.selectbox(
            "Partner choose karo:",
            options=sorted(pending_df['Partner Name'].unique().tolist())
        )
        if selected:
            detail_rows = []
            pdata = pending_df[pending_df['Partner Name'] == selected]
            for oms, grp in pdata.groupby('OMS Order'):
                detail_rows.append({
                    'OMS Order': oms,
                    'Total Qty': int(grp['Qty'].sum()),
                    'SKU Count': int(len(grp)),
                    'Location': grp['Locations'].iloc[0]
                })
            detail_df = pd.DataFrame(detail_rows)
            st.dataframe(detail_df, use_container_width=True)
            st.info(
                f"Total: {len(detail_rows)} orders | "
                f"{int(pdata['Qty'].sum()):,} qty"
            )

else:
    st.info("👈 Sidebar mein Excel file upload karo!")
    st.markdown("""
### 🚀 Yeh Bot Kar Sakta Hai:
| Feature | Description |
|---------|-------------|
| 💬 Chatbot | Koi bhi sawaal poochho Hindi mein |
| 📊 Dashboard | Charts aur graphs se poora overview |
| 💰 Payment Pending | Partner wise detailed report |
""")
