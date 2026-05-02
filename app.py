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

if uploaded_file:
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

    # TAB 1: CHATBOT
    with tab1:
        st.markdown("### 💬 Sales Chatbot")

        sawaal = st.text_input(
            "❓ Apna sawaal likho:",
            placeholder="Jaise: LINK TELECOM ke orders ka status?"
        )
        filter_word = st.text_input(
            "🔍 Filter word (optional):",
            placeholder="Jaise: LINK TELECOM, Delhi, Delivered"
        )

        if st.button("🚀 Poochho!", type="primary"):
            if sawaal:
                with st.spinner("🤔 Answer dhundh raha hoon..."):
                    if filter_word.strip():
    filtered = [
        d for d in documents
        if filter_word.strip().lower() in d.lower()
    ]
    if not filtered:
        # Direct DataFrame se nikalo
        mask = df.apply(
            lambda row: row.astype(str).str.contains(
                filter_word.strip(), 
                case=False
            ).any(), 
            axis=1
        )
        filtered_df = df[mask].head(5)
        relevant = filtered_df.apply(
            lambda row: (
                f"Partner: {row['Partner Name']} | "
                f"SKU: {row['SKU']} | "
                f"Qty: {row['Qty']} | "
                f"Status: {row['PO/Delivery Status']} | "
                f"Location: {row['Locations']}"
            ), axis=1
        ).tolist()
    else:
        relevant = filtered[:5]
                        relevant = filtered[:5]
                        st.info(
                            f"✅ '{filter_word}' → "
                            f"{len(filtered)} records mile"
                        )
                    else:
                        sawaal_emb = model.encode([sawaal])
                        scores = cosine_similarity(
                            sawaal_emb, db_embeddings
                        )[0]
                        top_idx = np.argsort(scores)[::-1][:5]
                        relevant = [documents[i] for i in top_idx]
                        st.info(
                            f"✅ Top {len(relevant)} records mile"
                        )

                    if relevant:
                        prompt = f"""
Sales Records:
{chr(10).join(relevant)}

Sawaal: {sawaal}
Hindi mein seedha aur accurate jawab do.
Numbers aur facts include karo.
"""
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
                pending = df[
                    df['PO/Delivery Status'] == 'Payment Pending'
                ]
                st.metric("Pending Orders",
                    int(pending['OMS Order'].nunique()))
                st.metric("Total Qty",
                    f"{int(pending['Qty'].sum()):,}")
                st.metric("Partners",
                    int(pending['Partner Name'].nunique()))

        with col2:
            if st.button("📦 Top Partners"):
                top = df['Partner Name'].value_counts().head(5)
                st.dataframe(top)

        with col3:
            if st.button("🚚 Courier Stats"):
                courier = df['Courier'].value_counts().head(5)
                st.dataframe(courier)

    # TAB 2: DASHBOARD
    with tab2:
        st.markdown("### 📊 Complete Sales Dashboard")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📦 Total Rows", f"{len(df):,}")
        with col2:
            delivered = len(
                df[df['PO/Delivery Status'] == 'Delivered']
            )
            st.metric("✅ Delivered", f"{delivered:,}")
        with col3:
            pend = len(
                df[df['PO/Delivery Status'] == 'Payment Pending']
            )
            st.metric("🔴 Payment Pending", f"{pend:,}")
        with col4:
            st.metric(
                "🤝 Partners",
                f"{df['Partner Name'].nunique():,}"
            )

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📦 Top 10 Partners")
            top_p = df['Partner Name'].value_counts().head(10)
            st.bar_chart(top_p)

        with col2:
            st.markdown("#### 📊 Order Status")
            status = df['PO/Delivery Status'].value_counts()
            st.bar_chart(status)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📍 Top Locations")
            locs = df['Locations'].value_counts().head(10)
            st.bar_chart(locs)

        with col2:
            st.markdown("#### 🚚 Courier Performance")
            cour = df['Courier'].value_counts().head(6)
            st.bar_chart(cour)

        st.markdown("#### 📋 Status Breakdown")
        status_df = df.groupby('PO/Delivery Status').size(
        ).reset_index(name='Count').sort_values(
            'Count', ascending=False
        )
        st.dataframe(status_df, use_container_width=True)

    # TAB 3: PAYMENT PENDING
    with tab3:
        st.markdown("### 💰 Payment Pending Report")

        pending_df = df[
            df['PO/Delivery Status'] == 'Payment Pending'
        ].copy()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Orders",
                int(pending_df['OMS Order'].nunique())
            )
        with col2:
            st.metric(
                "Total Qty",
                f"{int(pending_df['Qty'].sum()):,}"
            )
        with col3:
            st.metric(
                "Partners",
                int(pending_df['Partner Name'].nunique())
            )

        st.markdown("---")

        # Partner wise - FIXED
        partner_group = []
        for partner, group in pending_df.groupby('Partner Name'):
            partner_group.append({
                'Partner Name': partner,
                'Unique Orders': group['OMS Order'].nunique(),
                'Total SKUs': len(group),
                'Total Qty': int(group['Qty'].sum())
            })

        result = pd.DataFrame(partner_group).sort_values(
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
        partners_list = sorted(
            pending_df['Partner Name'].unique().tolist()
        )
        selected = st.selectbox(
            "Partner choose karo:",
            options=partners_list
        )

        if selected:
            detail = []
            partner_data = pending_df[
                pending_df['Partner Name'] == selected
            ]
            for oms, grp in partner_data.groupby('OMS Order'):
                detail.append({
                    'OMS Order': oms,
                    'Total Qty': int(grp['Qty'].sum()),
                    'SKU Count': len(grp),
                    'Location': grp['Locations'].iloc[0]
                })

            detail_df = pd.DataFrame(detail)
            st.dataframe(detail_df, use_container_width=True)
            st.info(
                f"Total: {len(detail)} orders | "
                f"{int(partner_data['Qty'].sum()):,} qty"
            )

else:
    st.info(
        "👈 Sidebar mein Excel file upload karo "
        "shuru karne ke liye!"
    )
    st.markdown("""
### 🚀 Yeh Bot Kar Sakta Hai:
| Feature | Description |
|---------|-------------|
| 💬 Chatbot | Koi bhi sawaal poochho Hindi mein |
| 📊 Dashboard | Charts aur graphs se poora overview |
| 💰 Payment Pending | Partner wise detailed report |
""")
