import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import networkx as nx
import io
import joblib
import tempfile

# ==========================================
# 1. APP TITLE & DESCRIPTION
# ==========================================
st.set_page_config(page_title="Market Basket Analyzer", layout="wide")
st.title("🛒 Market Basket Analysis App")
st.markdown("""
This app allows you to upload transaction data (Excel), runs the **Apriori Algorithm**, 
and generates association rules using **Support, Confidence, and Lift**.
""")

# ==========================================
# 2. SIDEBAR - PARAMETERS
# ==========================================
st.sidebar.header("1. Adjust Parameters")
min_support = st.sidebar.slider("Minimum Support", 0.001, 0.5, 0.03, 0.001)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.3, 0.05)
min_lift = st.sidebar.number_input("Minimum Lift", min_value=1.0, value=1.0)

st.sidebar.header("2. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload an Excel file (.xlsx or .csv)", type=["xlsx", "csv"])

# ==========================================
# 3. MAIN LOGIC
# ==========================================
if uploaded_file is not None:
    st.write("---")
    st.write("### 🔍 Raw Data Preview")

    try:
        # LOAD DATA
        df_excel = (
            pd.read_csv(uploaded_file, header=None)
            if uploaded_file.name.endswith(".csv")
            else pd.read_excel(uploaded_file, header=None)
        )
        st.dataframe(df_excel.head())

        # DATA PREPROCESSING
        st.info("Preprocessing transactions...")
        transactions = []
        for _, row in df_excel.iterrows():
            items = row.dropna().astype(str).str.strip().tolist()
            if items:
                transactions.append(items)

        st.success(f"Processed {len(transactions)} transactions")

        # ONE-HOT ENCODING
        te = TransactionEncoder()
        encoded_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(encoded_array, columns=te.columns_)

        # APRIORI
        frequent_itemsets = apriori(
            df_encoded,
            min_support=min_support,
            use_colnames=True
        )

        if frequent_itemsets.empty:
            st.error("No frequent itemsets found. Try lowering Minimum Support.")
        else:
            # ASSOCIATION RULES
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_confidence
            )

            rules = rules[rules["lift"] >= min_lift]

            if rules.empty:
                st.warning("No rules found. Try lowering Confidence or Lift.")
            else:
                # CLEAN RESULTS
                final_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

                final_rules['antecedents'] = final_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                final_rules['consequents'] = final_rules['consequents'].apply(lambda x: ', '.join(list(x)))

                final_rules['support'] = final_rules['support'].round(4)
                final_rules['confidence'] = final_rules['confidence'].round(4)
                final_rules['lift'] = final_rules['lift'].round(4)

                # DISPLAY TABLE
                st.write("### 📊 Association Rules")
                st.dataframe(final_rules)

                # DOWNLOAD EXCEL
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    final_rules.to_excel(writer, index=False, sheet_name="Rules")

                st.download_button(
                    label="📥 Download Results as Excel",
                    data=buffer.getvalue(),
                    file_name="apriori_final_metrics.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # ==========================================
                # GRAPH VISUALIZATION
                # ==========================================
                st.write("### 🔗 Association Rules Graph")

                top_rules = rules.sort_values(by="confidence", ascending=False).head(15)

                G = nx.DiGraph()
                for _, row in top_rules.iterrows():
                    ant = list(row['antecedents'])[0]
                    con = list(row['consequents'])[0]
                    G.add_edge(ant, con, weight=row['confidence'])

                plt.figure(figsize=(12, 8))
                pos = nx.spring_layout(G, k=1)

                nx.draw(
                    G, pos,
                    with_labels=True,
                    node_size=2500,
                    node_color="lightgreen",
                    font_size=10,
                    font_weight="bold",
                    arrowsize=20,
                    edge_color="gray"
                )

                edge_labels = nx.get_edge_attributes(G, 'weight')
                formatted_edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_edge_labels)

                plt.title("Top 15 Association Rules")
                st.pyplot(plt)

                # ==========================================
                # SAVE & DOWNLOAD PKL
                # ==========================================
                model_objects = {
                    "transaction_encoder": te,
                    "frequent_itemsets": frequent_itemsets,
                    "rules": rules
                }

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
                joblib.dump(model_objects, tmp.name)

                with open(tmp.name, "rb") as f:
                    st.download_button(
                        label="📦 Download Apriori Model (.pkl)",
                        data=f,
                        file_name="apriori_model.pkl",
                        mime="application/octet-stream"
                    )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Please upload a dataset to begin analysis.")
