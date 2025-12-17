import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import io

# ==========================================
# 1. APP TITLE & DESCRIPTION
# ==========================================
st.set_page_config(page_title="Market Basket Analyzer", layout="wide")
st.title("🛒 Market Basket Analysis App")
st.markdown("""
This app allows you to upload transaction data (Excel), runs the **Apriori Algorithm**, 
and generates association rules (Support, Confidence, Lift).
""")

# ==========================================
# 2. SIDEBAR - PARAMETERS
# ==========================================
st.sidebar.header("1. Adjust Parameters")
min_support = st.sidebar.slider("Minimum Support", 0.001, 0.5, 0.03, 0.001)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.3, 0.05)
min_lift = st.sidebar.number_input("Minimum Lift", min_value=1.0, value=1.0)

st.sidebar.header("2. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

# ==========================================
# 3. MAIN LOGIC
# ==========================================
if uploaded_file is not None:
    st.write("---")
    st.write("### 🔍 Raw Data Preview")
    
    # LOAD DATASET (Matches Step 1 in your PDF)
    try:
        # Load Excel, header=None because data usually starts at row 1
        df_excel = pd.read_csv(uploaded_file, header=None) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, header=None)
        st.dataframe(df_excel.head(5))
        
        # DATA PREPROCESSING (Matches Step 2 in your PDF)
        st.info("Preprocessing data...")
        transactions = []
        for _, row in df_excel.iterrows():
            # Drop NaNs, convert to string, strip whitespace
            items = row.dropna().astype(str).str.strip().tolist()
            if items:
                transactions.append(items)
        
        st.success(f"Successfully processed {len(transactions)} transactions!")

        # ENCODING
        te = TransactionEncoder()
        encoded_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(encoded_array, columns=te.columns_)

        # RUN APRIORI (Matches Step 4 in your PDF)
        frequent_itemsets = apriori(
            df_encoded, 
            min_support=min_support, 
            use_colnames=True
        )

        if frequent_itemsets.empty:
            st.error("No itemsets found! Try lowering the Minimum Support.")
        else:
            # GENERATE ASSOCIATION RULES (Matches Step 5 in your PDF)
            rules = association_rules(
                frequent_itemsets, 
                metric="confidence", 
                min_threshold=min_confidence
            )

            # Filter by Lift
            rules = rules[rules['lift'] >= min_lift]

            if rules.empty:
                st.warning("No rules found. Try lowering Confidence or Lift.")
            else:
                # CLEANUP FOR DISPLAY (Matches Step 6 in your PDF)
                # Keep only required metrics
                final_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                
                # Format the sets into readable strings (removing 'frozenset')
                final_rules['antecedents'] = final_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                final_rules['consequents'] = final_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                
                # Round numbers for better display
                final_rules['support'] = final_rules['support'].round(4)
                final_rules['confidence'] = final_rules['confidence'].round(4)
                final_rules['lift'] = final_rules['lift'].round(4)

                # DISPLAY RESULTS
                st.write("### 📊 Results: Association Rules")
                st.dataframe(final_rules)

                # SAVE TO EXCEL (Matches Step 7 in your PDF)
                # We use a memory buffer because we can't save to the user's hard drive directly from the web
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    final_rules.to_excel(writer, index=False, sheet_name='Rules')
                
                # DOWNLOAD BUTTON
                st.download_button(
                    label="📥 Download Results as Excel",
                    data=buffer.getvalue(),
                    file_name="apriori_final_metrics.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload an Excel file in the sidebar to begin.")
