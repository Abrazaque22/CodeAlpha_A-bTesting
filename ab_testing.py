import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_sample_size(baseline_conv_rate, mde, alpha=0.05, power=0.8):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p = baseline_conv_rate
    sample_size = ((z_alpha + z_beta)**2 * p * (1 - p) * 2) / (mde**2)
    return np.ceil(sample_size)

def proportions_ztest(count1, nobs1, count2, nobs2):
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    p_pooled = (count1 + count2) / (nobs1 + nobs2)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/nobs1 + 1/nobs2))
    z_score = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return z_score, p_value

def conduct_ab_test(control_conversions, control_size, variation_conversions, variation_size):
    control_rate = control_conversions / control_size
    variation_rate = variation_conversions / variation_size
    
    z_score, p_value = proportions_ztest(
        control_conversions, control_size,
        variation_conversions, variation_size
    )
    
    relative_change = (variation_rate - control_rate) / control_rate
    
    return control_rate, variation_rate, z_score, p_value, relative_change

st.set_page_config(page_title="A/B Testing Analysis", layout="wide")

st.title("A/B Testing Analysis")

st.sidebar.header("Input Parameters")
control_conversions = st.sidebar.number_input("Control Conversions", min_value=0, value=100)
control_size = st.sidebar.number_input("Control Sample Size", min_value=1, value=1000)
variation_conversions = st.sidebar.number_input("Variation Conversions", min_value=0, value=120)
variation_size = st.sidebar.number_input("Variation Sample Size", min_value=1, value=1000)

if st.sidebar.button("Run Analysis"):
    control_rate, variation_rate, z_score, p_value, relative_change = conduct_ab_test(
        control_conversions, control_size, variation_conversions, variation_size
    )
    
    st.header("Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Control Conversion Rate", f"{control_rate:.2%}")
    col2.metric("Variation Conversion Rate", f"{variation_rate:.2%}")
    col3.metric("Relative Change", f"{relative_change:.2%}")
    
    st.subheader("Statistical Significance")
    st.write(f"Z-score: {z_score:.4f}")
    st.write(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        st.success("The result is statistically significant (p < 0.05)")
    else:
        st.warning("The result is not statistically significant (p >= 0.05)")
    
    st.subheader("Visualization")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot
    sns.barplot(x=['Control', 'Variation'], y=[control_rate, variation_rate], ax=ax1)
    ax1.set_ylabel('Conversion Rate')
    ax1.set_title('Conversion Rates Comparison')
    
    # Confidence Intervals
    control_ci = stats.norm.interval(0.95, loc=control_rate, scale=np.sqrt(control_rate * (1 - control_rate) / control_size))
    variation_ci = stats.norm.interval(0.95, loc=variation_rate, scale=np.sqrt(variation_rate * (1 - variation_rate) / variation_size))
    
    ax2.errorbar(['Control', 'Variation'], [control_rate, variation_rate], 
                 yerr=[[control_rate - control_ci[0], variation_rate - variation_ci[0]],
                       [control_ci[1] - control_rate, variation_ci[1] - variation_rate]],
                 fmt='o', capsize=5, capthick=2)
    ax2.set_ylabel('Conversion Rate')
    ax2.set_title('Conversion Rates with 95% Confidence Intervals')
    
    st.pyplot(fig)
    
    st.subheader("Interpretation")
    st.write(f"The control group had a conversion rate of {control_rate:.2%}, while the variation group had a conversion rate of {variation_rate:.2%}.")
    st.write(f"This represents a relative change of {relative_change:.2%} in the conversion rate.")
    
    if p_value < 0.05:
        st.write("Since the p-value is less than 0.05, we can conclude that there is a statistically significant difference between the two groups.")
    else:
        st.write("Since the p-value is greater than or equal to 0.05, we cannot conclude that there is a statistically significant difference between the two groups.")
    
    st.write("Consider the following when interpreting the results:")
    st.write("1. Statistical significance does not always imply practical significance.")
    st.write("2. Consider the effect size and confidence intervals when making decisions.")
    st.write("3. Think about the business impact and cost-benefit analysis of implementing the variation.")
    
    st.subheader("Sample Size Calculator")
    st.write("Use this calculator to determine the required sample size for future A/B tests:")
    
    baseline_conv_rate = st.number_input("Baseline Conversion Rate", min_value=0.0, max_value=1.0, value=0.1)
    mde = st.number_input("Minimum Detectable Effect", min_value=0.0, max_value=1.0, value=0.05)
    
    sample_size = calculate_sample_size(baseline_conv_rate, mde)
    st.write(f"Required sample size per group: {sample_size:.0f}")

st.sidebar.markdown("---")
st.sidebar.write("Created with Streamlit")