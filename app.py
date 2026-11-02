import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Calculus Function Reconstructor", layout="wide")

st.title("📐 Calculus Function & Derivative Analyzer")
st.write("Reconstruct original functions $f(x)$ from derivatives, initial conditions, or critical points, and visualize corresponding points aligned across vertically stacked plots.")

# -------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------
st.sidebar.header("Function Inputs")

mode = st.sidebar.radio(
    "Input Mode",
    ["Given Derivative f'(x)", "Given 2nd Derivative f''(x)", "Given Direct Function f(x)"]
)

x = sp.Symbol('x', real=True)

if mode == "Given Derivative f'(x)":
    f_prime_expr_str = st.sidebar.text_input("Enter f'(x)", "3*x**2 - 6*x")
    x0 = st.sidebar.number_input("Initial condition x₀", value=0.0)
    y0 = st.sidebar.number_input("Initial condition f(x₀)", value=1.0)
    
    try:
        f_prime_expr = sp.sympify(f_prime_expr_str)
        f_expr_integrated = sp.integrate(f_prime_expr, x)
        C1 = y0 - f_expr_integrated.subs(x, x0)
        f_expr = f_expr_integrated + C1
        f_double_prime_expr = sp.diff(f_prime_expr, x)
    except Exception as e:
        st.error(f"Error parsing expression: {e}")
        st.stop()

elif mode == "Given 2nd Derivative f''(x)":
    f_dp_expr_str = st.sidebar.text_input("Enter f''(x)", "6*x - 6")
    st.sidebar.markdown("**Initial Conditions**")
    x0 = st.sidebar.number_input("x₀ for f(x₀)", value=0.0)
    y0 = st.sidebar.number_input("f(x₀)", value=1.0)
    x1 = st.sidebar.number_input("x₁ for f'(x₁)", value=0.0)
    dy0 = st.sidebar.number_input("f'(x₁)", value=0.0)
    
    try:
        f_dp_expr = sp.sympify(f_dp_expr_str)
        f_prime_integrated = sp.integrate(f_dp_expr, x)
        C_prime = dy0 - f_prime_integrated.subs(x, x1)
        f_prime_expr = f_prime_integrated + C_prime
        
        f_integrated = sp.integrate(f_prime_expr, x)
        C_f = y0 - f_integrated.subs(x, x0)
        f_expr = f_integrated + C_f
        f_double_prime_expr = f_dp_expr
    except Exception as e:
        st.error(f"Error parsing expression: {e}")
        st.stop()

else:
    f_expr_str = st.sidebar.text_input("Enter f(x)", "x**3 - 3*x**2 + 1")
    try:
        f_expr = sp.sympify(f_expr_str)
        f_prime_expr = sp.diff(f_expr, x)
        f_double_prime_expr = sp.diff(f_prime_expr, x)
    except Exception as e:
        st.error(f"Error parsing expression: {e}")
        st.stop()

# Plot settings
st.sidebar.header("Plot Range")
x_min = st.sidebar.number_input("x min", value=-2.0)
x_max = st.sidebar.number_input("x max", value=4.0)

# Display solved equations
st.subheader("Calculated Mathematical Expressions")
col1, col2, col3 = st.columns(3)
with col1:
    st.latex(f"f(x) = {sp.latex(f_expr)}")
with col2:
    st.latex(f"f'(x) = {sp.latex(f_prime_expr)}")
with col3:
    st.latex(f"f''(x) = {sp.latex(f_double_prime_expr)}")

# -------------------------------------------------
# FIND CRITICAL POINTS AND INFLECTION POINTS
# -------------------------------------------------
critical_points = []
try:
    crit_sols = sp.solveset(f_prime_expr, x, domain=sp.S.Reals)
    for sol in crit_sols:
        val = float(sol.evalf())
        if x_min <= val <= x_max:
            critical_points.append(val)
except Exception:
    pass

inflection_points = []
try:
    infl_sols = sp.solveset(f_double_prime_expr, x, domain=sp.S.Reals)
    for sol in infl_sols:
        val = float(sol.evalf())
        if x_min <= val <= x_max:
            inflection_points.append(val)
except Exception:
    pass

# Numerical evaluation
x_vals = np.linspace(x_min, x_max, 500)
f_num = sp.lambdify(x, f_expr, "numpy")
f_prime_num = sp.lambdify(x, f_prime_expr, "numpy")
f_dp_num = sp.lambdify(x, f_double_prime_expr, "numpy")

y_f = f_num(x_vals)
y_fp = f_prime_num(x_vals)
y_fdp = f_dp_num(x_vals)

# Broadcast scalars if constant derivative
if np.isscalar(y_f): y_f = np.full_like(x_vals, y_f)
if np.isscalar(y_fp): y_fp = np.full_like(x_vals, y_fp)
if np.isscalar(y_fdp): y_fdp = np.full_like(x_vals, y_fdp)

# -------------------------------------------------
# VERTICALLY ALIGNED PLOTS
# -------------------------------------------------
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Original Function f(x)", "First Derivative f'(x)", "Second Derivative f''(x)")
)

# Curve traces
fig.add_trace(go.Scatter(x=x_vals, y=y_f, name="f(x)", line=dict(color="#1f77b4", width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=x_vals, y=y_fp, name="f'(x)", line=dict(color="#ff7f0e", width=3)), row=2, col=1)
fig.add_trace(go.Scatter(x=x_vals, y=y_fdp, name="f''(x)", line=dict(color="#2ca02c", width=3)), row=3, col=1)

# Zero lines for derivatives
fig.add_hline(y=0, line_dash="solid", line_color="rgba(150,150,150,0.5)", row=2, col=1)
fig.add_hline(y=0, line_dash="solid", line_color="rgba(150,150,150,0.5)", row=3, col=1)

# Alignment lines at Critical Points (Red Dashed)
for cp in critical_points:
    for row_idx in [1, 2, 3]:
        fig.add_vline(
            x=cp,
            line_dash="dash",
            line_color="crimson",
            line_width=1.5,
            row=row_idx, col=1
        )
    # Add marker on f(x)
    y_cp = float(f_expr.subs(x, cp).evalf())
    fig.add_trace(
        go.Scatter(x=[cp], y=[y_cp], mode="markers", marker=dict(size=8, color="crimson"), name=f"Critical x={cp:.2f}"),
        row=1, col=1
    )

# Alignment lines at Inflection Points (Purple Dashed)
for ip in inflection_points:
    for row_idx in [1, 2, 3]:
        fig.add_vline(
            x=ip,
            line_dash="dot",
            line_color="purple",
            line_width=1.5,
            row=row_idx, col=1
        )
    # Add marker on f(x)
    y_ip = float(f_expr.subs(x, ip).evalf())
    fig.add_trace(
        go.Scatter(x=[ip], y=[y_ip], mode="markers", marker=dict(size=8, color="purple"), name=f"Inflection x={ip:.2f}"),
        row=1, col=1
    )

fig.update_layout(
    height=800,
    showlegend=False,
    template="plotly_white",
    hovermode="x unified"
)

fig.update_xaxes(title_text="x", row=3, col=1)
fig.update_yaxes(title_text="f(x)", row=1, col=1)
fig.update_yaxes(title_text="f'(x)", row=2, col=1)
fig.update_yaxes(title_text="f''(x)", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# Legend summary below chart
st.markdown("""
**Graph Key:**
* 🔴 **Red Dashed Vertical Lines**: Critical Points ($f'(x) = 0$)
* 🟣 **Purple Dotted Vertical Lines**: Inflection Points ($f''(x) = 0$)
""")
