"""
Continuous Evaluation Dashboard
===============================

Real-time monitoring dashboard for production vs experimental systems.
Provides governance visibility and operational monitoring.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import json
from pathlib import Path
import time

# Import governance systems
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from shared.governance.graduation_ledger import get_graduation_ledger
    from shared.testing.contract_testing import get_experimental_metrics
    from shared.utils.circuit_breaker import get_graceful_degradation_manager
    from experimental import list_experimental_features, FeatureContext
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Initialize governance systems
graduation_ledger = get_graduation_ledger()
experimental_metrics = get_experimental_metrics()
graceful_degradation = get_graceful_degradation_manager()

def render_dashboard():
    """Render the main evaluation dashboard"""

    st.title("🧪 Oviya EI Continuous Evaluation Dashboard")
    st.markdown("*Real-time monitoring of production and experimental systems*")

    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 30, 300, 60)
    show_experimental = st.sidebar.checkbox("Show Experimental Components", True)
    risk_filter = st.sidebar.selectbox("Risk Level Filter", ["All", "Low", "Medium", "High", "Critical"])

    # Auto-refresh
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()

    # Main dashboard sections
    render_system_overview()
    render_performance_metrics()
    render_safety_compliance()
    render_graduation_pipeline()

    if show_experimental:
        render_experimental_status()

    render_recent_activity()

    # Footer with auto-refresh
    st.markdown("---")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: {refresh_rate}s*")

    # Auto-refresh functionality
    time.sleep(refresh_rate)
    st.rerun()

def render_system_overview():
    """Render overall system status overview"""

    st.header("🏥 System Overview")

    # Get system status data
    system_status = get_system_status_data()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Production Components",
            system_status["production_components"],
            help="Clinically validated production systems"
        )

    with col2:
        st.metric(
            "Experimental Components",
            system_status["experimental_components"],
            delta=system_status["experimental_delta"],
            help="Components under evaluation"
        )

    with col3:
        graduated_count = len(graduation_ledger.get_graduation_history())
        st.metric(
            "Graduated Components",
            graduated_count,
            delta=system_status["graduation_delta"],
            help="Components promoted to production"
        )

    with col4:
        uptime = system_status["system_uptime"]
        st.metric(
            "System Uptime",
            f"{uptime:.1f}%",
            delta=system_status["uptime_delta"],
            help="Overall system availability"
        )

    # System health indicators
    health_col1, health_col2 = st.columns(2)

    with health_col1:
        st.subheader("Production Health")
        prod_health = system_status["production_health"]
        if prod_health["status"] == "excellent":
            st.success("🟢 Excellent")
        elif prod_health["status"] == "good":
            st.warning("🟡 Good")
        else:
            st.error("🔴 Needs Attention")

        st.caption(f"Crisis Detection: {prod_health['crisis_detection']}%")
        st.caption(f"PII Redaction: {prod_health['pii_redaction']}%")

    with health_col2:
        st.subheader("Experimental Health")
        exp_health = system_status["experimental_health"]
        if exp_health["status"] == "operational":
            st.info("🔵 Operational")
        elif exp_health["status"] == "degraded":
            st.warning("🟡 Degraded")
        else:
            st.error("🔴 Critical Issues")

        active_breakers = exp_health["active_circuit_breakers"]
        st.caption(f"Active Circuit Breakers: {active_breakers}")

def render_performance_metrics():
    """Render performance comparison charts"""

    st.header("📊 Performance Metrics")

    # Get performance data
    perf_data = get_performance_data()

    # Create performance comparison chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Production',
        x=list(perf_data["production"].keys()),
        y=list(perf_data["production"].values()),
        marker_color='green',
        opacity=0.8
    ))

    fig.add_trace(go.Bar(
        name='Experimental (Avg)',
        x=list(perf_data["experimental"].keys()),
        y=list(perf_data["experimental"].values()),
        marker_color='orange',
        opacity=0.8
    ))

    fig.add_trace(go.Bar(
        name='Target',
        x=list(perf_data["targets"].keys()),
        y=list(perf_data["targets"].values()),
        marker_color='red',
        opacity=0.3
    ))

    fig.update_layout(
        barmode='group',
        title="Production vs Experimental Performance",
        xaxis_title="Metrics",
        yaxis_title="Values"
    )

    st.plotly_chart(fig)

    # Performance status indicators
    col1, col2, col3 = st.columns(3)

    with col1:
        latency_status = "✅" if perf_data["production"]["Latency (ms)"] <= 130 else "⚠️"
        st.metric("Latency Status", f"{latency_status} {perf_data['production']['Latency (ms)']}ms")

    with col2:
        error_status = "✅" if perf_data["production"]["Error Rate (%)"] <= 0.5 else "⚠️"
        st.metric("Error Rate", f"{error_status} {perf_data['production']['Error Rate (%)']}%")

    with col3:
        safety_status = "✅" if perf_data["production"]["Safety Score"] >= 95 else "⚠️"
        st.metric("Safety Score", f"{safety_status} {perf_data['production']['Safety Score']}")

def render_safety_compliance():
    """Render safety and compliance metrics"""

    st.header("🛡️ Safety & Compliance")

    # Get safety data
    safety_data = get_safety_data()

    # Safety incidents chart
    fig1 = go.Figure(data=[
        go.Bar(
            x=safety_data["categories"],
            y=safety_data["incidents"],
            marker_color=['green', 'orange', 'red'],
            name='Safety Incidents'
        )
    ])
    fig1.update_layout(title="Safety Incidents by Category")
    st.plotly_chart(fig1)

    # Compliance scores
    fig2 = go.Figure(data=[
        go.Bar(
            x=safety_data["categories"],
            y=safety_data["compliance"],
            marker_color=['green', 'orange', 'red'],
            name='Compliance Score'
        )
    ])
    fig2.update_layout(title="Compliance Scores by Category")
    st.plotly_chart(fig2)

    # Compliance status table
    st.subheader("Compliance Status")
    compliance_df = pd.DataFrame({
        "Category": safety_data["categories"],
        "Compliance Score": safety_data["compliance"],
        "Incidents": safety_data["incidents"],
        "Status": ["✅ Compliant" if score >= 95 else "⚠️ Review Needed" for score in safety_data["compliance"]]
    })
    st.dataframe(compliance_df)

def render_graduation_pipeline():
    """Render graduation pipeline status"""

    st.header("🎓 Graduation Pipeline")

    # Get graduation data
    grad_data = get_graduation_data()

    # Graduation status overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Components Graduated", grad_data["total_graduated"])

    with col2:
        st.metric("Pending Review", grad_data["pending_review"])

    with col3:
        st.metric("Ready for Graduation", grad_data["ready_for_graduation"])

    # Recent graduations
    if grad_data["recent_graduations"]:
        st.subheader("Recent Graduations")
        for grad in grad_data["recent_graduations"][-3:]:  # Show last 3
            with st.expander(f"🎉 {grad['component']} → {grad['promoted_to']}"):
                st.write(f"**Version:** {grad['version']}")
                st.write(f"**Date:** {grad['decision_date']}")
                st.write(f"**Safety Score:** {grad['metrics']['safety_score']}")
                st.write(f"**Approved by:** {', '.join(grad['approved_by'])}")

    # Components ready for graduation
    if grad_data["graduation_candidates"]:
        st.subheader("Ready for Graduation")
        for candidate in grad_data["graduation_candidates"]:
            st.success(f"✅ {candidate}")

def render_experimental_status():
    """Render experimental component status"""

    st.header("🧪 Experimental Component Status")

    # Get experimental status
    exp_status = get_experimental_status()

    # Circuit breaker status
    st.subheader("Circuit Breaker Status")
    breaker_df = pd.DataFrame(exp_status["circuit_breakers"])
    if not breaker_df.empty:
        st.dataframe(breaker_df)
    else:
        st.info("No circuit breakers active")

    # Experimental features status
    st.subheader("Experimental Features")
    features_df = pd.DataFrame({
        "Feature": list(exp_status["features"].keys()),
        "Enabled": list(exp_status["features"].values()),
        "Status": ["✅ Active" if enabled else "⏸️ Disabled" for enabled in exp_status["features"].values()]
    })
    st.dataframe(features_df)

    # Performance metrics for experimental components
    if exp_status["performance_metrics"]:
        st.subheader("Experimental Performance")
        perf_df = pd.DataFrame(exp_status["performance_metrics"])
        st.dataframe(perf_df)

def render_recent_activity():
    """Render recent system activity"""

    st.header("📈 Recent Activity")

    # Get recent activity data
    activity_data = get_recent_activity()

    # Activity timeline
    if activity_data:
        for activity in activity_data[-10:]:  # Show last 10 activities
            timestamp = activity.get("timestamp", "Unknown")
            event_type = activity.get("type", "Unknown")
            description = activity.get("description", "No description")

            if event_type == "graduation":
                st.success(f"🎉 {timestamp}: {description}")
            elif event_type == "safety_incident":
                st.error(f"🚨 {timestamp}: {description}")
            elif event_type == "performance_degradation":
                st.warning(f"⚠️ {timestamp}: {description}")
            else:
                st.info(f"ℹ️ {timestamp}: {description}")
    else:
        st.info("No recent activity to display")

# Data retrieval functions (would integrate with actual monitoring systems)

def get_system_status_data():
    """Get overall system status data"""
    return {
        "production_components": 30,
        "experimental_components": 35,
        "experimental_delta": "+2",
        "graduation_delta": "+1",
        "system_uptime": 99.7,
        "uptime_delta": "+0.1",
        "production_health": {
            "status": "excellent",
            "crisis_detection": 100,
            "pii_redaction": 100
        },
        "experimental_health": {
            "status": "operational",
            "active_circuit_breakers": 0
        }
    }

def get_performance_data():
    """Get performance comparison data"""
    return {
        "production": {
            "Latency (ms)": 125,
            "Error Rate (%)": 0.02,
            "Safety Score": 100,
            "User Satisfaction": 4.6
        },
        "experimental": {
            "Latency (ms)": 145,
            "Error Rate (%)": 0.15,
            "Safety Score": 98,
            "User Satisfaction": 4.3
        },
        "targets": {
            "Latency (ms)": 150,
            "Error Rate (%)": 0.5,
            "Safety Score": 95,
            "User Satisfaction": 4.0
        }
    }

def get_safety_data():
    """Get safety and compliance data"""
    return {
        "categories": ["Production", "Experimental", "Graduated"],
        "incidents": [0, 2, 1],
        "compliance": [100, 98, 99]
    }

def get_graduation_data():
    """Get graduation pipeline data"""
    graduations = graduation_ledger.get_graduation_history()
    return {
        "total_graduated": len(graduations),
        "pending_review": 3,  # Would be calculated from actual data
        "ready_for_graduation": len(experimental_metrics.check_graduation_candidates(graduation_ledger.contract_tester)),
        "recent_graduations": graduations[-5:] if graduations else []
    }

def get_experimental_status():
    """Get experimental component status"""
    features = list_experimental_features()
    circuit_breakers = graceful_degradation.get_system_status()

    return {
        "features": features,
        "circuit_breakers": [
            {
                "Component": name,
                "State": status["state"].upper(),
                "Failures": status["failures"],
                "Last Failure": status.get("last_failure", "None")
            }
            for name, status in circuit_breakers.get("circuit_breakers", {}).items()
        ],
        "performance_metrics": []  # Would be populated with actual metrics
    }

def get_recent_activity():
    """Get recent system activity"""
    # This would integrate with actual logging/monitoring systems
    return [
        {
            "timestamp": "2025-10-30 14:30",
            "type": "graduation",
            "description": "audio_input.py promoted to production"
        },
        {
            "timestamp": "2025-10-30 14:15",
            "type": "performance_check",
            "description": "All production components within performance targets"
        },
        {
            "timestamp": "2025-10-30 14:00",
            "type": "safety_check",
            "description": "PII redaction validated across all components"
        }
    ]

# Main app
if __name__ == "__main__":
    render_dashboard()
