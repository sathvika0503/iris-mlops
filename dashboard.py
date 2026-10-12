
import streamlit as st
import requests
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
API_URL = "http://127.0.0.1:8000"
MLFLOW_URL = "http://127.0.0.1:5000"
DATA_PATH = Path(__file__).parent / "data" / "iris.csv"

st.set_page_config(
    page_title="Iris MLOps Dashboard",
    page_icon="🌸",
    layout="wide"
)

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def api_request(method, endpoint, **kwargs):
    """Call the FastAPI backend safely."""
    try:
        response = requests.request(
            method,
            f"{API_URL}{endpoint}",
            timeout=60,
            **kwargs
        )
        response.raise_for_status()

        if not response.content:
            return {}

        return response.json()

    except requests.exceptions.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None

    except ValueError:
        st.error("The API returned a response that is not valid JSON.")
        return None


def mlflow_request(method, endpoint, **kwargs):
    """Call the MLflow REST API safely."""
    try:
        response = requests.request(
            method,
            f"{MLFLOW_URL}{endpoint}",
            timeout=20,
            **kwargs
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as exc:
        st.error(f"MLflow request failed: {exc}")
        return None

    except ValueError:
        st.error("MLflow returned a response that is not valid JSON.")
        return None


def find_value(data, possible_names, default=None):
    """Find the first matching key in a dictionary."""
    if not isinstance(data, dict):
        return default

    for name in possible_names:
        if name in data:
            return data[name]

    return default


def get_experiment_id():
    """Find the MLflow experiment used by this project."""
    result = mlflow_request(
        "GET",
        "/api/2.0/mlflow/experiments/get-by-name",
        params={"experiment_name": "iris-mlops-starter"}
    )

    if not result:
        return None

    experiment = result.get("experiment", {})
    return experiment.get("experiment_id")


def get_run_value(items, key):
    """Read a metric or parameter from MLflow's list format."""
    if not isinstance(items, list):
        return None

    for item in items:
        if isinstance(item, dict) and item.get("key") == key:
            return item.get("value")

    return None


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🌸 Iris MLOps Dashboard")
st.write(
    "Manage the dataset, predict flower species, "
    "retrain the model, and monitor model performance."
)

st.caption(
    f"FastAPI: {API_URL}  |  MLflow: {MLFLOW_URL}"
)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset",
    "🔮 Prediction",
    "🔄 Retraining",
    "📈 Monitoring"
])


# ==================================================
# TAB 1: DATASET
# ==================================================
with tab1:
    st.header("📊 Iris Dataset")

    if DATA_PATH.exists():
        try:
            dataset = pd.read_csv(DATA_PATH)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", len(dataset))
            col2.metric("Total Columns", len(dataset.columns))

            if "species" in dataset.columns:
                col3.metric(
                    "Species",
                    dataset["species"].nunique()
                )
            else:
                col3.metric("Species", "Column not found")

            st.subheader("Dataset Preview")
            st.dataframe(dataset.head(20), use_container_width=True)

            st.subheader("Edit Dataset")
            st.caption(
                "Edit the table below and save your changes to data/iris.csv."
            )

            edited_dataset = st.data_editor(
                dataset,
                num_rows="dynamic",
                use_container_width=True,
                key="iris_dataset_editor"
            )

            if st.button("💾 Save Dataset", key="save_dataset"):
                try:
                    DATA_PATH.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )
                    edited_dataset.to_csv(DATA_PATH, index=False)
                    st.success("Dataset saved successfully!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save dataset: {exc}")

            st.subheader("Dataset Statistics")
            st.dataframe(
                dataset.describe(include="all").transpose(),
                use_container_width=True
            )

            if "species" in dataset.columns:
                st.subheader("Species Distribution")
                counts = dataset["species"].value_counts()
                st.bar_chart(counts)

        except Exception as exc:
            st.error(f"Could not read the dataset: {exc}")
    else:
        st.warning(f"Dataset file not found: {DATA_PATH}")
        st.info(
            "Check that data/iris.csv exists inside your project folder."
        )


# ==================================================
# TAB 2: PREDICTION
# ==================================================
with tab2:
    st.header("🔮 Predict Iris Flower Species")
    st.write("Enter the flower's measurements in centimeters.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            sepal_length = st.number_input(
                "Sepal Length (cm)",
                min_value=0.1,
                max_value=10.0,
                value=5.1,
                step=0.1
            )

            sepal_width = st.number_input(
                "Sepal Width (cm)",
                min_value=0.1,
                max_value=10.0,
                value=3.5,
                step=0.1
            )

        with col2:
            petal_length = st.number_input(
                "Petal Length (cm)",
                min_value=0.1,
                max_value=10.0,
                value=1.4,
                step=0.1
            )

            petal_width = st.number_input(
                "Petal Width (cm)",
                min_value=0.1,
                max_value=10.0,
                value=0.2,
                step=0.1
            )

        predict_button = st.form_submit_button(
            "🌸 Predict Species",
            use_container_width=True
        )

    if predict_button:
        input_data = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }

        result = api_request(
            "POST",
            "/predict",
            json=input_data
        )

        if result is not None:
            st.subheader("Prediction Result")

            prediction = find_value(
                result,
                ["prediction", "predicted_class", "species", "predicted_species"]
            )

            if prediction is not None:
                st.success(f"Predicted Species: **{prediction}**")
            else:
                st.info("The API returned a response. See the details below.")

            confidence = find_value(
                result,
                ["confidence", "prediction_confidence"]
            )

            if isinstance(confidence, (int, float)):
                st.metric("Confidence", f"{confidence:.2%}")

            with st.expander("View Full Prediction Response"):
                st.json(result)


# ==================================================
# TAB 3: RETRAINING
# ==================================================
with tab3:
    st.header("🔄 Model Retraining")
    st.write(
        "Retrain and evaluate the machine-learning models "
        "using the current dataset."
    )

    st.warning(
        "Retraining may take a little time. Keep FastAPI and MLflow running."
    )

    if st.button(
        "🚀 Retrain Model",
        type="primary",
        key="retrain_model"
    ):
        with st.spinner("Retraining model. Please wait..."):
            result = api_request("POST", "/retrain")

        if result is not None:
            st.session_state["last_training_result"] = result
            st.success("Retraining request completed!")
            st.rerun()

    result = st.session_state.get("last_training_result")

    if result is not None:
        st.subheader("Training Results")

        st.json(result)

        # Display commonly used training metrics if available.
        if isinstance(result, dict):
            metrics_source = result.get("metrics", result)

            if isinstance(metrics_source, dict):
                metric_columns = st.columns(3)

                accuracy = find_value(
                    metrics_source,
                    ["test_accuracy", "accuracy"]
                )
                cv_accuracy = find_value(
                    metrics_source,
                    ["cv_accuracy", "cross_validation_accuracy"]
                )
                algorithm = find_value(
                    result,
                    ["selected_algorithm", "best_model", "algorithm"],
                    default="See full response"
                )

                if isinstance(accuracy, (int, float)):
                    metric_columns[0].metric(
                        "Test Accuracy",
                        f"{accuracy:.2%}"
                    )

                if isinstance(cv_accuracy, (int, float)):
                    metric_columns[1].metric(
                        "CV Accuracy",
                        f"{cv_accuracy:.2%}"
                    )

                metric_columns[2].write(
                    f"**Selected Model**\n\n{algorithm}"
                )

            comparison = result.get("model_comparison")

            if isinstance(comparison, list) and comparison:
                st.subheader("Model Comparison")
                comparison_df = pd.DataFrame(comparison)
                st.dataframe(
                    comparison_df,
                    use_container_width=True
                )

            elif isinstance(comparison, dict) and comparison:
                st.subheader("Model Comparison")
                st.dataframe(
                    pd.DataFrame(comparison),
                    use_container_width=True
                )

            confusion_matrix = result.get("confusion_matrix")

            if isinstance(confusion_matrix, list):
                st.subheader("Confusion Matrix")
                labels = result.get("confusion_matrix_labels")

                if labels and len(labels) == len(confusion_matrix):
                    cm_df = pd.DataFrame(
                        confusion_matrix,
                        index=labels,
                        columns=labels
                    )
                else:
                    cm_df = pd.DataFrame(confusion_matrix)

                st.dataframe(cm_df, use_container_width=True)

            report = result.get("classification_report")

            if isinstance(report, dict):
                st.subheader("Classification Report")
                try:
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(
                        report_df,
                        use_container_width=True
                    )
                except Exception:
                    st.json(report)

        with st.expander("View Full Training Response"):
            st.json(result)

    else:
        st.info(
            "No training result is saved in this dashboard session yet. "
            "Click Retrain Model to start."
        )


# ==================================================
# TAB 4: MODEL MONITORING
# ==================================================
with tab4:
    st.header("📈 Model Monitoring Dashboard")
    st.write(
        "Monitor backend health, training history, "
        "and model performance recorded by MLflow."
    )

    if st.button("🔄 Refresh Monitoring", key="refresh_monitoring"):
        st.rerun()

    # ----------------------------------------------
    # SYSTEM HEALTH
    # ----------------------------------------------
    st.subheader("System Health")

    health = api_request("GET", "/health")

    col1, col2 = st.columns(2)

    if health is not None:
        col1.success("API: Online")

        with col1.expander("API Health Response"):
            st.json(health)

        model_status = find_value(
            health,
            ["model_loaded", "model_available", "model_status"]
        )

        if model_status is False:
            col2.error("Model: Not Available")
        elif model_status is True:
            col2.success("Model: Available")
        elif isinstance(model_status, str):
            col2.info(f"Model: {model_status}")
        else:
            # API is responding, but the health endpoint may not
            # explicitly report the model's loaded status.
            col2.info("Model status not specified by /health")
    else:
        col1.error("API: Offline")
        col2.warning("Model: Status Unknown")
    
    st.subheader("⚙️ Alert Configuration")

    accuracy_threshold = st.slider(
        "Minimum acceptable test accuracy (%)",
        min_value=50,
        max_value=100,
        value=95,
        step=1,
        help="Show a warning when the latest recorded test accuracy is below this value."
    )

    st.caption(
        f"Alert threshold: {accuracy_threshold}%"
    )

    # ----------------------------------------------
    # MLFLOW TRAINING HISTORY
    # ----------------------------------------------
    st.subheader("Training History")

    experiment_id = get_experiment_id()

    if experiment_id is None:
        st.warning(
            "Could not find the 'iris-mlops-starter' experiment in MLflow. "
            "Check that MLflow is running and the experiment exists."
        )
    else:
        runs_response = mlflow_request(
            "POST",
            "/api/2.0/mlflow/runs/search",
            json={
                "experiment_ids": [str(experiment_id)],
                "max_results": 100,
                "order_by": ["attributes.start_time DESC"]
            }
        )

        if runs_response is not None:
            runs = runs_response.get("runs", [])

            if not isinstance(runs, list):
                st.error("MLflow returned an unexpected runs format.")
                runs = []

            if not runs:
                st.info(
                    "No training runs found yet. Retrain the model "
                    "to record a run in MLflow."
                )
            else:
                history = []

                for run in runs:
                    if not isinstance(run, dict):
                        continue

                    info = run.get("info", {})
                    data = run.get("data", {})

                    params = data.get("params", [])
                    metrics = data.get("metrics", [])

                    algorithm = (
                        get_run_value(params, "selected_algorithm")
                        or get_run_value(params, "algorithm")
                        or get_run_value(params, "model_name")
                        or "Not recorded"
                    )

                    test_accuracy = (
                        get_run_value(metrics, "test_accuracy")
                    )

                    cv_accuracy = (
                        get_run_value(metrics, "cv_accuracy")
                        or get_run_value(metrics, "cross_validation_accuracy")
                    )

                    start_time = info.get("start_time")
                    if start_time:
                        try:
                            start_time = datetime.fromtimestamp(
                                start_time / 1000
                            ).strftime("%Y-%m-%d %H:%M:%S")
                        except (TypeError, ValueError, OSError):
                            start_time = str(start_time)
                    else:
                        start_time = "Unknown"

                    history.append({
                        "Algorithm": algorithm,
                        "Test Accuracy": (
                            float(test_accuracy)
                            if test_accuracy is not None
                            else None
                        ),
                        "CV Accuracy": (
                            float(cv_accuracy)
                            if cv_accuracy is not None
                            else None
                        ),
                        "Training Time": start_time,
                        "Run ID": info.get("run_id", "")
                    })

                if history:
                    history_df = pd.DataFrame(history)

                    # Summary metrics for the most recent run.
                    latest = history_df.iloc[0]
                    
                    latest_accuracy = latest["Test Accuracy"]

                    if pd.isna(latest_accuracy):
                        st.warning(
                            "⚠️ PERFORMANCE UNKNOWN: "
                            "The latest training run has no recorded test accuracy."
                        )
                    elif latest_accuracy < accuracy_threshold / 100:
                        st.error(
                            f"🚨 PERFORMANCE ALERT: Test accuracy is "
                            f"{latest_accuracy:.2%}, below your "
                            f"{accuracy_threshold}% threshold!"
                        )
                    else:
                        st.success(
                            f"✅ MODEL HEALTHY: Test accuracy is "
                            f"{latest_accuracy:.2%}, meeting the "
                            f"{accuracy_threshold}% threshold."
                        )
                    
                    # Compare latest accuracy with previous training run
                    if len(history_df) >= 2:
                        previous_accuracy = history_df.iloc[1]["Test Accuracy"]

                        if (
                            pd.notna(latest_accuracy)
                            and pd.notna(previous_accuracy)
                        ):
                            accuracy_drop = (
                                previous_accuracy - latest_accuracy
                            ) * 100

                            if accuracy_drop > 0:
                                st.warning(
                                    f"📉 Accuracy dropped by "
                                    f"{accuracy_drop:.2f} percentage points "
                                    f"compared with the previous run."
                                )
                            else:
                                st.info(
                                    "📊 No accuracy drop compared with "
                                    "the previous training run."
                                )
                    else:
                        st.info(
                            "Run retraining at least twice to compare "
                            "accuracy between runs."
                        )


                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "Recorded Runs",
                        len(history_df)
                    )

                    latest_test = latest["Test Accuracy"]
                    if pd.notna(latest_test):
                        col2.metric(
                            "Latest Test Accuracy",
                            f"{latest_test:.2%}"
                        )
                    else:
                        col2.metric(
                            "Latest Test Accuracy",
                            "Not recorded"
                        )

                    latest_cv = latest["CV Accuracy"]
                    if pd.notna(latest_cv):
                        col3.metric(
                            "Latest CV Accuracy",
                            f"{latest_cv:.2%}"
                        )
                    else:
                        col3.metric(
                            "Latest CV Accuracy",
                            "Not recorded"
                        )

                    st.subheader("MLflow Run History")
                    st.dataframe(
                        history_df,
                        use_container_width=True
                    )
                    
                    # PDF Monitoring Report
                    st.subheader("📄 Download PDF Report")

                    from io import BytesIO
                    from reportlab.lib.pagesizes import A4, landscape
                    from reportlab.platypus import (
                        SimpleDocTemplate,
                        Paragraph,
                        Spacer,
                        Table,
                        TableStyle
                    )
                    from reportlab.lib import colors
                    from reportlab.lib.styles import getSampleStyleSheet

                    pdf_buffer = BytesIO()

                    pdf_doc = SimpleDocTemplate(
                        pdf_buffer,
                        pagesize=landscape(A4)
                    )

                    styles = getSampleStyleSheet()
                    story = []

                    story.append(
                        Paragraph(
                            "Iris MLOps Monitoring Report",
                            styles["Title"]
                        )
                    )
                    story.append(Spacer(1, 15))

                    story.append(
                        Paragraph(
                            f"Report generated: "
                            f"{datetime.now().strftime('%d-%m-%Y %H:%M')}",
                            styles["Normal"]
                        )
                    )
                    story.append(Spacer(1, 15))

                    pdf_rows = [[
                        "Algorithm",
                        "Test Accuracy",
                        "CV Accuracy",
                        "Training Time",
                        "Run ID"
                    ]]

                    for _, row in history_df.iterrows():
                        test_acc = row["Test Accuracy"]
                        cv_acc = row["CV Accuracy"]

                        pdf_rows.append([
                            str(row["Algorithm"]),
                            f"{test_acc:.2%}" if pd.notna(test_acc) else "N/A",
                            f"{cv_acc:.2%}" if pd.notna(cv_acc) else "N/A",
                            str(row["Training Time"]),
                            str(row["Run ID"])
                        ])

                    pdf_table = Table(pdf_rows, repeatRows=1)

                    pdf_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]))

                    story.append(pdf_table)
                    pdf_doc.build(story)

                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name="iris_monitoring_report.pdf",
                        mime="application/pdf",
                        key="download_pdf_direct"
                    )

                    
                    # Downloadable Model Monitoring Report
                    st.subheader("📥 Download Monitoring Report")

                    report_df = history_df.copy()

                    report_df["Alert Threshold (%)"] = (
                        accuracy_threshold
                    )

                    def get_alert_status(row):
                        accuracy = row["Test Accuracy"]

                        if pd.isna(accuracy):
                            return "Unknown"
                        elif accuracy < accuracy_threshold / 100:
                            return "Performance Alert"
                        else:
                            return "Healthy"

                    report_df["Alert Status"] = report_df.apply(
                        get_alert_status,
                        axis=1
                    )

                    csv_data = report_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Monitoring Report (CSV)",
                        data=csv_data,
                        file_name="iris_model_monitoring_report.csv",
                        mime="text/csv",
                        key="download_monitoring_report"
                    )


                    accuracy_df = history_df[
                        ["Training Time", "Test Accuracy", "CV Accuracy"]
                    ].copy()

                    accuracy_df = accuracy_df.dropna(
                        subset=["Test Accuracy", "CV Accuracy"],
                        how="all"
                    )

                    if not accuracy_df.empty:
                        
                        st.subheader("📈 Model Accuracy Across Runs")

                        chart_df = history_df[
                            ["Training Time", "Test Accuracy", "CV Accuracy"]
                        ].copy()

                        chart_df = chart_df.dropna(
                            subset=["Test Accuracy", "CV Accuracy"],
                            how="all"
                        )

                        chart_df = chart_df.iloc[::-1]

                        chart_df["Alert Threshold"] = (
                            accuracy_threshold / 100
                        )

                        chart_df = chart_df.set_index("Training Time")

                        st.line_chart(
                            chart_df[
                                [
                                    "Test Accuracy",
                                    "CV Accuracy",
                                    "Alert Threshold"
                                ]
                            ],
                        
                        
                        )

                        st.caption(
                            "The Alert Threshold line shows your configured "
                            "minimum acceptable test accuracy."
                        )


                    with st.expander("View Raw MLflow Response"):
                        st.json(runs_response)

                else:
                    st.info(
                        "MLflow returned runs, but no usable run details "
                        "were found."
                    )

    st.caption(
        "Monitoring displays information recorded by the API and MLflow. "
        "It does not automatically detect data drift unless drift checks "
        "are implemented separately."
    )
