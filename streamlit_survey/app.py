import streamlit as st
import pandas as pd
from pathlib import Path
import random
from datetime import datetime, UTC
import re
import gspread
from gspread.exceptions import APIError, WorksheetNotFound
import time
import os

DATA_PATH = Path(__file__).parent / "comments_id_and_content_survey.csv"
ATTENTION_CHECK_PATH = Path(__file__).parent / "attention_check.csv"
RESULTS_PATH = Path("results.csv")

RANDOM_SEED = 42

COMMENTS_PER_PARTICIPANT = 13
ATTENTION_CHECK_IDS = ["attention_check_1", "attention_check_2"]
ATTENTION_CHECK_PASS_SCORE = 2
TOTAL_QUESTIONS = COMMENTS_PER_PARTICIPANT + len(ATTENTION_CHECK_IDS)

GOOGLE_SHEETS_URL = os.getenv("GOOGLE_SHEETS_URL")
if not GOOGLE_SHEETS_URL:
    raise ValueError("Missing GOOGLE_SHEETS_URL environment variable")

COMPLETION_CODE_SUCCESS = os.getenv("COMPLETION_CODE_SUCCESS", "C1HPZ924")
COMPLETION_CODE_ATTENTION_FAIL = os.getenv("COMPLETION_CODE_ATTENTION_FAIL", "CP6DIHIC")
COMPLETION_CODE_LLM_YES = os.getenv("COMPLETION_CODE_LLM_YES", "CSS5E0VU")

STATUS_ASSIGNED = "assigned"
STATUS_VALID = "valid"
STATUS_INVALID_ATTN = "invalid_attention"
STATUS_INVALID_LLM = "invalid_llm"
STATUS_RETURNED = "returned"

TOTAL_BATCHES = 12

PARTICIPANTS_PER_BATCH = 15

gsheets_secrets = st.secrets["connections"]["gsheets"]

def get_batches(file_ids):
    """Split *regular* comment ids into deterministic batches of equal size."""
    ids = file_ids.copy()
    random.Random(RANDOM_SEED).shuffle(ids)
    return [
        ids[i * COMMENTS_PER_PARTICIPANT : (i + 1) * COMMENTS_PER_PARTICIPANT]
        for i in range(TOTAL_BATCHES)
    ]

def _retry_on_quota(func, *args, **kwargs):
    """Run a gspread call, retrying on 429 quota errors after a 60-second wait."""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except APIError as err:
            if getattr(err, "response", None) and err.response.status_code == 429:
                if attempt < max_attempts - 1:
                    time.sleep(60)
                    continue
            raise

@st.cache_resource(show_spinner=False)
def init_sheets():
    client = gspread.service_account_from_dict(gsheets_secrets)
    spreadsheet = _retry_on_quota(
        client.open_by_url,
        GOOGLE_SHEETS_URL,
    )

    def get_ws(name):
        try:
            return _retry_on_quota(spreadsheet.worksheet, name)
        except WorksheetNotFound:
            return _retry_on_quota(spreadsheet.add_worksheet, title=name, rows="1000", cols="6")

    results = get_ws("results")
    order = get_ws("order")
    return results, order

def safe_append(ws, row):
    _retry_on_quota(ws.append_row, row)

results_ws, order_ws = init_sheets()

def main():
    """Streamlit app for rating climate policy engagement comments."""
    st.set_page_config(page_title="Climate Policy Engagement Survey", layout="wide")

    defaults = {
        "prolific_id": "",
        "comment_order": [],
        "current_index": 0,
        "pending_rows": [],
        "attention_failures": 0,
        "failed_attention": False,
        "llm_answered": False,
        "llm_used_yes": False,
        "order_id": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    if not st.session_state["prolific_id"]:
        prolific_id_screen()
    elif not st.session_state["comment_order"]:
        setup_survey()
        introduction_screen()
    elif st.session_state["current_index"] < TOTAL_QUESTIONS:
        survey_screen()
    elif not st.session_state["llm_answered"]:
        llm_usage_screen()
    else:
        thank_you_screen()

def prolific_id_screen():
    """Screen for entering Prolific ID and verifying before continuing."""
    st.title("Enter your Prolific ID")
    with st.form("prolific_id_form", clear_on_submit=False):
        prolific_id = st.text_input("Please enter your Prolific ID to begin:")
        if prolific_id:
            st.markdown(f"Your Prolific ID will be recorded as: **{prolific_id}**")
        submitted = st.form_submit_button("Continue")
        if submitted and prolific_id:
            try:
                existing_orders = order_ws.get_all_records()
            except Exception:
                existing_orders = []

            for order in existing_orders:
                if str(order.get("prolific_id", "")).strip() == prolific_id.strip():
                    st.error(
                        "A survey session is already registered for this Prolific ID. "
                        "If you believe this is a mistake, please contact the organiser."
                    )
                    return

            st.session_state["prolific_id"] = prolific_id.strip()
            st.rerun()

def assign_comment_order(prolific_id: str, df_comments: pd.DataFrame) -> list[str]:
    """Return list of **regular** comment file-ids to present to this participant.

    The assignment is persisted to the *order* worksheet so a participant gets the
    exact same set of comments if they return to the survey later.
    """

    try:
        existing = order_ws.get_all_records()
    except Exception:
        existing = []

    for row in existing:
        if row.get("prolific_id") == prolific_id:
            try:
                st.session_state["order_id"] = int(row.get("order_id", 0))
            except (TypeError, ValueError):
                st.session_state["order_id"] = None
            return row["comment_order"].split(",")

    valid_counts: dict[int, int] = {}
    for r in existing:
        if r.get("status") in {STATUS_VALID, STATUS_ASSIGNED}:
            try:
                b_idx = int(r.get("batch_index", 0))
            except (TypeError, ValueError):
                continue
            valid_counts[b_idx] = valid_counts.get(b_idx, 0) + 1

    batch_index = 0
    for b in range(TOTAL_BATCHES):
        if valid_counts.get(b, 0) < PARTICIPANTS_PER_BATCH:
            batch_index = b
            break

    max_id = 0
    for r in existing:
        try:
            max_id = max(max_id, int(str(r.get("order_id", "")).strip()))
        except ValueError:
            continue
    order_index = max_id + 1

    batches = get_batches(df_comments["file_id"].tolist())
    comment_ids = batches[batch_index].copy()
    rnd = random.Random(hash(prolific_id) & 0xFFFFFFFF)
    rnd.shuffle(comment_ids)

    timestamp = datetime.now(UTC).isoformat()
    row = [order_index, prolific_id, ",".join(comment_ids), timestamp, batch_index, STATUS_ASSIGNED]
    safe_append(order_ws, row)

    st.session_state["order_id"] = order_index

    return comment_ids

def setup_survey():
    """Assign deterministic comment order and merge in the attention checks."""

    if not DATA_PATH.exists() or not ATTENTION_CHECK_PATH.exists():
        st.error("Data files not found on server. Please contact the organiser.")
        st.stop()

    df_comments = pd.read_csv(DATA_PATH, dtype=str)

    df_attention = pd.read_csv(ATTENTION_CHECK_PATH, dtype=str)

    df_comments["file_id"] = df_comments["file_id"].str.strip()
    df_attention["file_id"] = df_attention["file_id"].str.strip()

    df_comments = df_comments[~df_comments["file_id"].isin(ATTENTION_CHECK_IDS)]

    regular_order = assign_comment_order(st.session_state["prolific_id"], df_comments)

    comment_order = regular_order.copy()
    insert_positions = [6, 9]
    for pos, ac_id in zip(insert_positions, ATTENTION_CHECK_IDS):
        comment_order.insert(pos, ac_id)

    combined_df = pd.concat(
        [df_comments.set_index("file_id"), df_attention.set_index("file_id")]
    )
    st.session_state["comment_order"] = comment_order
    st.session_state["comments_df"] = combined_df.loc[comment_order]

def introduction_screen():
    """Show the detailed task introduction."""
    st.title("Climate Policy Engagement Survey")
    st.markdown(
        """
        In this study, you will be asked to evaluate comments submitted in response to a U.S. regulator’s policy proposal that recommends steps financial institutions should take to address climate risk. You will be provided with a summary of each comment under the “Summary” tab. Please read the full summary and rate the comment on a scale from 1 to 5, as described below. If needed, you can refer to the full text of the comment under the “Comment” tab.
        """
    )

    st.markdown(
        """
        **Scoring definitions:**

        1 = Strong opposition to climate action by the regulator. Explicitly resists climate measures. May deny climate change or climate risks.

        2 = Skeptical or hesitant. Questions the need for special treatment or warns about costs and unintended consequences.

        3 = Neutral. Takes no strong position for or against climate action.

        4 = Supportive. Backs climate actions of the regulator. May support other climate measures. May advocate for more incremental steps.

        5 = Strong advocate. Fully supports ambitious, binding climate targets and broad reforms. May seek to strengthen proposed initiatives.
        """
    )

    st.markdown(
        """
        **Important:**

        • Please do not reload the page or close the browser tab until you have completed the survey.<br/>
        • You may have to double-click the **Next** button to proceed to the next question.
        """,
        unsafe_allow_html=True,
    )

    if st.button("Start Survey"):
        st.session_state["current_index"] = 0

def format_paragraphs(text: str) -> str:
    """Add extra space between paragraphs for readability in HTML."""
    text = re.sub(r"\n{2,}", "<br><br>", text)
    text = re.sub(r"(?<!<br>)\n", "<br>", text)
    return text

def survey_screen():
    """Screen that displays a single comment, gathers score, and handles logic."""
    idx = st.session_state["current_index"]
    file_id = st.session_state["comment_order"][idx]
    row = st.session_state["comments_df"].loc[file_id]

    st.subheader(f"Comment {idx + 1} of {TOTAL_QUESTIONS}")

    tabs = st.tabs(["Summary", "Comment"])
    with tabs[0]:
        summary_html = format_paragraphs(str(row["summary"]))
        st.markdown(
            f"<div style='max-height:500px;overflow:auto;resize:vertical;white-space:pre-wrap;font-family:sans-serif;font-size:12pt;background:#f8f9fa;padding:1em;border-radius:6px'>{summary_html}</div>",
            unsafe_allow_html=True,
        )
    with tabs[1]:
        comment_html = format_paragraphs(str(row["comment"]))
        st.markdown(
            f"<div style='max-height:500px;overflow:auto;resize:vertical;white-space:pre-wrap;font-family:sans-serif;font-size:12pt;background:#f8f9fa;padding:1em;border-radius:6px'>{comment_html}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("**How would you rate this comment?**")
    score_key = f"score_{file_id}_{idx}"
    if score_key not in st.session_state:
        st.session_state[score_key] = None

    score = st.radio(
        label="Select a score:",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{x} = {score_label(x)}",
        key=score_key,
        index=None,
    )

    if st.button("Next") and score is not None:
        prolific_id = st.session_state["prolific_id"]
        timestamp = datetime.now(UTC).isoformat()

        if file_id in ATTENTION_CHECK_IDS and score != ATTENTION_CHECK_PASS_SCORE:
            st.session_state["attention_failures"] += 1
            if st.session_state["attention_failures"] >= 2:
                st.session_state["failed_attention"] = True

        st.session_state["pending_rows"].append([prolific_id, file_id, score, timestamp])
        st.session_state["current_index"] += 1

        del st.session_state[score_key]
        st.rerun()

def llm_usage_screen():
    """Ask about LLM usage after all 15 comments."""
    st.title("Final question")
    st.markdown("Did you use a large language model (LLM), such as the one behind ChatGPT, to help you answer questions in this survey?")

    answer = st.radio("Select one:", ["No", "Yes"], index=None)
    if st.button("Submit") and answer is not None:
        prolific_id = st.session_state["prolific_id"]
        timestamp = datetime.now(UTC).isoformat()
        st.session_state["pending_rows"].append([prolific_id, "LLM_USAGE", answer, timestamp])
        st.session_state["llm_answered"] = True
        st.session_state["llm_used_yes"] = (answer == "Yes")

def failed_attention_screen():
    """Screen shown when a participant fails both attention checks."""
    flush_results()
    st.title("Survey ended")
    st.title("Thank you for participating!")
    st.write("Your responses have been recorded.")
    st.write(f"The completion code is: {COMPLETION_CODE_ATTENTION_FAIL}")

def score_label(score: int) -> str:
    labels = {
        1: "Strong opposition to climate action by the regulator. Explicitly resists climate measures. May deny climate change or climate risks.",
        2: "Skeptical or hesitant. Questions the need for special treatment or warns about costs and unintended consequences.",
        3: "Neutral. Takes no strong position for or against climate action.",
        4: "Supportive. Backs climate actions of the regulator. May support other climate measures. May advocate for more incremental steps.",
        5: "Strong advocate. Fully supports ambitious, binding climate targets and broad reforms. May seek to strengthen proposed initiatives.",
    }
    return labels[score]

def flush_results():
    """Append all pending rows to the results sheet in one API call."""
    rows = st.session_state.get("pending_rows", [])
    if not rows:
        return

    def _append():
        results_ws.append_rows(rows, value_input_option="RAW")

    _retry_on_quota(_append)
    st.session_state["pending_rows"] = []

def _update_order_status(new_status: str) -> None:
    """Update the status of the current participant's order row in the sheet."""
    order_id = st.session_state.get("order_id")
    if not order_id:
        return
    try:

        cell = order_ws.find(str(order_id))

        status_col = 6
        order_ws.update_cell(cell.row, status_col, new_status)
    except Exception:

        pass

def thank_you_screen():
    """Final thank-you screen after participant reaches the end of the survey."""
    flush_results()

    if st.session_state.get("attention_failures", 0) >= 2:
        code = COMPLETION_CODE_ATTENTION_FAIL
        status = STATUS_INVALID_ATTN
    elif st.session_state.get("llm_used_yes"):
        code = COMPLETION_CODE_LLM_YES
        status = STATUS_INVALID_LLM
    else:
        code = COMPLETION_CODE_SUCCESS
        status = STATUS_VALID

    _update_order_status(status)

    st.title("Thank you for participating!")
    st.write("Your responses have been recorded.")
    st.write(f"The completion code is: {code}")

if __name__ == "__main__":
    main()
