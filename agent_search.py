import streamlit as st
import openai
import requests
import os
import json
import csv
import io
from typing import Dict, List, Optional
from datetime import datetime
import re


# ==================== UX Monitoring & Logging Module ====================

MONITOR_LOG_KEY = "ux_monitor_log"
MONITOR_ACTIVE_KEY = "ux_monitor_active"
MONITOR_START_TIME_KEY = "ux_monitor_start_time"
MONITOR_SESSION_ID_KEY = "ux_monitor_session_id"
MONITOR_COUNTER_KEY = "ux_monitor_event_counter"


def init_monitor():
    """Initialize the UX monitoring system."""
    if MONITOR_LOG_KEY not in st.session_state:
        st.session_state[MONITOR_LOG_KEY] = []
    if MONITOR_ACTIVE_KEY not in st.session_state:
        st.session_state[MONITOR_ACTIVE_KEY] = False
    if MONITOR_START_TIME_KEY not in st.session_state:
        st.session_state[MONITOR_START_TIME_KEY] = None
    if MONITOR_SESSION_ID_KEY not in st.session_state:
        st.session_state[MONITOR_SESSION_ID_KEY] = None
    if MONITOR_COUNTER_KEY not in st.session_state:
        st.session_state[MONITOR_COUNTER_KEY] = 0


def start_monitoring():
    """Start a new monitoring session."""
    init_monitor()
    now = datetime.now()
    session_id = now.strftime("%Y%m%d_%H%M%S")
    st.session_state[MONITOR_ACTIVE_KEY] = True
    st.session_state[MONITOR_START_TIME_KEY] = now.isoformat()
    st.session_state[MONITOR_SESSION_ID_KEY] = session_id
    st.session_state[MONITOR_LOG_KEY] = []
    st.session_state[MONITOR_COUNTER_KEY] = 0
    log_operation("MONITOR_START", {"session_id": session_id})


def stop_monitoring():
    """Stop the current monitoring session."""
    log_operation("MONITOR_END", {
        "session_id": st.session_state.get(MONITOR_SESSION_ID_KEY, ""),
        "total_events": st.session_state.get(MONITOR_COUNTER_KEY, 0),
    })
    st.session_state[MONITOR_ACTIVE_KEY] = False


def is_monitoring() -> bool:
    """Check if monitoring is currently active."""
    return st.session_state.get(MONITOR_ACTIVE_KEY, False)


def log_operation(operation: str, details: Optional[Dict] = None):
    """
    Record an operation with a timestamp.

    Parameters
    ----------
    operation : str
        A short, machine-readable label for the operation, e.g.
        ``SEARCH_INITIATED``, ``PAPER_ADDED_TO_FOLDER``.
    details : dict, optional
        Arbitrary key/value pairs providing context.
    """
    init_monitor()

    # Always log if monitoring is active, or if it's the start/end event itself
    if not is_monitoring() and operation not in ("MONITOR_START", "MONITOR_END"):
        return

    now = datetime.now()
    st.session_state[MONITOR_COUNTER_KEY] = st.session_state.get(MONITOR_COUNTER_KEY, 0) + 1

    elapsed = ""
    start_str = st.session_state.get(MONITOR_START_TIME_KEY)
    if start_str:
        try:
            start_dt = datetime.fromisoformat(start_str)
            delta = now - start_dt
            elapsed = f"{delta.total_seconds():.2f}s"
        except Exception:
            elapsed = ""

    entry = {
        "event_number": st.session_state[MONITOR_COUNTER_KEY],
        "timestamp": now.isoformat(),
        "elapsed": elapsed,
        "operation": operation,
        "details": details or {},
    }
    st.session_state[MONITOR_LOG_KEY].append(entry)


def get_log_entries() -> List[Dict]:
    """Return all recorded log entries."""
    init_monitor()
    return st.session_state[MONITOR_LOG_KEY]


def export_log_as_json() -> str:
    """Export the full log as a JSON string."""
    entries = get_log_entries()
    export = {
        "session_id": st.session_state.get(MONITOR_SESSION_ID_KEY, ""),
        "start_time": st.session_state.get(MONITOR_START_TIME_KEY, ""),
        "total_events": len(entries),
        "events": entries,
    }
    return json.dumps(export, indent=2, ensure_ascii=False)


def export_log_as_csv() -> str:
    """Export the full log as a CSV string."""
    entries = get_log_entries()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["event_number", "timestamp", "elapsed", "operation", "details"])
    for e in entries:
        writer.writerow([
            e["event_number"],
            e["timestamp"],
            e["elapsed"],
            e["operation"],
            json.dumps(e["details"], ensure_ascii=False),
        ])
    return output.getvalue()


def render_monitor_panel():
    """Render the monitoring control panel in the sidebar."""
    init_monitor()

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔴 UX Monitor & Logger")

    if is_monitoring():
        st.sidebar.success("● Recording in progress…")
        event_count = st.session_state.get(MONITOR_COUNTER_KEY, 0)
        start_str = st.session_state.get(MONITOR_START_TIME_KEY, "")
        if start_str:
            try:
                start_dt = datetime.fromisoformat(start_str)
                elapsed = datetime.now() - start_dt
                mins, secs = divmod(int(elapsed.total_seconds()), 60)
                st.sidebar.caption(f"⏱ Elapsed: {mins}m {secs}s  |  Events: {event_count}")
            except Exception:
                st.sidebar.caption(f"Events: {event_count}")

        if st.sidebar.button("⏹ End Monitoring", use_container_width=True, type="primary"):
            stop_monitoring()
            st.rerun()
    else:
        entries = get_log_entries()
        if entries:
            st.sidebar.info(f"Last session: {len(entries)} events recorded")
        else:
            st.sidebar.caption("Click **Start** to begin recording user operations.")

        if st.sidebar.button("▶️ Start Monitoring", use_container_width=True, type="primary"):
            start_monitoring()
            st.rerun()

    # Log viewer & export (always available if there are entries)
    entries = get_log_entries()
    if entries:
        with st.sidebar.expander(f"📋 View Log ({len(entries)} events)", expanded=False):
            # Show last 10 events in reverse order
            for e in reversed(entries[-10:]):
                st.caption(
                    f"**#{e['event_number']}** `{e['elapsed']}` — "
                    f"**{e['operation']}**  \n"
                    f"{json.dumps(e['details'], ensure_ascii=False)[:120]}"
                )
            if len(entries) > 10:
                st.caption(f"… and {len(entries) - 10} more events")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.download_button(
                "⬇ JSON",
                data=export_log_as_json(),
                file_name=f"ux_log_{st.session_state.get(MONITOR_SESSION_ID_KEY, 'session')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "⬇ CSV",
                data=export_log_as_csv(),
                file_name=f"ux_log_{st.session_state.get(MONITOR_SESSION_ID_KEY, 'session')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if st.sidebar.button("🗑 Clear Log", use_container_width=True):
            st.session_state[MONITOR_LOG_KEY] = []
            st.session_state[MONITOR_COUNTER_KEY] = 0
            st.session_state[MONITOR_SESSION_ID_KEY] = None
            st.session_state[MONITOR_START_TIME_KEY] = None
            st.rerun()


# ==================== Helper: resolve POE API key ====================

def get_poe_api_key() -> str:
    """Return the POE API key from session_state (user input) or env var."""
    return (
        st.session_state.get("poe_api_key_input", "").strip()
        or os.getenv("POE_API_KEY", "")
    )


def init_poe_client():
    api_key = get_poe_api_key()
    if not api_key:
        return None
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://api.poe.com/v1",
    )


SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# Paper management storage keys
PAPERS_DB_KEY = "papers_database"
FOLDERS_DB_KEY = "folders_database"
FOLDER_PAPERS_KEY = "folder_papers"

# Semantic Scholar fields of study
FIELDS_OF_STUDY = [
    "Computer Science",
    "Medicine",
    "Chemistry",
    "Biology",
    "Materials Science",
    "Physics",
    "Geology",
    "Psychology",
    "Art",
    "History",
    "Geography",
    "Sociology",
    "Business",
    "Political Science",
    "Economics",
    "Philosophy",
    "Mathematics",
    "Engineering",
    "Environmental Science",
    "Agricultural and Food Sciences",
    "Education",
    "Law",
    "Linguistics"
]

# Semantic Scholar publication types
PUBLICATION_TYPES = [
    "Review",
    "JournalArticle",
    "CaseReport",
    "ClinicalTrial",
    "Dataset",
    "Editorial",
    "LettersAndComments",
    "MetaAnalysis",
    "News",
    "Study",
    "Book",
    "BookSection",
    "Conference"
]

AVAILABLE_FIELDS = {
    "basic": ["paperId", "title", "url"],
    "publication": ["publicationDate", "publicationTypes", "venue", "year"],
    "metrics": ["citationCount", "influentialCitationCount"],
    "content": ["abstract"],
    "people": ["authors", "firstAuthor"],
    "references": ["references", "citations"],
    "access": ["openAccessPdf", "isOpenAccess"]
}


def get_field_string(include_advanced: bool = False) -> str:
    """Build field string for API requests"""
    fields = [
        "paperId",
        "title",
        "url",
        "publicationDate",
        "citationCount",
        "authors",
        "abstract",
        "venue",
        "year",
        "publicationTypes",
        "openAccessPdf"
    ]

    if include_advanced:
        fields.extend([
            "influentialCitationCount",
            "isOpenAccess"
        ])

    return ",".join(fields)


def search_papers(
    query: str,
    year_range: str = "2023-",
    limit: int = 10,
    sort_by: str = "citationCount",
    token: Optional[str] = None,
    fields_of_study: Optional[List[str]] = None,
    min_citation_count: Optional[int] = None,
    publication_types: Optional[List[str]] = None,
    open_access_only: bool = False,
    venue: Optional[str] = None
) -> Dict:
    """Search papers using Semantic Scholar Academic Graph API bulk search endpoint"""
    log_operation("API_SEARCH_REQUEST", {
        "query": query,
        "year_range": year_range,
        "limit": limit,
        "sort_by": sort_by,
        "fields_of_study": fields_of_study,
        "min_citation_count": min_citation_count,
        "publication_types": publication_types,
        "open_access_only": open_access_only,
        "venue": venue,
    })
    try:
        url = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search/bulk"

        params = {
            "query": query,
            "fields": get_field_string(include_advanced=True),
            "limit": min(limit, 100),
            "sort": sort_by
        }

        if year_range and year_range != "all":
            params["year"] = year_range

        if token:
            params["token"] = token

        # Additional filters
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        if min_citation_count is not None and min_citation_count > 0:
            params["minCitationCount"] = min_citation_count

        if publication_types:
            params["publicationTypes"] = ",".join(publication_types)

        if open_access_only:
            params["openAccessPdf"] = ""

        if venue and venue.strip():
            params["venue"] = venue.strip()

        headers = {}
        if SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

        response = requests.get(url, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            result = response.json()
            log_operation("API_SEARCH_SUCCESS", {
                "papers_returned": len(result.get("data", [])),
            })
            return result
        elif response.status_code == 400:
            log_operation("API_SEARCH_ERROR", {"status": 400, "detail": response.text[:200]})
            return {"error": f"Request format error: {response.text}"}
        elif response.status_code == 401:
            log_operation("API_SEARCH_ERROR", {"status": 401})
            return {"error": "Invalid or missing API key"}
        elif response.status_code == 429:
            log_operation("API_SEARCH_ERROR", {"status": 429})
            return {"error": "Too many requests, please try again later"}
        else:
            log_operation("API_SEARCH_ERROR", {"status": response.status_code})
            return {"error": f"API request failed: {response.status_code}"}

    except requests.exceptions.Timeout:
        log_operation("API_SEARCH_ERROR", {"detail": "timeout"})
        return {"error": "Request timed out, please check network connection"}
    except Exception as e:
        log_operation("API_SEARCH_ERROR", {"detail": str(e)[:200]})
        return {"error": str(e)}


def get_paper_details(paper_id: str, include_references: bool = False) -> Dict:
    """Get paper details using Academic Graph API paper details endpoint"""
    log_operation("API_PAPER_DETAILS_REQUEST", {
        "paper_id": paper_id,
        "include_references": include_references,
    })
    try:
        url = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/{paper_id}"

        fields = [
            "paperId",
            "title",
            "abstract",
            "citationCount",
            "influentialCitationCount",
            "authors",
            "year",
            "venue",
            "publicationDate",
            "publicationTypes",
            "openAccessPdf",
            "isOpenAccess",
            "tldr",
            "url"
        ]

        if include_references:
            fields.extend(["references", "citations"])

        params = {
            "fields": ",".join(fields)
        }

        headers = {}
        if SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

        response = requests.get(url, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            log_operation("API_PAPER_DETAILS_SUCCESS", {"paper_id": paper_id})
            return response.json()
        elif response.status_code == 404:
            log_operation("API_PAPER_DETAILS_ERROR", {"paper_id": paper_id, "status": 404})
            return {"error": "Paper not found"}
        elif response.status_code == 401:
            log_operation("API_PAPER_DETAILS_ERROR", {"paper_id": paper_id, "status": 401})
            return {"error": "Invalid API key"}
        else:
            log_operation("API_PAPER_DETAILS_ERROR", {"paper_id": paper_id, "status": response.status_code})
            return {"error": f"Failed to get paper details: {response.status_code}"}

    except requests.exceptions.Timeout:
        log_operation("API_PAPER_DETAILS_ERROR", {"paper_id": paper_id, "detail": "timeout"})
        return {"error": "Request timed out"}
    except Exception as e:
        log_operation("API_PAPER_DETAILS_ERROR", {"paper_id": paper_id, "detail": str(e)[:200]})
        return {"error": str(e)}


def search_author(author_id: str) -> Dict:
    """Query author information using Academic Graph API"""
    log_operation("API_AUTHOR_SEARCH", {"author_id": author_id})
    try:
        url = f"{SEMANTIC_SCHOLAR_BASE_URL}/author/{author_id}"

        params = {
            "fields": "authorId,name,url,paperCount,citationCount,hIndex,papers"
        }

        headers = {}
        if SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

        response = requests.get(url, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to get author information: {response.status_code}"}

    except Exception as e:
        return {"error": str(e)}


# ==================== Paper Management Features ====================

def init_paper_management():
    """Initialize paper management system"""
    if PAPERS_DB_KEY not in st.session_state:
        st.session_state[PAPERS_DB_KEY] = {}
    if FOLDERS_DB_KEY not in st.session_state:
        st.session_state[FOLDERS_DB_KEY] = {}
    if FOLDER_PAPERS_KEY not in st.session_state:
        st.session_state[FOLDER_PAPERS_KEY] = {}


def create_folder(folder_name: str, description: str = "") -> bool:
    """Create a new folder"""
    init_paper_management()

    if folder_name in st.session_state[FOLDERS_DB_KEY]:
        log_operation("FOLDER_CREATE_DUPLICATE", {"folder_name": folder_name})
        return False

    st.session_state[FOLDERS_DB_KEY][folder_name] = {
        "created_at": datetime.now().isoformat(),
        "description": description,
        "paper_count": 0
    }
    st.session_state[FOLDER_PAPERS_KEY][folder_name] = []
    log_operation("FOLDER_CREATED", {"folder_name": folder_name, "description": description})
    return True


def delete_folder(folder_name: str) -> bool:
    """Delete a folder"""
    init_paper_management()

    if folder_name not in st.session_state[FOLDERS_DB_KEY]:
        return False

    paper_count = st.session_state[FOLDERS_DB_KEY][folder_name].get("paper_count", 0)
    del st.session_state[FOLDERS_DB_KEY][folder_name]
    if folder_name in st.session_state[FOLDER_PAPERS_KEY]:
        del st.session_state[FOLDER_PAPERS_KEY][folder_name]
    log_operation("FOLDER_DELETED", {"folder_name": folder_name, "papers_removed": paper_count})
    return True


def add_paper_to_folder(folder_name: str, paper: Dict) -> bool:
    """Add paper to folder"""
    init_paper_management()

    if folder_name not in st.session_state[FOLDERS_DB_KEY]:
        return False

    paper_id = paper.get('paperId', 'unknown')

    if folder_name not in st.session_state[FOLDER_PAPERS_KEY]:
        st.session_state[FOLDER_PAPERS_KEY][folder_name] = []

    existing_ids = [p.get('paperId') for p in st.session_state[FOLDER_PAPERS_KEY][folder_name]]
    if paper_id in existing_ids:
        log_operation("PAPER_ADD_DUPLICATE", {
            "folder": folder_name,
            "paper_id": paper_id,
            "title": paper.get("title", "")[:80],
        })
        return False

    st.session_state[FOLDER_PAPERS_KEY][folder_name].append(paper)
    st.session_state[FOLDERS_DB_KEY][folder_name]["paper_count"] = len(st.session_state[FOLDER_PAPERS_KEY][folder_name])
    log_operation("PAPER_ADDED_TO_FOLDER", {
        "folder": folder_name,
        "paper_id": paper_id,
        "title": paper.get("title", "")[:80],
    })
    return True


def remove_paper_from_folder(folder_name: str, paper_id: str) -> bool:
    """Remove paper from folder"""
    init_paper_management()

    if folder_name not in st.session_state[FOLDER_PAPERS_KEY]:
        return False

    papers = st.session_state[FOLDER_PAPERS_KEY][folder_name]
    papers_before = len(papers)
    papers[:] = [p for p in papers if p.get('paperId') != paper_id]

    if len(papers) < papers_before:
        st.session_state[FOLDERS_DB_KEY][folder_name]["paper_count"] = len(papers)
        log_operation("PAPER_REMOVED_FROM_FOLDER", {
            "folder": folder_name,
            "paper_id": paper_id,
        })
        return True
    return False


def get_folder_papers(folder_name: str) -> List[Dict]:
    """Get all papers in a folder"""
    init_paper_management()

    if folder_name not in st.session_state[FOLDER_PAPERS_KEY]:
        return []

    return st.session_state[FOLDER_PAPERS_KEY][folder_name]


def get_all_folders() -> List[str]:
    """Get all folders"""
    init_paper_management()
    return list(st.session_state[FOLDERS_DB_KEY].keys())


def export_folder_as_json(folder_name: str) -> str:
    """Export folder as JSON format"""
    init_paper_management()

    if folder_name not in st.session_state[FOLDERS_DB_KEY]:
        return ""

    papers = get_folder_papers(folder_name)
    export_data = {
        "folder_name": folder_name,
        "folder_info": st.session_state[FOLDERS_DB_KEY][folder_name],
        "papers": papers
    }

    log_operation("FOLDER_EXPORTED_JSON", {"folder": folder_name, "paper_count": len(papers)})
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def ask_ai_about_folder(client: openai.OpenAI, folder_name: str, user_question: str) -> str:
    """Ask AI assistant about folder contents"""
    log_operation("AI_FOLDER_QUESTION", {"folder": folder_name, "question": user_question[:120]})
    papers = get_folder_papers(folder_name)

    if not papers:
        return "No papers in this folder."

    papers_context = f"Papers in folder '{folder_name}':\n\n"
    for i, paper in enumerate(papers, 1):
        papers_context += f"{i}. **{paper.get('title', 'N/A')}**\n"
        papers_context += f"   Authors: {', '.join([a.get('name', 'N/A') for a in paper.get('authors', [])[:3]])}\n"
        papers_context += f"   Year: {paper.get('year', 'N/A')}\n"
        papers_context += f"   Citation count: {paper.get('citationCount', 0)}\n"

        if paper.get('abstract'):
            abstract_preview = paper.get('abstract', '')[:200]
            papers_context += f"   Abstract: {abstract_preview}...\n"

        papers_context += "\n"

    prompt = f"""You are a research expert. The user has a collection of papers and wants to get information from this collection.

Papers collection information:
{papers_context}

User question: {user_question}

Please answer the user's question based on the provided paper information. If the question involves specific data analysis or deep understanding, please provide comprehensive analysis based on all papers."""

    response = client.chat.completions.create(
        model="claude-haiku-4.5",
        messages=[{"role": "user", "content": prompt}]
    )

    log_operation("AI_FOLDER_ANSWER_RECEIVED", {"folder": folder_name})
    return response.choices[0].message.content


def generate_search_plan(client: openai.OpenAI, user_query: str) -> str:
    """Generate search plan using AI Agent"""
    log_operation("AI_SEARCH_PLAN_REQUEST", {"query": user_query[:120]})
    prompt = f"""You are an AI assistant specialized in generating search plans for academic literature research.

User query: {user_query}

Please analyze this query and generate a detailed search plan. Include:
1. Core keywords (enclosed in quotes for exact matching)
2. Related keywords and synonyms
3. Recommended year range for the search
4. Expected number of search results
5. Recommended sorting metric (citationCount - citation count, publicationDate - publication date, paperId - relevance)
6. Explanation of search strategy

Please return the search plan in a structured format."""

    response = client.chat.completions.create(
        model="claude-haiku-4.5",
        messages=[{"role": "user", "content": prompt}]
    )

    log_operation("AI_SEARCH_PLAN_RECEIVED", {"query": user_query[:120]})
    return response.choices[0].message.content


def extract_search_keyword(client: openai.OpenAI, search_plan: str, user_query: str) -> tuple:
    """Extract keywords and sorting method from search plan"""
    log_operation("AI_KEYWORD_EXTRACTION_REQUEST", {"query": user_query[:120]})
    extraction_prompt = f"""Extract the following from the search plan:
1. The most relevant search keywords (enclosed in quotes for exact matching), return only 1-2 most important keywords
2. Recommended sorting method (one of: citationCount, publicationDate, or paperId)

Search plan:
{search_plan}

Return format (one per line):
Keyword: [your keyword]
Sort: [sorting method]

Do not add any other text."""

    response = client.chat.completions.create(
        model="claude-haiku-4.5",
        messages=[{"role": "user", "content": extraction_prompt}]
    )

    result = response.choices[0].message.content.strip()

    keyword = ""
    sort_by = "citationCount"

    for line in result.split('\n'):
        if 'Keyword:' in line:
            keyword = line.replace('Keyword:', '').strip()
        elif 'Sort:' in line:
            sort_method = line.replace('Sort:', '').strip()
            sort_method_lower = sort_method.lower()
            if 'citation' in sort_method_lower:
                sort_by = 'citationCount'
            elif 'publication' in sort_method_lower or 'date' in sort_method_lower:
                sort_by = 'publicationDate'
            elif 'paper' in sort_method_lower or 'id' in sort_method_lower or 'relevance' in sort_method_lower:
                sort_by = 'paperId'
            else:
                sort_by = 'citationCount'

    log_operation("AI_KEYWORD_EXTRACTED", {"keyword": keyword, "sort_by": sort_by})
    return keyword, sort_by


def filter_and_rerank_papers(client: openai.OpenAI, user_query: str, papers: List[Dict], limit: int) -> List[Dict]:
    """Use AI to filter and rerank search results"""
    if not papers:
        return []

    log_operation("AI_FILTER_RERANK_REQUEST", {
        "query": user_query[:120],
        "input_count": len(papers),
        "limit": limit,
    })

    papers_summary = "Papers to analyze:\n\n"
    for i, paper in enumerate(papers[:20], 1):
        papers_summary += f"{i}. Title: {paper.get('title', 'N/A')}\n"
        papers_summary += f"   Abstract: {(paper.get('abstract', 'N/A')[:200] + '...') if paper.get('abstract') else 'N/A'}\n"
        papers_summary += f"   Year: {paper.get('year', 'N/A')}\n"
        papers_summary += f"   Citation Count: {paper.get('citationCount', 0)}\n"
        papers_summary += f"   Venue: {paper.get('venue', 'N/A')}\n\n"

    filter_prompt = f"""You are a research expert. Given the user's research query and a list of papers, please:

1. Analyze the relevance of each paper to the user's query
2. Filter out papers that are NOT directly relevant
3. Rerank the remaining papers by relevance (most relevant first)
4. Consider factors: relevance to query, citation impact, publication year, venue quality

User Query: {user_query}

{papers_summary}

IMPORTANT: Return ONLY a JSON array with the indices of papers to keep (in the new order), like:
[3, 1, 5, 2, ...]

Do NOT include any other text before or after the JSON array. Only return the JSON array."""

    try:
        response = client.chat.completions.create(
            model="claude-haiku-4.5",
            messages=[{"role": "user", "content": filter_prompt}]
        )

        result_text = response.choices[0].message.content.strip()

        json_match = re.search(r'\[\s*[\d\s,]*\]', result_text)
        if json_match:
            json_str = json_match.group(0)
            indices = json.loads(json_str)
        else:
            indices = json.loads(result_text)

        reranked_papers = []
        for idx in indices:
            if isinstance(idx, int) and 0 < idx <= len(papers):
                reranked_papers.append(papers[idx - 1])

        result = reranked_papers[:limit] if reranked_papers else papers[:limit]
        log_operation("AI_FILTER_RERANK_SUCCESS", {"output_count": len(result)})
        return result

    except json.JSONDecodeError as e:
        st.warning(f"AI filtering failed: {str(e)}, using original order")
        log_operation("AI_FILTER_RERANK_FAILED", {"error": str(e)[:100]})
        return papers[:limit]
    except Exception as e:
        st.warning(f"AI filtering error: {str(e)}, using original order")
        log_operation("AI_FILTER_RERANK_FAILED", {"error": str(e)[:100]})
        return papers[:limit]


def organize_results(client: openai.OpenAI, user_query: str, search_results: Dict) -> str:
    """Use AI Agent to organize and summarize search results"""
    log_operation("AI_SUMMARY_REQUEST", {"query": user_query[:120]})

    if "error" in search_results:
        results_text = f"Search error: {search_results['error']}"
    else:
        papers = search_results.get("data", [])
        if not papers:
            results_text = "No relevant papers found."
        else:
            results_text = f"Found {len(papers)} relevant papers:\n\n"
            for i, paper in enumerate(papers[:10], 1):
                results_text += f"{i}. Title: {paper.get('title', 'N/A')}\n"
                results_text += f"   Publication Year: {paper.get('year', paper.get('publicationDate', 'N/A'))}\n"
                results_text += f"   Citation Count: {paper.get('citationCount', 0)}\n"

                if paper.get('authors'):
                    authors = ', '.join([a.get('name', 'N/A') for a in paper.get('authors', [])[:3]])
                    results_text += f"   Authors: {authors}\n"

                if paper.get('abstract'):
                    abstract_preview = paper.get('abstract', '')[:150]
                    results_text += f"   Abstract: {abstract_preview}...\n"

                if paper.get('venue'):
                    results_text += f"   Published in: {paper.get('venue')}\n"

                results_text += "\n"

    organization_prompt = f"""Based on the following user query and search results, please generate a structured summary:

Original user query: {user_query}

Search results:
{results_text}

Please provide:
1. Search results overview (number of papers found, main research directions)
2. Most relevant papers (sorted by citation count, list top 3-5 with brief commentary)
3. Key research hotspots and trends
4. Recommendations for further reading directions
5. If the user wants to deepen the search, suggest potential keywords or search strategies

Please organize information in a clear and easy-to-read format."""

    response = client.chat.completions.create(
        model="claude-haiku-4.5",
        messages=[{"role": "user", "content": organization_prompt}]
    )

    log_operation("AI_SUMMARY_RECEIVED", {"query": user_query[:120]})
    return response.choices[0].message.content


def format_paper_display(paper: Dict, index: int) -> Dict:
    """Format paper display information"""
    title = paper.get('title', 'N/A')
    authors = ', '.join([a.get('name', 'N/A') for a in paper.get('authors', [])]) if paper.get('authors') else 'N/A'
    year = paper.get('year', paper.get('publicationDate', 'N/A'))
    citation_count = paper.get('citationCount', 0)
    influential_citations = paper.get('influentialCitationCount', 0)
    abstract = paper.get('abstract') or 'N/A'
    venue = paper.get('venue', 'N/A')
    url = paper.get('url', '')
    pdf_url = paper.get('openAccessPdf', {}).get('url', '') if paper.get('openAccessPdf') else ''
    is_open_access = paper.get('isOpenAccess', False)
    tldr = paper.get('tldr', '')

    return {
        'title': title,
        'authors': authors,
        'year': year,
        'citation_count': citation_count,
        'influential_citations': influential_citations,
        'abstract': abstract,
        'venue': venue,
        'url': url,
        'pdf_url': pdf_url,
        'is_open_access': is_open_access,
        'tldr': tldr
    }


# ==================== UI Components ====================

def render_search_filters() -> Dict:
    """Render search filter controls when AI-enhanced search is disabled."""

    filters = {}

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Search Filters")

    # Sort option
    sort_option = st.sidebar.selectbox(
        "Sort results by",
        ["citationCount", "publicationDate", "paperId"],
        index=0,
        format_func=lambda x: {
            "citationCount": "📊 Citation Count",
            "publicationDate": "📅 Publication Date",
            "paperId": "🔤 Relevance (Paper ID)"
        }.get(x, x),
        help="Choose how to sort results from the Semantic Scholar API"
    )
    filters["sort_by"] = sort_option

    # Fields of study
    selected_fields = st.sidebar.multiselect(
        "Fields of Study",
        FIELDS_OF_STUDY,
        default=[],
        help="Filter papers by academic discipline. Leave empty for all fields."
    )
    filters["fields_of_study"] = selected_fields if selected_fields else None

    # Minimum citation count
    min_citations = st.sidebar.number_input(
        "Minimum Citation Count",
        min_value=0,
        max_value=100000,
        value=0,
        step=10,
        help="Only return papers with at least this many citations"
    )
    filters["min_citation_count"] = min_citations if min_citations > 0 else None

    # Publication types
    selected_pub_types = st.sidebar.multiselect(
        "Publication Types",
        PUBLICATION_TYPES,
        default=[],
        format_func=lambda x: {
            "Review": "📝 Review",
            "JournalArticle": "📰 Journal Article",
            "CaseReport": "📋 Case Report",
            "ClinicalTrial": "🏥 Clinical Trial",
            "Dataset": "💾 Dataset",
            "Editorial": "✏️ Editorial",
            "LettersAndComments": "💬 Letters & Comments",
            "MetaAnalysis": "📊 Meta-Analysis",
            "News": "📢 News",
            "Study": "🔬 Study",
            "Book": "📚 Book",
            "BookSection": "📖 Book Section",
            "Conference": "🎤 Conference"
        }.get(x, x),
        help="Filter by type of publication. Leave empty for all types."
    )
    filters["publication_types"] = selected_pub_types if selected_pub_types else None

    # Open access only
    open_access_only = st.sidebar.checkbox(
        "🔓 Open Access Only",
        value=False,
        help="Only return papers that have a free PDF available"
    )
    filters["open_access_only"] = open_access_only

    # Venue filter
    venue_filter = st.sidebar.text_input(
        "Venue / Journal / Conference",
        value="",
        placeholder="e.g. Nature, NeurIPS, ICML",
        help="Filter by publication venue. Supports comma-separated values."
    )
    filters["venue"] = venue_filter if venue_filter.strip() else None

    # Display active filters summary
    active_filters = []
    if selected_fields:
        active_filters.append(f"Fields: {', '.join(selected_fields)}")
    if min_citations > 0:
        active_filters.append(f"Min citations: {min_citations}")
    if selected_pub_types:
        active_filters.append(f"Types: {', '.join(selected_pub_types)}")
    if open_access_only:
        active_filters.append("Open Access only")
    if venue_filter.strip():
        active_filters.append(f"Venue: {venue_filter}")

    if active_filters:
        st.sidebar.markdown("**Active filters:**")
        for f in active_filters:
            st.sidebar.caption(f"✅ {f}")
    else:
        st.sidebar.caption("No additional filters applied")

    log_operation("FILTERS_CONFIGURED", {k: v for k, v in filters.items() if v})
    return filters


def render_folder_management_panel():
    """Render paper library management panel"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Paper Library Management")

    folders = get_all_folders()

    with st.sidebar.expander("➕ Create New Folder", expanded=False):
        folder_name = st.text_input("Folder name", key="new_folder_name")
        folder_desc = st.text_area("Folder description (optional)", key="new_folder_desc", height=80)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create", key="create_folder_btn"):
                if folder_name.strip():
                    if create_folder(folder_name.strip(), folder_desc.strip()):
                        st.success(f"✅ Folder '{folder_name}' created successfully")
                        st.rerun()
                    else:
                        st.error(f"❌ Folder '{folder_name}' already exists")
                else:
                    st.error("❌ Folder name cannot be empty")

    if folders:
        st.sidebar.markdown("**Existing folders:**")
        for folder in folders:
            folder_info = st.session_state[FOLDERS_DB_KEY][folder]
            papers_count = folder_info.get("paper_count", 0)

            col1, col2 = st.columns([3, 1], gap="small")
            with col1:
                if st.button(f"📁 {folder} ({papers_count})", key=f"select_folder_{folder}", use_container_width=True):
                    st.session_state["selected_folder"] = folder
                    st.session_state["show_folder_view"] = True
                    log_operation("FOLDER_OPENED", {"folder": folder, "paper_count": papers_count})
            with col2:
                if st.button("🗑️", key=f"delete_folder_{folder}", help="Delete folder"):
                    delete_folder(folder)
                    if st.session_state.get("selected_folder") == folder:
                        st.session_state["selected_folder"] = None
                    st.success("✅ Folder deleted")
                    st.rerun()
    else:
        st.sidebar.info("📭 No folders created yet")


def render_folder_view():
    """Render folder view interface"""
    st.divider()

    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            log_operation("FOLDER_VIEW_CLOSED", {"folder": st.session_state.get("selected_folder", "")})
            st.session_state["show_folder_view"] = False
            st.session_state["selected_folder"] = None
            st.rerun()
    with col2:
        st.subheader("📁 Folder Details")

    selected_folder = st.session_state.get("selected_folder")
    if not selected_folder:
        st.info("Please select a folder from the sidebar")
        return

    folder_info = st.session_state[FOLDERS_DB_KEY].get(selected_folder)
    papers = get_folder_papers(selected_folder)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Papers", len(papers))
    with col2:
        created_date = folder_info.get("created_at", "N/A")
        if created_date != "N/A":
            created_date = created_date.split('T')[0]
        st.metric("Created Date", created_date)
    with col3:
        st.metric("Description", folder_info.get("description", "None") if folder_info.get("description") else "None")

    st.markdown("---")

    with st.expander("🤖 AI Assistant (Ask questions about papers in this folder)", expanded=False):
        poe_key = get_poe_api_key()
        if not poe_key:
            st.warning("⚠️ POE API Key not configured. Please enter your key in the sidebar to use AI features.")
        else:
            try:
                client = init_poe_client()

                col1, col2 = st.columns([4, 1])
                with col1:
                    user_question = st.text_input("Ask a question", key="ai_folder_question")
                with col2:
                    ask_button = st.button("Ask", key="ask_ai_btn")

                if ask_button and user_question:
                    with st.spinner("AI is thinking..."):
                        ai_response = ask_ai_about_folder(client, selected_folder, user_question)
                        st.markdown(ai_response)

            except Exception as e:
                st.error(f"AI feature error: {str(e)}")

    st.markdown("---")

    if papers:
        st.subheader(f"📚 Papers ({len(papers)} papers)")

        for i, paper in enumerate(papers, 1):
            paper_info = format_paper_display(paper, i)

            expander_title = f"{'🔓' if paper_info['is_open_access'] else '🔒'} {i}. {paper_info['title'][:60]}"
            if paper_info['citation_count'] > 100:
                expander_title += " ⭐"

            with st.expander(expander_title, expanded=(i == 1)):
                log_operation("PAPER_VIEWED_IN_FOLDER", {
                    "folder": selected_folder,
                    "paper_index": i,
                    "paper_id": paper.get("paperId", ""),
                    "title": paper_info["title"][:80],
                })
                st.markdown(f"**Full Title:** {paper_info['title']}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Citation Count", paper_info['citation_count'])
                with col2:
                    st.metric("Influential Citations", paper_info['influential_citations'])
                with col3:
                    st.metric("Publication Year", paper_info['year'])

                st.markdown(f"**Authors:** {paper_info['authors']}")
                st.markdown(f"**Published in:** {paper_info['venue']}")

                if paper_info['abstract'] and paper_info['abstract'] != 'N/A':
                    st.markdown("**Abstract:**")
                    abstract_text = paper_info['abstract']
                    display_text = abstract_text[:500] + "..." if len(abstract_text) > 500 else abstract_text
                    st.markdown(display_text)

                col1, col2, col3 = st.columns(3)
                with col1:
                    if paper_info['url']:
                        st.markdown(f"[📖 View on Semantic Scholar]({paper_info['url']})")
                with col2:
                    if paper_info['pdf_url']:
                        st.markdown(f"[📥 Download Open Access PDF]({paper_info['pdf_url']})")
                with col3:
                    if st.button("🗑️ Remove from folder", key=f"remove_paper_{i}_{paper.get('paperId')}"):
                        remove_paper_from_folder(selected_folder, paper.get('paperId'))
                        st.success("✅ Paper removed from folder")
                        st.rerun()

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export as JSON", use_container_width=True):
                json_data = export_folder_as_json(selected_folder)
                st.download_button(
                    label="Download JSON file",
                    data=json_data,
                    file_name=f"{selected_folder}_papers.json",
                    mime="application/json",
                    use_container_width=True
                )

        with col2:
            if st.button("📋 Generate Paper List", use_container_width=True):
                log_operation("FOLDER_EXPORTED_MARKDOWN", {"folder": selected_folder, "paper_count": len(papers)})
                markdown_text = f"# {selected_folder}\n\n"
                markdown_text += f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                markdown_text += f"Total papers: {len(papers)}\n\n"

                for i, paper in enumerate(papers, 1):
                    markdown_text += f"## {i}. {paper.get('title', 'N/A')}\n\n"
                    markdown_text += f"**Authors:** {', '.join([a.get('name', 'N/A') for a in paper.get('authors', [])]) if paper.get('authors') else 'N/A'}\n\n"
                    markdown_text += f"**Year:** {paper.get('year', 'N/A')}\n\n"
                    markdown_text += f"**Published in:** {paper.get('venue', 'N/A')}\n\n"
                    markdown_text += f"**Citation Count:** {paper.get('citationCount', 0)}\n\n"

                    if paper.get('abstract'):
                        markdown_text += f"**Abstract:** {paper.get('abstract')}\n\n"

                    if paper.get('url'):
                        markdown_text += f"[View Paper]({paper.get('url')})\n\n"

                    markdown_text += "---\n\n"

                st.download_button(
                    label="Download Markdown file",
                    data=markdown_text,
                    file_name=f"{selected_folder}_papers.md",
                    mime="text/markdown",
                    use_container_width=True
                )
    else:
        st.info("📭 No papers in this folder yet, add papers after searching")


def display_search_results(filtered_papers: List[Dict], include_references: bool = False):
    """Display search results with individual add to folder buttons"""
    if not filtered_papers:
        st.warning("No results to display")
        return

    folders = get_all_folders()
    has_folders = len(folders) > 0

    st.success(f"✅ Showing **{len(filtered_papers)}** papers")
    log_operation("SEARCH_RESULTS_DISPLAYED", {"count": len(filtered_papers)})

    for i, paper in enumerate(filtered_papers, 1):
        paper_info = format_paper_display(paper, i)

        expander_title = f"{'🔓' if paper_info['is_open_access'] else '🔒'} {i}. {paper_info['title'][:70]}"
        if paper_info['citation_count'] > 100:
            expander_title += " ⭐"

        with st.expander(expander_title, expanded=(i == 1)):
            log_operation("PAPER_EXPANDED", {
                "paper_index": i,
                "paper_id": paper.get("paperId", ""),
                "title": paper_info["title"][:80],
            })
            st.markdown(f"**Full Title:** {paper_info['title']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Citation Count", paper_info['citation_count'])
            with col2:
                st.metric("Influential Citations", paper_info['influential_citations'])
            with col3:
                st.metric("Publication Year", paper_info['year'])

            st.markdown(f"**Authors:** {paper_info['authors']}")
            st.markdown(f"**Published in:** {paper_info['venue']}")

            if paper_info['abstract'] and paper_info['abstract'] != 'N/A':
                st.markdown("**Abstract:**")
                abstract_text = paper_info['abstract']
                display_text = abstract_text[:500] + "..." if len(abstract_text) > 500 else abstract_text
                st.markdown(display_text)

            if paper_info['tldr']:
                st.info(f"💡 **Key Point Summary:** {paper_info['tldr']}")

            with st.expander("Get Key Point Summary (TLDR)"):
                log_operation("TLDR_REQUESTED", {"paper_id": paper.get("paperId", "")})
                with st.spinner("Retrieving key point summary..."):
                    detailed_info = get_paper_details(paper.get('paperId'))
                    if "error" not in detailed_info and detailed_info.get('tldr'):
                        st.info(f"💡 **Key Point Summary:** {detailed_info['tldr']}")
                    else:
                        st.info("TLDR not available for this paper")

            col1, col2 = st.columns(2)
            with col1:
                if paper_info['url']:
                    st.markdown(f"[📖 View on Semantic Scholar]({paper_info['url']})")
            with col2:
                if paper_info['pdf_url']:
                    st.markdown(f"[📥 Download Open Access PDF]({paper_info['pdf_url']})")
                elif paper_info['is_open_access']:
                    st.markdown("✅ Open access paper")

            if include_references:
                with st.expander("📖 View Citation Information"):
                    log_operation("CITATIONS_REQUESTED", {"paper_id": paper.get("paperId", "")})
                    with st.spinner("Retrieving citation and reference information..."):
                        detailed_info = get_paper_details(paper.get('paperId'), include_references=True)

                        if "error" not in detailed_info:
                            if detailed_info.get('references'):
                                st.markdown("**Number of References:** " + str(len(detailed_info.get('references', []))))
                            if detailed_info.get('citations'):
                                st.markdown("**Number of Citations:** " + str(len(detailed_info.get('citations', []))))
                        else:
                            st.warning(f"Unable to retrieve citation information: {detailed_info['error']}")

            st.markdown("---")
            st.markdown("**Add to Folder:**")

            if has_folders:
                col1, col2 = st.columns([3, 1])

                with col1:
                    selected_folder = st.selectbox(
                        "Choose folder",
                        folders,
                        key=f"folder_select_{i}_{paper.get('paperId')}",
                        label_visibility="collapsed"
                    )

                with col2:
                    if st.button("➕ Add", key=f"add_paper_{i}_{paper.get('paperId')}", use_container_width=True):
                        if add_paper_to_folder(selected_folder, paper):
                            st.success(f"✅ Added to '{selected_folder}'")
                        else:
                            st.info(f"ℹ️ Paper already in '{selected_folder}'")
            else:
                st.warning("⚠️ Please create a folder first (in the left sidebar)")


# ==================== Full-page Log Viewer ====================

def render_full_log_viewer():
    """Render a dedicated full-page log viewer."""
    st.subheader("📋 UX Session Log Viewer")

    entries = get_log_entries()
    if not entries:
        st.info("No log entries yet. Start monitoring and perform some actions to see events here.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", len(entries))
    with col2:
        session_id = st.session_state.get(MONITOR_SESSION_ID_KEY, "N/A")
        st.metric("Session ID", session_id)
    with col3:
        start_time = st.session_state.get(MONITOR_START_TIME_KEY, "")
        if start_time:
            st.metric("Start Time", start_time.split("T")[1][:8] if "T" in start_time else start_time[:19])
        else:
            st.metric("Start Time", "N/A")
    with col4:
        if entries:
            last_elapsed = entries[-1].get("elapsed", "N/A")
            st.metric("Last Event At", last_elapsed)

    st.markdown("---")

    # Filter by operation type
    all_ops = sorted(set(e["operation"] for e in entries))
    selected_ops = st.multiselect("Filter by operation type", all_ops, default=all_ops)

    filtered = [e for e in entries if e["operation"] in selected_ops]

    # Display as a table
    if filtered:
        table_data = []
        for e in filtered:
            details_str = json.dumps(e["details"], ensure_ascii=False) if e["details"] else ""
            table_data.append({
                "#": e["event_number"],
                "Elapsed": e["elapsed"],
                "Timestamp": e["timestamp"].split("T")[1][:12] if "T" in e["timestamp"] else e["timestamp"],
                "Operation": e["operation"],
                "Details": details_str[:150] + ("…" if len(details_str) > 150 else ""),
            })

        st.dataframe(table_data, use_container_width=True, height=min(len(table_data) * 40 + 50, 600))
    else:
        st.info("No events match the selected filters.")


def main():
    st.set_page_config(page_title="AI Literature Search Assistant", layout="wide")
    st.title("🔍 AI-Powered Literature Search Assistant with Paper Management")
    st.markdown("Intelligent literature search using AI Agent and Semantic Scholar Academic Graph API with folder management")

    # Initialize systems
    init_paper_management()
    init_monitor()

    # ==================== Sidebar Configuration ====================
    with st.sidebar:
        st.header("⚙️ Configuration")

        # ---- POE API Key input ----
        st.subheader("🔑 AI API Key")
        env_key = os.getenv("POE_API_KEY", "")
        if env_key:
            st.success("✅ POE API Key detected from environment variable.")
            st.caption("You can override it below if needed.")

        poe_key_input = st.text_input(
            "POE API Key",
            value=st.session_state.get("poe_api_key_input", ""),
            type="password",
            placeholder="Enter your Poe API key here",
            help=(
                "Required for AI-Enhanced Search and Folder AI Assistant. "
                "Get your key from https://poe.com/api_key. "
                "Alternatively set the POE_API_KEY environment variable."
            ),
            key="poe_api_key_widget"
        )
        # Persist the input into session_state under a stable key
        st.session_state["poe_api_key_input"] = poe_key_input

        has_poe_key = bool(get_poe_api_key())
        if has_poe_key:
            st.success("✅ POE API Key is configured")
        else:
            st.warning("⚠️ No POE API Key — AI features are disabled")

        st.markdown("---")

        # AI-enhanced search toggle
        ai_enhanced = st.toggle(
            "🤖 AI-Enhanced Search",
            value=True,
            help=(
                "When enabled, AI generates a search plan, extracts optimised keywords, "
                "filters & reranks results, and produces an intelligent summary. "
                "When disabled, your query is sent directly to the Semantic Scholar API "
                "and you can control results with manual filters."
            )
        )

        # Show inline warning when AI is on but key is missing
        if ai_enhanced and not has_poe_key:
            st.error(
                "🚫 AI-Enhanced Search requires a POE API Key. "
                "Please enter your key above, or disable AI-Enhanced Search to use direct API mode."
            )

        # Year range (always available)
        year_range = st.selectbox(
            "Select year range",
            ["2024-", "2023-", "2022-", "2020-", "2015-", "all"],
            help="Select the year range for papers to search"
        )
        year_param = "" if year_range == "all" else year_range

        # Result limit (always available)
        result_limit = st.slider(
            "Number of results to return",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="Number of papers returned per search"
        )

        # ---- Filters shown only when AI is disabled ----
        search_filters = {}
        if not ai_enhanced:
            search_filters = render_search_filters()

        # ---- AI-specific options shown only when AI is enabled ----
        if ai_enhanced:
            sort_option = "citationCount"  # AI will determine sort
            show_search_plan = st.checkbox("Show search plan", value=True)
            show_ai_summary = st.checkbox("Show AI intelligent summary", value=True)
        else:
            sort_option = search_filters.get("sort_by", "citationCount")
            show_search_plan = False
            show_ai_summary = False

        include_references = st.checkbox(
            "Get paper citation information (will slow down speed)",
            value=False
        )

        show_tldr = st.checkbox("Show paper key point summary (TLDR)", value=True)

        st.divider()
        st.subheader("API Information")
        if SEMANTIC_SCHOLAR_API_KEY:
            st.success("✅ Semantic Scholar API key is configured")
        else:
            st.warning("⚠️ API key not configured, using shared rate limit")
            st.caption("Set SEMANTIC_SCHOLAR_API_KEY environment variable for higher request limits")

        # Paper management panel
        render_folder_management_panel()

        # UX Monitor panel (at bottom of sidebar)
        render_monitor_panel()

    # Try to create the client (may be None if no key)
    client = init_poe_client()

    # ==================== Navigation tabs ====================
    tab_search, tab_log = st.tabs(["🔍 Search", "📋 UX Event Log"])

    with tab_log:
        render_full_log_viewer()

    with tab_search:
        # ==================== Folder View ====================
        if st.session_state.get("show_folder_view"):
            render_folder_view()
            return

        # ==================== Main Search Interface ====================
        st.subheader("📝 Enter Your Search Query")

        # Show mode indicator
        if ai_enhanced:
            if has_poe_key:
                st.caption("🤖 **Mode: AI-Enhanced Search** — AI will plan, optimise, filter, and summarise your search.")
            else:
                st.error(
                    "🚫 **AI-Enhanced Search is enabled but no POE API Key is configured.** "
                    "Please enter your POE API Key in the sidebar under 🔑 AI API Key, "
                    "or toggle off AI-Enhanced Search to use direct API mode."
                )
        else:
            st.caption("⚡ **Mode: Direct API Search** — Your query goes straight to Semantic Scholar with manual filters.")

        user_query = st.text_area(
            "Enter literature topic or keywords to search",
            placeholder="Example: Deep learning applications in medical imaging",
            height=100
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            search_button = st.button("🚀 Start Search", use_container_width=True, type="primary")

        with col2:
            clear_button = st.button("🔄 Clear Results", use_container_width=True)

        with col3:
            api_test_button = st.button("🧪 Test API Connection", use_container_width=True)

        # Test API connection
        if api_test_button:
            log_operation("API_CONNECTION_TEST", {})
            st.info("Testing Semantic Scholar API connection...")
            test_result = search_papers('"machine learning"', limit=1)

            if "error" in test_result:
                st.error(f"❌ API connection failed: {test_result['error']}")
                log_operation("API_CONNECTION_TEST_FAILED", {"error": test_result["error"][:100]})
            else:
                st.success("✅ API connection successful!")
                st.success(f"API available, retrieved {len(test_result.get('data', []))} papers")
                log_operation("API_CONNECTION_TEST_SUCCESS", {})

        # Handle clear
        if clear_button:
            log_operation("RESULTS_CLEARED", {})
            keys_to_clear = [k for k in list(st.session_state.keys())
                             if 'search' in k.lower() or 'filter' in k.lower()
                             or 'result' in k.lower() or 'last_' in k.lower()
                             or 'ai_summary' in k.lower()]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        # Block AI search if key is missing
        if search_button and user_query and ai_enhanced and not has_poe_key:
            st.error(
                "🚫 Cannot start AI-Enhanced Search without a POE API Key. "
                "Please enter your key in the sidebar or switch to Direct API Search mode."
            )
            st.stop()

        # Handle search button press
        if search_button and user_query:
            log_operation("SEARCH_INITIATED", {
                "query": user_query[:200],
                "ai_enhanced": ai_enhanced,
                "year_range": year_param,
                "result_limit": result_limit,
                "sort": sort_option,
            })
            st.session_state["last_search_query"] = user_query
            st.session_state["last_year_param"] = year_param
            st.session_state["last_result_limit"] = result_limit
            st.session_state["last_include_references"] = include_references
            st.session_state["last_show_search_plan"] = show_search_plan
            st.session_state["last_show_ai_summary"] = show_ai_summary
            st.session_state["last_ai_enhanced"] = ai_enhanced
            st.session_state["last_sort_option"] = sort_option
            st.session_state["last_search_filters"] = search_filters
            # Invalidate cached results
            st.session_state.pop("search_results_cached", None)
            st.session_state.pop("ai_summary_cached", None)

        # ==================== Display Search Results ====================
        if st.session_state.get("last_search_query"):
            user_query = st.session_state["last_search_query"]
            year_param = st.session_state.get("last_year_param", "")
            result_limit = st.session_state.get("last_result_limit", 20)
            include_references = st.session_state.get("last_include_references", False)
            show_search_plan = st.session_state.get("last_show_search_plan", False)
            show_ai_summary = st.session_state.get("last_show_ai_summary", False)
            is_ai_enhanced = st.session_state.get("last_ai_enhanced", True)
            sort_option = st.session_state.get("last_sort_option", "citationCount")
            cached_filters = st.session_state.get("last_search_filters", {})

            st.divider()

            # ===================================================================
            # PATH A: AI-Enhanced Search
            # ===================================================================
            if is_ai_enhanced:
                if client is None:
                    st.error(
                        "🚫 AI features require a valid POE API Key. "
                        "Please enter your key in the sidebar under 🔑 AI API Key."
                    )
                    return

                if "search_results_cached" not in st.session_state:
                    # Step 1: Generate search plan
                    search_plan_text = ""
                    if show_search_plan:
                        st.subheader("📋 Step 1: AI-Generated Search Plan")
                        with st.spinner("Analyzing query and generating search plan..."):
                            search_plan_text = generate_search_plan(client, user_query)
                            st.markdown(search_plan_text)

                    # Step 2: Extract keywords and sorting method
                    st.subheader("🔎 Step 2: Extract Search Parameters")
                    with st.spinner("Extracting search keywords and sorting method..."):
                        if search_plan_text:
                            search_keyword, extracted_sort = extract_search_keyword(client, search_plan_text, user_query)
                        else:
                            with st.spinner("AI is planning the search..."):
                                hidden_plan = generate_search_plan(client, user_query)
                            search_keyword, extracted_sort = extract_search_keyword(client, hidden_plan, user_query)

                        st.info(f"📌 Using keyword: **{search_keyword}** | Sort by: **{extracted_sort}**")

                    # Step 3: Execute search
                    st.subheader("📊 Step 3: Execute Literature Search")
                    with st.spinner("Searching Semantic Scholar database..."):
                        search_results = search_papers(
                            search_keyword,
                            year_range=year_param if year_param else "2023-",
                            limit=result_limit * 2,
                            sort_by=extracted_sort
                        )

                    # Step 3.5: AI Filter and Rerank
                    st.subheader("🎯 Step 3.5: AI Filter and Rerank Results")

                    if "error" not in search_results:
                        papers = search_results.get("data", [])
                        with st.spinner("AI is filtering and reranking results based on relevance..."):
                            filtered_papers = filter_and_rerank_papers(client, user_query, papers, result_limit)

                        if filtered_papers:
                            st.success(f"✅ Filtered and ranked to **{len(filtered_papers)}** most relevant papers")
                        else:
                            st.warning("No relevant papers found after filtering")
                            filtered_papers = papers[:result_limit]
                    else:
                        st.error(f"❌ Search error: {search_results['error']}")
                        filtered_papers = []

                    st.session_state["search_results_cached"] = filtered_papers
                    log_operation("SEARCH_COMPLETED", {
                        "mode": "ai_enhanced",
                        "result_count": len(filtered_papers),
                    })
                else:
                    filtered_papers = st.session_state.get("search_results_cached", [])

                # Step 4: Display search results
                st.subheader("📚 Step 4: Search Results")
                if filtered_papers:
                    display_search_results(filtered_papers, include_references)

                # Step 5: AI summary and organization
                if show_ai_summary and filtered_papers:
                    if "ai_summary_cached" not in st.session_state:
                        st.divider()
                        st.subheader("✨ Step 5: AI Intelligent Summary and Analysis")
                        with st.spinner("Organizing and analyzing search results..."):
                            organized_summary = organize_results(client, user_query, {"data": filtered_papers})
                            st.session_state["ai_summary_cached"] = organized_summary

                    st.divider()
                    st.subheader("✨ Step 5: AI Intelligent Summary and Analysis")
                    st.markdown(st.session_state.get("ai_summary_cached", ""))

            # ===================================================================
            # PATH B: Direct Semantic Scholar API Search (AI disabled)
            # ===================================================================
            else:
                if "search_results_cached" not in st.session_state:
                    st.subheader("📊 Direct API Search Results")

                    # Build a readable summary of active filters
                    filter_parts = [f"Query: **{user_query}**"]
                    filter_parts.append(f"Sort: **{sort_option}**")
                    filter_parts.append(f"Year: **{year_param if year_param else 'all'}**")
                    filter_parts.append(f"Limit: **{result_limit}**")

                    if cached_filters.get("fields_of_study"):
                        filter_parts.append(f"Fields: **{', '.join(cached_filters['fields_of_study'])}**")
                    if cached_filters.get("min_citation_count") and cached_filters["min_citation_count"] > 0:
                        filter_parts.append(f"Min citations: **{cached_filters['min_citation_count']}**")
                    if cached_filters.get("publication_types"):
                        filter_parts.append(f"Types: **{', '.join(cached_filters['publication_types'])}**")
                    if cached_filters.get("open_access_only"):
                        filter_parts.append("**Open Access only**")
                    if cached_filters.get("venue"):
                        filter_parts.append(f"Venue: **{cached_filters['venue']}**")

                    st.info("📌 " + " | ".join(filter_parts))

                    with st.spinner("Searching Semantic Scholar database..."):
                        search_results = search_papers(
                            user_query,
                            year_range=year_param if year_param else "",
                            limit=result_limit,
                            sort_by=sort_option,
                            fields_of_study=cached_filters.get("fields_of_study"),
                            min_citation_count=cached_filters.get("min_citation_count"),
                            publication_types=cached_filters.get("publication_types"),
                            open_access_only=cached_filters.get("open_access_only", False),
                            venue=cached_filters.get("venue")
                        )

                    if "error" in search_results:
                        st.error(f"❌ Search error: {search_results['error']}")
                        filtered_papers = []
                    else:
                        filtered_papers = search_results.get("data", [])[:result_limit]
                        if not filtered_papers:
                            st.warning("No papers found. Try broadening your query or relaxing the filters.")

                    st.session_state["search_results_cached"] = filtered_papers
                    log_operation("SEARCH_COMPLETED", {
                        "mode": "direct_api",
                        "result_count": len(filtered_papers),
                    })
                else:
                    filtered_papers = st.session_state.get("search_results_cached", [])

                # Display search results
                st.subheader("📚 Search Results")
                if filtered_papers:
                    display_search_results(filtered_papers, include_references)

        # ==================== Footer ====================
        st.divider()
        st.markdown("""
        ---
        ### 📖 Usage Instructions
        - 📝 Enter your research topic or keywords in the input box
        - 🚀 Click the "Start Search" button to initiate the search process
        - 🤖 Toggle **AI-Enhanced Search** on/off in the sidebar
          - **ON:** AI generates a search plan, extracts keywords, filters & reranks results, and provides an intelligent summary
          - **OFF:** Your query is sent directly to the Semantic Scholar API — use the **Search Filters** panel to refine results manually
        - 📊 Display detailed search results
        - 📂 Add papers to folders for better organization
        - 🤖 Ask AI questions based on your paper collections

        ### 🔑 Setting Up AI Features
        - Enter your **POE API Key** in the sidebar under 🔑 AI API Key
        - Alternatively set the `POE_API_KEY` environment variable before starting the app
        - Get your key from [poe.com/api_key](https://poe.com/api_key)

        ### 🔧 Search Filters (Direct API Mode)
        When AI-Enhanced Search is disabled, the following filters become available in the sidebar:
        - **Sort by:** Citation Count, Publication Date, or Relevance
        - **Fields of Study:** Filter by academic discipline (Computer Science, Medicine, etc.)
        - **Minimum Citation Count:** Only show papers with at least N citations
        - **Publication Types:** Journal Article, Conference, Review, Meta-Analysis, etc.
        - **Open Access Only:** Only show papers with free PDFs
        - **Venue / Journal / Conference:** Filter by specific publication venue

        ### 📚 Paper Management Features
        - 📁 **Create Folders:** Organize papers by topics or projects
        - ➕ **Add Papers:** Click the ➕ button next to each paper to add to folders
        - 🗑️ **Remove Papers:** Delete papers from folders
        - 🤖 **AI Assistant:** Ask questions about papers in a folder
        - 📥 **Export:** Download papers as JSON or Markdown

        ### 🔴 UX Monitor & Logger
        - Click **▶️ Start Monitoring** in the sidebar to begin recording
        - All user operations (search, filter changes, paper views, folder actions, etc.) are timestamped
        - Click **⏹ End Monitoring** to stop the session
        - View the full event log in the **📋 UX Event Log** tab
        - Export logs as **JSON** or **CSV** for analysis

        ### 🔍 Supported Search Syntax
        - **Single keyword:** `deep learning`
        - **Exact phrase:** `"neural network optimization"`
        - **Boolean query:** `(artificial intelligence | machine learning) medical imaging`
        - **Exclude word:** `artificial intelligence -privacy`

        ### 💡 Tips
        - Using quotes for exact matching can improve search accuracy
        - Adjust year range to get latest or classic papers
        - Check citation count to understand paper impact
        - ⭐ Indicates highly influential papers with over 100 citations
        - 🔓/🔒 Indicates whether paper is open access
        - Turn off AI-Enhanced Search for faster results when you already know the exact keywords
        - Use the minimum citation filter to quickly find high-impact papers
        """)


if __name__ == "__main__":
    main()