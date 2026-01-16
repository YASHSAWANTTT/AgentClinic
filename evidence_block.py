# evidence_block.py
import json
import re
import unicodedata
from jsonschema import validate, ValidationError

def load_schema(path="schemas/evidence_block_schema.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

EVIDENCE_SCHEMA = load_schema()

# Commit marker (accept DIAGNOSIS/Diagnosis, case-insensitive)
_COMMIT_RE = re.compile(r"(?:DIAGNOSIS|Diagnosis)\s+READY:", re.I)

# Tag that precedes the JSON
_TAG_RE = re.compile(r"EVIDENCE_BLOCK_JSON:\s*", re.I | re.S)

# ---------- sanitation helpers ----------
_ZW_CHARS = (
    "\u200b"  # zero width space
    "\u200c"  # zero width non-joiner
    "\u200d"  # zero width joiner
    "\ufeff"  # BOM
)

def _normalize_quotes(s: str) -> str:
    return (s.replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'"))

def _strip_zero_width(s: str) -> str:
    return s.translate({ord(ch): None for ch in _ZW_CHARS})

def _remove_trailing_commas(s: str) -> str:
    # remove commas before } or ]
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s

def _sanitize_json_text(raw: str) -> str:
    s = unicodedata.normalize("NFKC", raw)
    s = _strip_zero_width(s)
    s = _normalize_quotes(s)
    s = _remove_trailing_commas(s)
    return s.strip()

# ---------- balanced JSON extraction ----------
def _extract_balanced_json(text: str):
    """
    Find 'EVIDENCE_BLOCK_JSON:' then take the JSON object that follows by
    balancing braces (ignoring braces inside quoted strings).
    Returns (json_text, error|None).
    """
    if not text:
        return None, "missing"
    tag = _TAG_RE.search(text)
    if not tag:
        return None, "missing"

    # Limit search to before the next commit line (if present),
    # so stray braces after the JSON don't confuse us.
    commit = _COMMIT_RE.search(text, tag.end())
    end_limit = commit.start() if commit else len(text)

    i = text.find("{", tag.end(), end_limit)
    if i == -1:
        return None, "missing"

    depth = 0
    in_str = False
    esc = False
    for j in range(i, end_limit):
        ch = text[j]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # end of the balanced JSON
                    return text[i:j+1], None
    return None, "invalid_json:unterminated_object"

# ---------- public API ----------
def extract_evidence_block(text: str):
    """
    Return (obj, error_str|None).
    Extracts the JSON block after EVIDENCE_BLOCK_JSON:, sanitizes, parses, validates.
    """
    raw, err = _extract_balanced_json(text or "")
    if err:
        return None, err
    cleaned = _sanitize_json_text(raw)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, f"invalid_json:{str(e)}"
    try:
        validate(obj, EVIDENCE_SCHEMA)
    except ValidationError as e:
        return None, f"schema_error:{e.message}"
    return obj, None

def has_commit_line(text: str) -> bool:
    return bool(_COMMIT_RE.search(text or ""))

def extract_final_dx(text: str) -> str:
    if not text:
        return ""
    it = list(_COMMIT_RE.finditer(text))
    if not it:
        return ""
    last = it[-1]
    return text[last.end():].strip()
