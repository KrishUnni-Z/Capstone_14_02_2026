"""Decidr — Goal Extractor (v5).

AWS Bedrock Claude Haiku, grounded in the 35 canonical goals the System 2
scorer was trained on. Pre-fills metric/unit and full L1/L2/L3 buckets when
the user's description matches a known goal.
"""
import json, os, logging, re
from typing import Optional

import boto3
from dotenv import load_dotenv
from System_1.schemas import BUCKET_HIERARCHY

load_dotenv()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Canonical goals — verified against analytical_flat.csv (all 35).
#  Edit this table when goals are added or renamed in the dataset.
#  Format: (l1, l2, l3, metric_name, unit)
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_GOALS = [
    ("Marketing", "Paid Acquisition",        "Google Ads - Search",     "Performance Metric - Google Ads - Search",  "score"),
    ("Marketing", "Paid Acquisition",        "Google Ads - Display",    "Performance Metric - Google Ads - Display", "score"),
    ("Marketing", "Paid Acquisition",        "Social Media Ads",        "ROAS - Social",                              "ratio"),
    ("Marketing", "Content & SEO",           "Blog Content Production", "Organic Traffic",                            "visitors"),
    ("Marketing", "Content & SEO",           "SEO Optimization",        "SEO Ranking Score",                          "score"),
    ("Marketing", "Content & SEO",           "Video Content",           "Video Engagement Rate",                      "percentage"),
    ("Marketing", "Partnerships",            "Partner Acquisition",     "Partner Count",                              "count"),
    ("Marketing", "Partnerships",            "Co-Marketing Campaigns",  "Co-Marketing ROI",                           "ratio"),
    ("Marketing", "Partnerships",            "Channel Partnerships",    "Channel Revenue",                            "dollars"),
    ("Marketing", "Events & Community",      "Virtual Events",          "Virtual Event Attendance",                   "attendees"),
    ("Marketing", "Events & Community",      "In-Person Events",        "Event Lead Generation",                      "leads"),
    ("Marketing", "Events & Community",      "Community Management",    "Community Engagement Score",                 "score"),
    ("Product",   "Core Platform",           "Backend Services",        "API Response Time",                          "ms"),
    ("Product",   "Core Platform",           "Frontend Development",    "Page Load Time",                             "seconds"),
    ("Product",   "Core Platform",           "APIs & Integrations",     "Performance Metric - APIs & Integrations",   "score"),
    ("Product",   "New Features",            "Feature Development",     "Feature Adoption Rate",                      "percentage"),
    ("Product",   "New Features",            "A/B Testing",             "Performance Metric - A/B Testing",           "score"),
    ("Product",   "New Features",            "Feature Launch",          "Launch Success Rate",                        "percentage"),
    ("Product",   "Technical Debt",          "Code Refactoring",        "Code Quality Score",                         "score"),
    ("Product",   "Technical Debt",          "Infrastructure Upgrades", "System Uptime",                              "percentage"),
    ("Product",   "Research & Innovation",   "Proof of Concepts",       "POC Success Rate",                           "percentage"),
    ("Product",   "Research & Innovation",   "Innovation Labs",         "Innovation Pipeline",                        "projects"),
    ("Operations","Customer Success",        "Onboarding",              "Time to Value",                              "days"),
    ("Operations","Customer Success",        "Support Tickets",         "Ticket Resolution Time",                     "hours"),
    ("Operations","Customer Success",        "Account Management",      "NPS Score",                                  "score"),
    ("Operations","Infrastructure & DevOps", "Cloud Infrastructure",    "Infrastructure Reliability",                 "percentage"),
    ("Operations","Infrastructure & DevOps", "CI/CD Pipeline",          "Performance Metric - CI/CD Pipeline",        "score"),
    ("Operations","Data & Analytics",        "Data Engineering",        "Data Pipeline Reliability",                  "percentage"),
    ("Operations","Data & Analytics",        "Analytics & Reporting",   "Performance Metric - Analytics & Reporting", "score"),
    ("G&A",       "Finance & Legal",         "Financial Planning",      "Budget Variance",                            "percentage"),
    ("G&A",       "Finance & Legal",         "Legal & Compliance",      "Performance Metric - Legal & Compliance",    "score"),
    ("G&A",       "HR & Recruiting",         "Talent Acquisition",      "Time to Hire",                               "days"),
    ("G&A",       "HR & Recruiting",         "Employee Development",    "Employee Satisfaction",                      "score"),
    ("G&A",       "General Admin",           "Office Operations",       "Operational Efficiency",                     "score"),
    ("G&A",       "General Admin",           "Vendor Management",       "Vendor SLA Compliance",                      "percentage"),
]

# Quick lookup: lowercase metric name → canonical record. Used by the
# heuristic fallback so it can match user input to known goals even
# when Bedrock is unreachable.
_METRIC_LOOKUP = {m.lower(): (l1, l2, l3, m, u) for l1, l2, l3, m, u in KNOWN_GOALS}


def _hier_str():
    """Render BUCKET_HIERARCHY as a tree the LLM can scan."""
    lines = []
    for l1, l2s in BUCKET_HIERARCHY.items():
        lines.append(f"L1: {l1}")
        for l2, l3s in l2s.items():
            lines.append(f"  L2: {l2}")
            for l3 in l3s:
                lines.append(f"    L3: {l3}")
    return "\n".join(lines)


def _known_goals_str():
    """Compact, scannable list of the 35 canonical goals for the LLM."""
    lines = []
    for l1, l2, l3, metric, unit in KNOWN_GOALS:
        lines.append(f"  {l1} > {l2} > {l3}: {metric} ({unit})")
    return "\n".join(lines)


PROMPT = f"""You are a goal parser for the Decidr Coherence Engine.

The system tracks 35 specific goals across this organisation hierarchy. Your job is to map a user's plain-English description to one of these known goals — or, if the description doesn't match any known goal, extract whatever structured information you can.

ORGANISATION HIERARCHY:
{_hier_str()}

KNOWN GOALS (l1 > l2 > l3 : metric_name (unit)):
{_known_goals_str()}

YOUR TASK

Given a user's description, return JSON with these fields:
- goal_title: concise summary, max 8 words, title case, no filler ("I want to", "We need to")
- scope: "goal" (specific function), "l2_bucket" (department), "l1_bucket" (division)
- bucket_l1, bucket_l2, bucket_l3: from the hierarchy above
- target_value, initial_value: numbers extracted from "from X to Y", "by 40%", "reach 85" — null if absent
- metric_suggestion: prefer the exact metric_name from the KNOWN GOALS list above when the user's description matches one. If the user describes a goal that isn't in the list, suggest a sensible canonical name (e.g. "Conversion Rate", "MRR"). Leave blank only if you genuinely can't tell what's being measured.
- unit_suggestion: one of score, percentage, ratio, count, dollars, visitors, leads, attendees, ms, seconds, hours, days, projects

MATCHING RULES

1. If the user's description matches a known goal, fill ALL of l1/l2/l3/metric/unit from the table — even cross-cutting metrics like "NPS" map to a specific bucket in this system (NPS Score lives under Operations > Customer Success > Account Management).

2. If the user describes something at department or division level ("all paid channels", "across content"), set scope accordingly and leave l3 null. Fill l1/l2 from the hierarchy.

3. If the user describes a goal that doesn't exist in the known list (e.g. "MRR", "headcount", "p99 latency"), still fill metric_suggestion and unit_suggestion with a sensible canonical value. Leave bucket fields blank rather than guessing wrong.

4. Pre-fill every field you can identify with reasonable confidence (>70%). Blank/null is appropriate ONLY when the text genuinely doesn't tell you.

EXAMPLES

Input: "Improve NPS score from 60 to 95 over 24 months"
Output: {{"goal_title":"Improve NPS Score","scope":"goal","bucket_l1":"Operations","bucket_l2":"Customer Success","bucket_l3":"Account Management","target_value":95,"initial_value":60,"metric_suggestion":"NPS Score","unit_suggestion":"score"}}

Input: "Boost organic traffic from 25k to 50k visitors"
Output: {{"goal_title":"Boost Organic Traffic","scope":"goal","bucket_l1":"Marketing","bucket_l2":"Content & SEO","bucket_l3":"Blog Content Production","target_value":50000,"initial_value":25000,"metric_suggestion":"Organic Traffic","unit_suggestion":"visitors"}}

Input: "Cut Google Ads search performance gap, target 85"
Output: {{"goal_title":"Improve Google Ads Search Performance","scope":"goal","bucket_l1":"Marketing","bucket_l2":"Paid Acquisition","bucket_l3":"Google Ads - Search","target_value":85,"initial_value":null,"metric_suggestion":"Performance Metric - Google Ads - Search","unit_suggestion":"score"}}

Input: "Reduce time to hire from 60 days to 25"
Output: {{"goal_title":"Reduce Time To Hire","scope":"goal","bucket_l1":"G&A","bucket_l2":"HR & Recruiting","bucket_l3":"Talent Acquisition","target_value":25,"initial_value":60,"metric_suggestion":"Time to Hire","unit_suggestion":"days"}}

Input: "How is paid acquisition performing across all channels?"
Output: {{"goal_title":"Paid Acquisition Performance","scope":"l2_bucket","bucket_l1":"Marketing","bucket_l2":"Paid Acquisition","bucket_l3":null,"target_value":null,"initial_value":null,"metric_suggestion":"","unit_suggestion":""}}

Input: "Grow MRR to $2M by year end"
Output: {{"goal_title":"Grow MRR To $2M","scope":"goal","bucket_l1":"","bucket_l2":"","bucket_l3":null,"target_value":2000000,"initial_value":null,"metric_suggestion":"MRR","unit_suggestion":"dollars"}}

Input: "Launch success rate needs to hit 95%"
Output: {{"goal_title":"Improve Launch Success Rate","scope":"goal","bucket_l1":"Product","bucket_l2":"New Features","bucket_l3":"Feature Launch","target_value":95,"initial_value":null,"metric_suggestion":"Launch Success Rate","unit_suggestion":"percentage"}}

Input: "Improve all of marketing"
Output: {{"goal_title":"Improve Marketing","scope":"l1_bucket","bucket_l1":"Marketing","bucket_l2":"","bucket_l3":null,"target_value":null,"initial_value":null,"metric_suggestion":"","unit_suggestion":""}}

Return ONLY the JSON object, no preamble, no commentary."""


class GoalExtractor:
    def __init__(
        self,
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-west-2",
        use_llm=True,
    ):
        self.model_id = model_id or os.getenv(
            "BEDROCK_MODEL_ID",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        )
        self.region_name = region_name or os.getenv("AWS_REGION", "us-west-2")
        self.use_llm = use_llm

        self.client = None
        if self.use_llm:
            try:
                self.client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.region_name,
                )
            except Exception as e:
                logger.warning(f"Could not initialise Bedrock client. Falling back to heuristics: {e}")
                self.use_llm = False

    def extract(self, nl):
        if self.use_llm:
            try:
                return self._llm(nl)
            except Exception as e:
                logger.warning(f"Bedrock LLM failed: {e}")
        return self._heuristic(nl)

    def _llm(self, nl):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 700,
            "temperature": 0.1,
            "system": PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": nl,
                        }
                    ],
                }
            ],
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        text = result["content"][0]["text"].strip()

        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in Bedrock response")
            d = json.loads(match.group(0))

        # Validate buckets against the canonical hierarchy. If the LLM
        # hallucinates a bucket that doesn't exist, drop it — the user
        # picks manually rather than trusting a wrong auto-fill.
        l1, l2, l3 = d.get("bucket_l1", ""), d.get("bucket_l2", ""), d.get("bucket_l3")
        if l1 and l1 not in BUCKET_HIERARCHY:
            l1 = ""
        if l2 and l1 and l2 not in BUCKET_HIERARCHY.get(l1, {}):
            l2 = ""
        if l3 and l1 and l2 and l3 not in BUCKET_HIERARCHY.get(l1, {}).get(l2, []):
            l3 = None

        scope = d.get("scope", "goal")
        if scope not in ("goal", "l2_bucket", "l1_bucket"):
            scope = "goal"
        # If l3 is missing but l2 is set, we're at department scope, not goal scope.
        if not l3 and l2 and scope == "goal":
            scope = "l2_bucket"

        return {
            "goal_title": d.get("goal_title", "")[:80],
            "scope": scope,
            "bucket_l1": l1,
            "bucket_l2": l2,
            "bucket_l3": l3,
            "target_value": d.get("target_value"),
            "initial_value": d.get("initial_value"),
            "metric_suggestion": d.get("metric_suggestion", ""),
            "unit_suggestion": d.get("unit_suggestion", ""),
        }

    def _heuristic(self, nl):
        """Fallback when Bedrock is unreachable. Now also scans for known
        metric names so 'improve NPS' still pre-fills the correct bucket
        even without the LLM."""
        t = nl.strip()
        for sep in [".", ",", " by ", " through "]:
            parts = nl.split(sep)
            if len(parts) > 1:
                t = parts[0].strip()
                break
        for p in ["I want to", "We need to", "Our goal is to", "We want to"]:
            if t.lower().startswith(p.lower()):
                t = t[len(p):]
                break
        t = " ".join(t.split()[:8])

        tl = nl.lower()
        scope = "goal"
        if any(s in tl for s in ["all channels", "across", "all of", "entire"]):
            scope = "l2_bucket"

        # First pass: scan KNOWN_GOALS for a literal metric-name match.
        # If the user typed 'NPS' or 'Organic Traffic' the heuristic
        # fills the full bucket path from the canonical table.
        l1 = l2 = ""
        l3_match = None
        metric_suggestion = ""
        unit_suggestion = ""
        for key, (kl1, kl2, kl3, kmetric, kunit) in _METRIC_LOOKUP.items():
            # Match the metric name OR a strong substring of it. "nps" matches
            # "nps score"; "organic traffic" matches itself; "google ads search"
            # matches "performance metric - google ads - search".
            short = key.replace("performance metric - ", "")
            if short in tl or key in tl:
                l1, l2, l3_match = kl1, kl2, kl3
                metric_suggestion = kmetric
                unit_suggestion = kunit
                break

        # Second pass: bucket-name match (only if metric scan didn't fill).
        if not l1:
            for l1k, l2s in BUCKET_HIERARCHY.items():
                if l1k.lower() in tl:
                    l1 = l1k
                    for l2k, l3s in l2s.items():
                        if l2k.lower() in tl:
                            l2 = l2k
                            for l3v in l3s:
                                if l3v.lower() in tl:
                                    l3_match = l3v
                                    break
                            break
                    break

        # Numeric extraction. Handles "from 60 to 95", "from 25k to 50k",
        # "from $2M to $5M", and bare "target 85" / "to 95".
        def _num(s):
            s = s.replace(",", "").replace("$", "").strip().lower()
            mult = 1.0
            if s.endswith("k"):  s, mult = s[:-1], 1_000
            elif s.endswith("m"): s, mult = s[:-1], 1_000_000
            elif s.endswith("b"): s, mult = s[:-1], 1_000_000_000
            try: return float(s) * mult
            except ValueError: return None

        tv = iv = None
        m = re.search(r"from\s+\$?(\d+\.?\d*[kmbKMB]?)\s+to\s+\$?(\d+\.?\d*[kmbKMB]?)", tl)
        if m:
            iv = _num(m.group(1))
            tv = _num(m.group(2))
        else:
            # Just a target: "to 95", "target 85", "reach 50000", "hit 95%"
            m = re.search(r"(?:to|target|reach|hit)\s+\$?(\d+\.?\d*[kmbKMB]?)", tl)
            if m: tv = _num(m.group(1))

        # Unit fallback if metric scan didn't set one
        if not unit_suggestion:
            for u, kws in {
                "percentage": ["%", "percent", "rate"],
                "score":      ["score"],
                "ratio":      ["ratio", "roas"],
                "visitors":   ["visitor", "traffic"],
                "dollars":    ["$", "revenue"],
            }.items():
                if any(k in tl for k in kws):
                    unit_suggestion = u
                    break

        return {
            "goal_title": t.capitalize() if t else "",
            "scope": scope,
            "bucket_l1": l1,
            "bucket_l2": l2,
            "bucket_l3": l3_match,
            "target_value": tv,
            "initial_value": iv,
            "metric_suggestion": metric_suggestion,
            "unit_suggestion": unit_suggestion,
        }
