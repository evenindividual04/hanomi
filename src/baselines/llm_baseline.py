"""LLM baseline: serialize B-Rep subgraph → JSON → Claude/GPT → classification.

Baseline 1: never serialize the full model. Instead:
  1. Run per-face similarity to find top-K candidate seeds
  2. Serialize 2-hop neighborhood around each seed as JSON
  3. Send to LLM with structured prompt
  4. Parse JSON response: {is_match, confidence, reasoning}

Token cost: ~800 tokens per subgraph query.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

SURFACE_TYPE_NAMES = {0: "plane", 1: "cylinder", 2: "cone",
                       3: "sphere", 4: "torus", 5: "other"}


def _decode_surface_type(val: float) -> int:
    return round(float(val) * 11)


SYSTEM_PROMPT_TEMPLATE = """\
You are a CAD feature recognition system. Given a JSON description of a set of
connected faces from a B-Rep CAD model, determine if they represent a {feature_type}.

{feature_description}

Respond with JSON only (no other text):
{{"is_match": true/false, "confidence": 0.0-1.0, "reasoning": "one sentence"}}
"""

FEATURE_DESCRIPTIONS = {
    "through_hole": (
        "A through-hole consists of a single cylindrical face connected via concave "
        "edges to flat (planar) faces on both ends. Size and orientation are irrelevant."
    ),
    "counterbored_hole": (
        "A counterbored hole consists of: (1) a large-radius outer cylinder connected "
        "via a concave edge to a flat annular plane, which is connected via a concave "
        "edge to (2) a smaller-radius inner cylinder. The outer bore is wider than the inner."
    ),
    "blind_hole": (
        "A blind hole is a cylinder with one open circular end and one flat planar bottom. "
        "It does NOT pass through the part."
    ),
    "rectangular_pocket": (
        "A rectangular pocket has four vertical planar walls, four concave edges at "
        "corners, and one flat planar bottom face."
    ),
}


def _get_neighbors(data: Data, face_idx: int) -> List[int]:
    """Return neighbor face indices."""
    mask = data.edge_index[0] == face_idx
    return data.edge_index[1, mask].tolist()


def _get_dihedral_angles(data: Data, face_idx: int) -> List[float]:
    """Return dihedral angles of edges incident to face_idx."""
    mask = data.edge_index[0] == face_idx
    if data.edge_attr is None:
        return []
    return [round(float(data.edge_attr[i, 0]), 3)
            for i in mask.nonzero(as_tuple=True)[0].tolist()]


def _get_convexity_labels(data: Data, face_idx: int) -> List[str]:
    """Return convexity of edges incident to face_idx."""
    if data.edge_attr is None:
        return []
    mask = data.edge_index[0] == face_idx
    labels = []
    for i in mask.nonzero(as_tuple=True)[0].tolist():
        conv = float(data.edge_attr[i, 0])
        if conv > 0.5:
            labels.append("convex")
        elif conv < -0.5:
            labels.append("concave")
        else:
            labels.append("smooth")
    return labels


def serialize_subgraph(data: Data, face_indices: List[int]) -> str:
    """Serialize a local subgraph to JSON string for LLM consumption.

    Args:
        data         : PyG Data for one model
        face_indices : list of face indices to serialize (2-hop neighborhood)

    Returns:
        JSON string (≤ ~800 tokens for typical subgraphs)
    """
    face_set = set(face_indices)
    faces_json = []
    for idx in sorted(face_indices):
        surf_type = SURFACE_TYPE_NAMES.get(_decode_surface_type(data.x[idx, 4].item()), "other")
        area      = round(float(data.x[idx, 0].item()), 5)
        nbrs      = _get_neighbors(data, idx)
        nbrs_in   = [n for n in nbrs if n in face_set]
        nbrs_out  = [n for n in nbrs if n not in face_set]
        convexity = _get_convexity_labels(data, idx)

        faces_json.append({
            "face_id":        idx,
            "surface_type":   surf_type,
            "area":           area,
            "adjacent_in_subgraph": nbrs_in,
            "adjacent_outside":     nbrs_out[:5],  # truncate to avoid token explosion
            "edge_convexities": convexity,
        })
    return json.dumps({"faces": faces_json}, indent=2)


def _two_hop_subgraph(data: Data, seed: int) -> List[int]:
    """Return 2-hop face neighborhood around a seed face."""
    visited = {seed}
    frontier = {seed}
    for _ in range(2):
        next_frontier = set()
        for f in frontier:
            mask = data.edge_index[0] == f
            nbrs = data.edge_index[1, mask].tolist()
            for n in nbrs:
                if n not in visited:
                    visited.add(n)
                    next_frontier.add(n)
        frontier = next_frontier
    return sorted(visited)


def _parse_json_response(text: str) -> Optional[Dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
    return None


def query_llm(
    prompt: str,
    system: str,
    model: str = "gemini-2.0-flash",
    max_tokens: int = 256,
) -> Optional[Dict]:
    """Send a prompt to an LLM and parse JSON response.

    Supported providers (detected by model name prefix):
      - gemini-*   → Google Gemini via google-genai
      - groq-*     → Groq via groq SDK
      - llama-*    → Groq via groq SDK
      - mixtral-*  → Groq via groq SDK
      - claude-*   → Anthropic via anthropic SDK

    Returns parsed dict or None on failure.
    """
    def _call_once(active_model: str) -> Optional[Dict]:
        if active_model.startswith(("gemini-", "gemma-")):
            from google import genai
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            full_prompt = f"{system}\n\n{prompt}"
            response = client.models.generate_content(
                model=active_model,
                contents=full_prompt,
            )
            return _parse_json_response(response.text)

        if active_model.startswith(("groq-", "llama-", "mixtral-", "deepseek-")):
            from groq import Groq
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            response = client.chat.completions.create(
                model=active_model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return _parse_json_response(response.choices[0].message.content)

        if active_model.startswith("claude-"):
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=active_model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_json_response(response.content[0].text)

        print(f"Unknown model prefix: {active_model}")
        return None

    def _is_retryable_error(error: Exception) -> bool:
        message = str(error).lower()
        non_retryable_tokens = (
            'api key',
            'apikey',
            'authentication',
            'unauthorized',
            'invalid model',
            'unsupported',
            'module not found',
            'no module named',
            'importerror',
            'keyerror',
            'valueerror',
        )
        return not any(token in message for token in non_retryable_tokens)

    fallback_chain = [model]
    if model.startswith(("gemini-", "gemma-")) and os.environ.get("GROQ_API_KEY"):
        fallback_chain.append("llama-3.3-70b-versatile")
    if model.startswith(("gemini-", "gemma-", "groq-", "llama-", "mixtral-", "deepseek-")) and os.environ.get("ANTHROPIC_API_KEY"):
        fallback_chain.append("claude-3-5-sonnet-latest")

    for active_model in fallback_chain:
        for attempt in range(3):
            try:
                result = _call_once(active_model)
                if result is not None:
                    return result
            except Exception as e:
                if not _is_retryable_error(e):
                    print(f"LLM query configuration error on {active_model}: {e}")
                    break
                print(f"LLM query failed on {active_model} attempt {attempt + 1}/3: {e}")
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))

    return None


def run_llm_baseline(
    data: Data,
    feature_type: str,
    candidate_seeds: Optional[List[int]] = None,
    max_seeds: int = 10,
    llm_model: str = "gemini-2.0-flash",
) -> List[Dict]:
    """Run the LLM baseline on one model.

    If candidate_seeds is None, uses all cylindrical faces as seeds (for
    hole-type features). Otherwise uses the provided seed list.

    Returns:
        list of {face_ids, confidence, reasoning}
    """
    description = FEATURE_DESCRIPTIONS.get(
        feature_type,
        "Match the described feature type based on topology and surface types.",
    )
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        feature_type=feature_type,
        feature_description=description,
    )

    if candidate_seeds is None:
        if feature_type in ("through_hole", "counterbored_hole", "blind_hole"):
            candidate_seeds = [
                i for i in range(data.num_nodes)
                if _decode_surface_type(data.x[i, 4].item()) == 1
            ]
        else:
            candidate_seeds = list(range(data.num_nodes))

    seeds_to_query = candidate_seeds[:max_seeds]

    instances = []
    for seed in seeds_to_query:
        subgraph_indices = _two_hop_subgraph(data, seed)
        if len(subgraph_indices) > 50:
            subgraph_indices = subgraph_indices[:50]

        subgraph_json = serialize_subgraph(data, subgraph_indices)
        user_message  = (
            f"Determine if the following B-Rep subgraph represents a {feature_type}:\n\n"
            f"{subgraph_json}"
        )

        t0     = time.time()
        result = query_llm(user_message, system_prompt, model=llm_model)
        latency_ms = (time.time() - t0) * 1000

        if result and result.get("is_match") and result.get("confidence", 0) > 0.5:
            instances.append({
                "face_ids":    subgraph_indices,
                "confidence":  result.get("confidence", 0.5),
                "reasoning":   result.get("reasoning", ""),
                "latency_ms":  latency_ms,
                "_cluster_set": set(subgraph_indices),
            })

    from src.inference.nms import non_max_suppression
    return non_max_suppression(instances, iou_threshold=0.5)


def run_llm_baselines_multi(
    data: Data,
    feature_type: str,
    models: Optional[List[str]] = None,
    candidate_seeds: Optional[List[int]] = None,
    max_seeds: int = 10,
) -> Dict[str, List[Dict]]:
    """Run multiple LLM providers on the same model and return per-provider results.

    Args:
        models: list of model strings, e.g. ["gemini-2.0-flash", "llama-3.3-70b-versatile"]
                Defaults to ["gemini-2.0-flash"] if GEMINI_API_KEY is set,
                or ["llama-3.3-70b-versatile"] if GROQ_API_KEY is set.

    Returns:
        dict mapping model name → list of instance dicts
    """
    if models is None:
        models = []
        if os.environ.get("GEMINI_API_KEY"):
            models.append("gemini-2.0-flash")
        if os.environ.get("GROQ_API_KEY"):
            models.append("llama-3.3-70b-versatile")
        if not models:
            models = ["gemini-2.0-flash"]

    results = {}
    for m in models:
        results[m] = run_llm_baseline(
            data, feature_type,
            candidate_seeds=candidate_seeds,
            max_seeds=max_seeds,
            llm_model=m,
        )
    return results
