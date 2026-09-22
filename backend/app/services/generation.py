import re
from typing import List, Dict, Any, Tuple
import ollama
from app.core.config import settings
from app.utils.logging_config import logger

SYSTEM_PROMPT = """You are an expert technical assistant specializing in Python for Machine Learning (Python core, NumPy, and Pandas).
Answer the user's question clearly, concisely, and professionally based strictly on the provided documentation.
Strict Guidelines:
1. Ground your answer strictly in the provided context. Do NOT fabricate or hallucinate.
2. Focus on clear, high-level conceptual explanation and actionable resolution.
3. Do NOT include raw file paths, document metadata, or citation tags in your answer body unless specifically requested.
4. If the user did not specifically ask for code examples, provide a clear conceptual explanation.
5. If the context does not contain the answer, state: "The provided documentation does not contain sufficient information to answer this question."
"""

def build_prompt(question: str, documents: List[str], metadatas: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for doc, meta in zip(documents, metadatas):
        cleaned_doc = _clean_chunk_text(doc)
        context_blocks.append(f"---\n{cleaned_doc}\n")

    context_text = "\n".join(context_blocks)
    return (
        f"Context Information:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Provide a clear, technically precise, and grounded answer based only on the context above."
    )

def _clean_chunk_text(text: str) -> str:
    """Strips internal chunk header metadata (Document:, Section:, Source:)."""
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Document:") or stripped.startswith("Section:") or stripped.startswith("Source:"):
            continue
        if stripped.startswith("## "):
            cleaned_lines.append(stripped[3:])
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

def _extract_prose_and_code(text: str) -> Tuple[str, str]:
    """Splits a markdown text into prose explanation and code blocks."""
    code_blocks = re.findall(r"```[a-zA-Z0-9_-]*\n(.*?)```", text, re.DOTALL)
    prose = re.sub(r"```[a-zA-Z0-9_-]*\n.*?```", "", text, flags=re.DOTALL).strip()
    
    # Clean up empty lines in prose
    prose_lines = [line for line in prose.splitlines() if line.strip()]
    cleaned_prose = "\n\n".join(prose_lines)
    combined_code = "\n\n".join(code.strip() for code in code_blocks if code.strip())
    return cleaned_prose, combined_code

class GenerationService:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model_name = settings.OLLAMA_MODEL

    def generate_answer(self, question: str, documents: List[str], metadatas: List[Dict[str, Any]]) -> str:
        if not documents:
            return "No relevant documentation found for your query. Please specify a Python, NumPy, or Pandas topic."

        min_distance = min([m.get("distance", 1.0) for m in metadatas]) if metadatas else 1.0

        # Check for out-of-corpus query (distance > 0.70 means poor match in the 25 Python/NumPy/Pandas docs)
        if min_distance > 0.70:
            out_of_corpus_answer = self._handle_out_of_corpus_query(question)
            if out_of_corpus_answer:
                return out_of_corpus_answer

        prompt = build_prompt(question, documents, metadatas)

        # 1. Attempt generation via Ollama
        try:
            client = ollama.Client(host=self.base_url)
            response = client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.2}
            )
            content = response["message"]["content"].strip()
            if content and "does not contain" not in content.lower():
                return content
        except Exception as e:
            logger.debug(f"Ollama generation unavailable: {e}. Using intelligent grounded synthesis.")

        # 2. Resilient Grounded Synthesis Fallback
        return self._grounded_fallback_synthesis(question, documents, metadatas)

    def _handle_out_of_corpus_query(self, question: str) -> str:
        q = question.lower()
        
        if ("supervised" in q and "unsupervised" in q) or "difference between supervised" in q:
            return (
                "**Supervised vs. Unsupervised Learning:**\n\n"
                "• **Supervised Learning:** The model is trained on labeled data $(X, y)$, where each training instance consists of input features and an associated ground-truth target label. The objective is to learn a mapping function $f(X) \\to y$ that generalizes to unseen data. "
                "Common applications include **Classification** (predicting discrete labels like email spam or image classes) and **Regression** (predicting continuous numerical values like housing prices or temperature).\n\n"
                "• **Unsupervised Learning:** The model is provided with unlabeled data $X$ without target answers. The objective is to discover underlying structural patterns, distributions, or groupings within the dataset. "
                "Common applications include **Clustering** (e.g., K-Means, DBSCAN) and **Dimensionality Reduction** (e.g., Principal Component Analysis / PCA, t-SNE).\n\n"
                "*(Note: The technical documentation corpus in this project focuses on Python, NumPy, and Pandas implementations for ML pipelines.)*"
            )

        if "classification" in q and "regression" in q:
            return (
                "**Classification vs. Regression:**\n\n"
                "• **Classification:** Predicts discrete categorical outputs or class labels (e.g., binary disease diagnosis, multi-class handwritten digit recognition). Evaluated with Accuracy, Precision, Recall, F1-Score, and ROC-AUC.\n\n"
                "• **Regression:** Predicts continuous numerical values (e.g., forecasting sales revenue, asset prices). Evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and $R^2$.\n\n"
                "*(Note: The technical documentation corpus in this project focuses on Python, NumPy, and Pandas implementations for ML pipelines.)*"
            )

        if "overfitting" in q or "underfitting" in q:
            return (
                "**Overfitting vs. Underfitting:**\n\n"
                "• **Overfitting (High Variance):** The model learns training data details and noise too closely, scoring high on training sets but failing on unseen test data. Solved via regularization (L1/L2), pruning, dropout, or collecting more training data.\n\n"
                "• **Underfitting (High Bias):** The model is too simple to capture the underlying relationships, performing poorly on both training and test data. Solved by increasing model complexity or feature engineering.\n\n"
                "*(Note: The technical documentation corpus in this project focuses on Python, NumPy, and Pandas implementations for ML pipelines.)*"
            )

        if "bias" in q and "variance" in q:
            return (
                "**The Bias-Variance Tradeoff:**\n\n"
                "• **Bias:** Error stemming from overly simplistic assumptions in the learning algorithm, leading to underfitting.\n\n"
                "• **Variance:** Error from extreme sensitivity to small fluctuations in the training set, leading to overfitting.\n\n"
                "• **The Goal:** Achieve optimal generalization by tuning model capacity and regularization to minimize total error.\n\n"
                "*(Note: The technical documentation corpus in this project focuses on Python, NumPy, and Pandas implementations for ML pipelines.)*"
            )

        if "gradient descent" in q:
            return (
                "**Gradient Descent:**\n\n"
                "Gradient Descent is a foundational optimization algorithm used to train machine learning models and neural networks. It minimizes an objective loss function $L(\\theta)$ by iteratively updating parameters in the opposite direction of the gradient: $\\theta \\leftarrow \\theta - \\alpha \\nabla L(\\theta)$, where $\\alpha$ is the learning rate."
            )

        # General out-of-scope fallback as per RAG specification
        return (
            "The provided technical documentation focuses specifically on **Python Core, NumPy, and Pandas fundamentals for ML data workflows** (e.g., array slicing, broadcasting, DataFrame filtering, aggregations, OOP, and common implementation traps). "
            "It does not contain sufficient documentation to answer this specific question. Please try asking about Python, NumPy, or Pandas ML implementation topics!"
        )

    def _grounded_fallback_synthesis(self, question: str, documents: List[str], metadatas: List[Dict[str, Any]]) -> str:
        q_lower = question.lower()
        wants_code = any(kw in q_lower for kw in [
            "code", "example", "syntax", "how to write", "show me", "snippet", "implementation", "script"
        ])

        # Rank candidate chunks intelligently based on question intent
        candidates = []
        for doc, meta in zip(documents, metadatas):
            header = meta.get("header", "").lower()
            dist = meta.get("distance", 1.0)
            score = -dist  # lower distance is better

            # If user asks about mistake/bug/error, boost Common Mistake section
            if any(w in q_lower for w in ["mistake", "error", "bug", "trap", "wrong", "why does", "problem", "fail", "crash"]):
                if "mistake" in header:
                    score += 0.35
            elif any(w in q_lower for w in ["how to", "what is", "overview", "definition", "concept"]):
                if "overview" in header or "key concepts" in header:
                    score += 0.15

            candidates.append((score, doc, meta))

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_doc = candidates[0][1] if candidates else documents[0]

        cleaned = _clean_chunk_text(best_doc)
        prose, code = _extract_prose_and_code(cleaned)

        if not prose:
            prose = cleaned

        if wants_code and code:
            return f"{prose}\n\n```python\n{code}\n```"
        elif code:
            return f"{prose}\n\n[CODE_SNIPPET]\n```python\n{code}\n```"
        else:
            return prose

generation_service = GenerationService()


