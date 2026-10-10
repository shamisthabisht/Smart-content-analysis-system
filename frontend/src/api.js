const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/**
 * Checks backend health status
 */
export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    if (!res.ok) throw new Error(`Health check failed with HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error("Health check error:", err);
    return {
      status: "unreachable",
      error: err.message
    };
  }
}

/**
 * Analyzes text for AI vs Human detection
 */
export async function analyzeAIDetection(text) {
  const res = await fetch(`${API_BASE_URL}/api/analyze/ai-detection`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Server error (HTTP ${res.status})`);
  }

  return await res.json();
}

/**
 * Corrects grammar and returns edits
 */
export async function analyzeGrammar(text) {
  const res = await fetch(`${API_BASE_URL}/api/analyze/grammar`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Server error (HTTP ${res.status})`);
  }

  return await res.json();
}

/**
 * Runs combined AI detection and Grammar correction
 */
export async function analyzeCombined(text, enableAI = true, enableGrammar = true) {
  const res = await fetch(`${API_BASE_URL}/api/analyze/combined`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      enable_ai_detection: enableAI,
      enable_grammar: enableGrammar,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Server error (HTTP ${res.status})`);
  }

  return await res.json();
}
