import React, { useState } from "react";
import { Sparkles, Bot, ShieldCheck, Zap, Copy, Check, Trash2, AlertCircle } from "lucide-react";
import { analyzeCombined } from "../api";

export default function CombinedTool() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  const sampleCombined = "In this research paper, we demonstrates novel methodology for deep neural network training. It produce very high accuracy and outperform prior baseline.";

  const handleAnalyze = async () => {
    if (!text.trim() || text.trim().length < 5) {
      setError("Please enter at least 5 characters for combined analysis.");
      return;
    }

    setError(null);
    setLoading(true);
    setResult(null);

    try {
      const data = await analyzeCombined(text, true, true);
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to analyze text.");
    } finally {
      setLoading(false);
    }
  };

  const handleCopyClean = () => {
    const clean = result?.grammar_analysis?.corrected_text || text;
    navigator.clipboard.writeText(clean);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900/60 p-5 rounded-2xl border border-slate-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <Sparkles className="w-6 h-6 text-indigo-400" />
            <h2 className="text-xl font-bold text-white">Combined Content Intelligence</h2>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-gradient-to-r from-indigo-500/20 to-purple-500/20 text-indigo-300 border border-indigo-500/30">
              Unified Pipeline
            </span>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Simultaneously evaluates AI authorship probabilities and extracts sentence-level grammar repairs in one pass.
          </p>
        </div>

        <button
          onClick={() => {
            setText(sampleCombined);
            setResult(null);
          }}
          className="px-3 py-1.5 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
        >
          Load Combined Sample
        </button>
      </div>

      {/* Editor & Execution */}
      <div className="bg-slate-900/70 rounded-2xl border border-slate-800 p-4 shadow-xl space-y-3">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste or type content here for end-to-end AI detection and grammar correction..."
          rows={5}
          className="w-full bg-transparent text-slate-100 placeholder-slate-500 resize-none focus:outline-none text-sm sm:text-base leading-relaxed"
        />

        <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <button
              onClick={() => {
                setText("");
                setResult(null);
              }}
              disabled={!text}
              className="p-2 rounded-lg bg-slate-800 text-slate-400 hover:text-rose-400 transition disabled:opacity-40"
              title="Clear input"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>

          <button
            onClick={handleAnalyze}
            disabled={loading || !text.trim()}
            className="flex items-center space-x-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-medium text-sm shadow-lg shadow-indigo-600/30 transition disabled:opacity-50 cursor-pointer"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Running Pipeline...</span>
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                <span>Run Unified Analysis</span>
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/50 border border-rose-800/60 text-rose-300 flex items-start space-x-3 text-sm">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* Unified Results Grid */}
      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Card 1: AI Detection Results */}
          <div className="bg-slate-900/70 rounded-2xl border border-slate-800 p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center space-x-2">
                <Bot className="w-5 h-5 text-indigo-400" />
                <h3 className="font-semibold text-white">AI Detection Breakdown</h3>
              </div>
              <span
                className={`text-xs font-bold px-2.5 py-0.5 rounded-full ${
                  result.ai_detection?.is_short_text
                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                    : result.ai_detection?.is_ai
                    ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                    : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                }`}
              >
                {result.ai_detection?.confidence_level}
              </span>
            </div>

            {result.ai_detection?.is_short_text && (
              <div className="p-2.5 rounded-lg bg-amber-950/40 border border-amber-800/60 text-amber-300 text-xs">
                ⚠️ <strong>Short text notice:</strong> Input is under 50 words; n-gram statistics carry high variance.
              </div>
            )}

            {result.ai_detection && (
              <div className="space-y-4 text-xs">
                <div className="flex justify-between text-slate-300">
                  <span>AI Probability:</span>
                  <span className="font-bold text-rose-400">
                    {(result.ai_detection.ai_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span>Human Probability:</span>
                  <span className="font-bold text-emerald-400">
                    {(result.ai_detection.human_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden flex">
                  <div
                    style={{ width: `${result.ai_detection.human_probability * 100}%` }}
                    className="bg-emerald-500 h-full"
                  />
                  <div
                    style={{ width: `${result.ai_detection.ai_probability * 100}%` }}
                    className="bg-rose-500 h-full"
                  />
                </div>
                <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800 space-y-1.5 text-slate-400">
                  <div>Confidence Rating: <strong className="text-slate-200">{result.ai_detection.confidence_level} ({(result.ai_detection.confidence_score * 100).toFixed(1)}%)</strong></div>
                  <div>Word Count: <strong className="text-slate-200">{result.statistics?.word_count}</strong> | Characters: <strong className="text-slate-200">{result.statistics?.char_count}</strong></div>
                </div>
              </div>
            )}
          </div>

          {/* Card 2: Grammar Correction Results */}
          <div className="bg-slate-900/70 rounded-2xl border border-slate-800 p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center space-x-2">
                <ShieldCheck className="w-5 h-5 text-purple-400" />
                <h3 className="font-semibold text-white">Grammar Repairs</h3>
              </div>
              <span
                className={`text-xs font-bold px-2.5 py-0.5 rounded-full ${
                  result.grammar_analysis?.has_errors
                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                    : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                }`}
              >
                {result.grammar_analysis?.total_errors || 0} Corrections
              </span>
            </div>

            {result.grammar_analysis && (
              <div className="space-y-3">
                <div className="p-3.5 bg-slate-950/70 rounded-xl border border-slate-800 text-sm leading-relaxed text-slate-100">
                  {result.grammar_analysis.corrected_text}
                </div>

                <div className="flex items-center justify-between pt-2">
                  <span className="text-xs text-slate-400">
                    Total execution time: {result.execution_time_seconds}s
                  </span>
                  <button
                    onClick={handleCopyClean}
                    className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition cursor-pointer"
                  >
                    {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>{copied ? "Copied!" : "Copy Clean Text"}</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
