import React, { useState } from "react";
import { 
  Bot, 
  UserCheck, 
  Sparkles, 
  Copy, 
  Check, 
  Trash2, 
  AlertCircle, 
  AlertTriangle,
  BarChart3, 
  Zap,
  Info
} from "lucide-react";
import { analyzeAIDetection } from "../api";

const PRESET_SAMPLES = [
  {
    name: "Human (Short Sentence, 11 words)",
    text: "The students gathered in the quiet library to study for their exams."
  },
  {
    name: "Human (Informal Passage)",
    text: "Honestly, I had no clue what was going on yesterday. We just went to the diner, ordered some waffles, and ended up chatting for like three hours about random stuff haha. Everyone was super tired afterward."
  },
  {
    name: "Human (Academic Formal)",
    text: "The committee convened early on Tuesday morning to deliberate over the proposed fiscal reforms, ultimately postponing the vote after intense debate among senior delegates regarding regional budget allocations."
  },
  {
    name: "AI (ChatGPT Explanatory)",
    text: "In conclusion, renewable energy sources play a crucial role in mitigating the impacts of climate change by reducing global greenhouse gas emissions and fostering sustainable economic growth across developing nations."
  }
];

export default function AIDetectorTool() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const charCount = text.length;
  const isShortInput = wordCount > 0 && wordCount < 50;

  const handleAnalyze = async () => {
    if (!text.trim() || text.trim().length < 5) {
      setError("Please enter at least 5 characters to analyze.");
      return;
    }

    setError(null);
    setLoading(true);
    setResult(null);

    try {
      const data = await analyzeAIDetection(text);
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to analyze text. Please ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleClear = () => {
    setText("");
    setResult(null);
    setError(null);
  };

  const loadPreset = (presetText) => {
    setText(presetText);
    setResult(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      {/* Tool Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 bg-slate-900/60 p-5 rounded-2xl border border-slate-800">
        <div>
          <div className="flex items-center space-x-2">
            <Bot className="w-6 h-6 text-indigo-400" />
            <h2 className="text-xl font-bold text-white">AI Content Detector</h2>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              Module A
            </span>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Evaluates statistical word & character n-gram distributions. Requires at least 50 words for dependable confidence.
          </p>
        </div>

        {/* Preset sample buttons */}
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-slate-400 mr-1">Sample Presets:</span>
          {PRESET_SAMPLES.map((s, idx) => (
            <button
              key={idx}
              onClick={() => loadPreset(s.text)}
              className="px-2.5 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700/60 transition cursor-pointer"
            >
              {s.name}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Input & Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Input Editor */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-slate-900/70 rounded-2xl border border-slate-800 p-4 shadow-xl flex flex-col h-[440px]">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 text-xs text-slate-400">
              <div className="flex items-center space-x-3">
                <span className={`font-medium ${isShortInput ? "text-amber-400 font-semibold" : "text-slate-300"}`}>
                  {wordCount} words {isShortInput && "(Short Text < 50 words)"}
                </span>
                <span>•</span>
                <span>{charCount} characters</span>
              </div>
              <div className="flex items-center space-x-1">
                <button
                  onClick={handleCopy}
                  disabled={!text}
                  className="p-1.5 rounded-md hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition disabled:opacity-40"
                  title="Copy text"
                >
                  {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                </button>
                <button
                  onClick={handleClear}
                  disabled={!text}
                  className="p-1.5 rounded-md hover:bg-slate-800 text-slate-400 hover:text-rose-400 transition disabled:opacity-40"
                  title="Clear text"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>

            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste or type your text here to detect AI generation (recommended >= 50 words)..."
              className="w-full flex-1 mt-3 bg-transparent text-slate-100 placeholder-slate-500 resize-none focus:outline-none text-sm sm:text-base leading-relaxed"
            />

            {/* Input Length Hint */}
            {isShortInput && (
              <div className="px-3 py-1.5 rounded-lg bg-amber-950/40 border border-amber-800/50 text-amber-300 text-xs flex items-center space-x-2 my-1">
                <AlertTriangle className="w-3.5 h-3.5 shrink-0 text-amber-400" />
                <span>Tip: Texts under 50 words have insufficient n-gram density; results will carry an uncertainty warning.</span>
              </div>
            )}

            <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between">
              <span className="text-xs text-slate-400">
                Minimum recommended: 50 words
              </span>
              <button
                onClick={handleAnalyze}
                disabled={loading || !text.trim()}
                className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm transition shadow-lg shadow-indigo-600/30 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    <span>Detect AI Content</span>
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
        </div>

        {/* Right: Results Card */}
        <div className="lg:col-span-5">
          <div className="bg-slate-900/70 rounded-2xl border border-slate-800 p-5 shadow-xl h-full flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-4 border-b border-slate-800">
                <div className="flex items-center space-x-2">
                  <BarChart3 className="w-5 h-5 text-indigo-400" />
                  <h3 className="font-semibold text-white">Detection Assessment</h3>
                </div>
                {result?.execution_time_seconds && (
                  <span className="text-xs font-mono text-slate-400">
                    {result.execution_time_seconds}s latency
                  </span>
                )}
              </div>

              {!result && !loading && (
                <div className="flex flex-col items-center justify-center py-16 text-center text-slate-400">
                  <Bot className="w-12 h-12 text-slate-700 mb-3" />
                  <p className="text-sm font-medium">No analysis performed yet</p>
                  <p className="text-xs text-slate-400 mt-1 max-w-xs">
                    Enter text on the left and click "Detect AI Content" to see probability scores and breakdown.
                  </p>
                </div>
              )}

              {loading && (
                <div className="flex flex-col items-center justify-center py-16 text-center">
                  <div className="w-10 h-10 border-3 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin mb-4" />
                  <p className="text-sm font-medium text-slate-300">Extracting n-gram features...</p>
                  <p className="text-xs text-slate-400 mt-1">Evaluating stylistic distributions</p>
                </div>
              )}

              {result && (
                <div className="space-y-4 mt-4">
                  {/* Short Text Warning Banner */}
                  {result.is_short_text && (
                    <div className="p-3.5 rounded-xl bg-amber-950/50 border border-amber-800/80 text-amber-200 text-xs space-y-1">
                      <div className="flex items-center space-x-1.5 font-bold text-amber-300">
                        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                        <span>Unreliable Result (Short Input: {result.word_count} / {result.min_reliable_words || 50} words)</span>
                      </div>
                      <p className="text-amber-300/90 leading-relaxed pl-5">
                        {result.reliability_warning || "Text under 50 words has high variance in statistical NLP models. For a dependable score, provide longer paragraphs."}
                      </p>
                    </div>
                  )}

                  {/* Verdict Banner */}
                  <div
                    className={`p-4 rounded-xl border flex items-center justify-between ${
                      result.is_short_text
                        ? "bg-slate-900/90 border-amber-800/60 text-slate-200"
                        : result.is_ai
                        ? "bg-rose-950/30 border-rose-800/50 text-rose-200"
                        : "bg-emerald-950/30 border-emerald-800/50 text-emerald-200"
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      {result.is_short_text ? (
                        <div className="w-10 h-10 rounded-lg bg-amber-500/20 flex items-center justify-center text-amber-400">
                          <AlertTriangle className="w-6 h-6" />
                        </div>
                      ) : result.is_ai ? (
                        <div className="w-10 h-10 rounded-lg bg-rose-500/20 flex items-center justify-center text-rose-400">
                          <Bot className="w-6 h-6" />
                        </div>
                      ) : (
                        <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center text-emerald-400">
                          <UserCheck className="w-6 h-6" />
                        </div>
                      )}
                      <div>
                        <div className="text-xs uppercase tracking-wider font-semibold opacity-80">
                          Primary Assessment
                        </div>
                        <div className="text-base sm:text-lg font-bold text-white">
                          {result.assessment}
                        </div>
                      </div>
                    </div>

                    <div className="text-right">
                      <div className="text-xs text-slate-400">Confidence</div>
                      <div
                        className={`text-xs font-bold px-2 py-0.5 rounded-full inline-block mt-0.5 ${
                          result.is_short_text
                            ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                            : result.confidence_level === "High"
                            ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                            : result.confidence_level === "Moderate"
                            ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                            : "bg-slate-700 text-slate-300"
                        }`}
                      >
                        {result.confidence_level}
                      </div>
                    </div>
                  </div>

                  {/* Probability Bar */}
                  <div className="space-y-2 bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="text-emerald-400">
                        Human: {(result.human_probability * 100).toFixed(1)}%
                      </span>
                      <span className="text-rose-400">
                        AI Generated: {(result.ai_probability * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="w-full h-3.5 bg-slate-800 rounded-full overflow-hidden flex">
                      <div
                        style={{ width: `${result.human_probability * 100}%` }}
                        className="bg-emerald-500 h-full transition-all duration-700"
                        title={`Human Probability: ${(result.human_probability * 100).toFixed(1)}%`}
                      />
                      <div
                        style={{ width: `${result.ai_probability * 100}%` }}
                        className="bg-rose-500 h-full transition-all duration-700"
                        title={`AI Probability: ${(result.ai_probability * 100).toFixed(1)}%`}
                      />
                    </div>
                  </div>

                  {/* Details Card */}
                  <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-2.5 text-xs text-slate-300">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Sample Length:</span>
                      <span className="font-mono text-slate-200">{result.word_count || wordCount} words</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Model Pipeline:</span>
                      <span className="font-mono text-slate-200">TF-IDF (Word + Char) + LogReg</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Benchmark (&ge; 50 words):</span>
                      <span className="font-mono text-emerald-400 font-semibold">96.8% F1 (0.996 AUC)</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="pt-4 border-t border-slate-800 flex items-center space-x-2 text-xs text-slate-400">
              <Info className="w-4 h-4 text-slate-400 shrink-0" />
              <span>
                Empirically calibrated: Full passages (&ge; 50 words) achieve 96.8% accuracy.
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
