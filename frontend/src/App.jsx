import React, { useState, useEffect } from "react";
import Navbar from "./components/Navbar";
import AIDetectorTool from "./components/AIDetectorTool";
import GrammarCheckerTool from "./components/GrammarCheckerTool";
import CombinedTool from "./components/CombinedTool";
import { fetchHealth } from "./api";
import { Shield, Sparkles, CheckCircle2, FileCode, Layers, Activity } from "lucide-react";

export default function App() {
  const [activeTab, setActiveTab] = useState("ai-detector"); // "ai-detector" | "grammar" | "combined"
  const [health, setHealth] = useState(null);

  const checkBackendHealth = async () => {
    const data = await fetchHealth();
    setHealth(data);
  };

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 25000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-indigo-500 selection:text-white">
      {/* Top Navbar */}
      <Navbar
        health={health}
        onRefreshHealth={checkBackendHealth}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Active Tool View */}
        {activeTab === "ai-detector" && <AIDetectorTool />}
        {activeTab === "grammar" && <GrammarCheckerTool />}
        {activeTab === "combined" && <CombinedTool />}

        {/* System Capabilities & Model Cards */}
        <section className="pt-8 border-t border-slate-800/80">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5 text-xs text-slate-400">
            <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 space-y-2">
              <div className="flex items-center space-x-2 text-indigo-400 font-semibold">
                <Layers className="w-4 h-4" />
                <span className="text-slate-200">Module A: Detection</span>
              </div>
              <p>
                TF-IDF hybrid n-gram vectorizer (50,000 features) + Logistic Regression with 96.34% test accuracy and 0.995 ROC-AUC.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 space-y-2">
              <div className="flex items-center space-x-2 text-purple-400 font-semibold">
                <FileCode className="w-4 h-4" />
                <span className="text-slate-200">Module B: Grammar</span>
              </div>
              <p>
                T5 Seq2Seq neural error correction evaluated on CoEdit & JFLEG benchmarks achieving 89.4 BLEU and 94.0 chrF.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 space-y-2">
              <div className="flex items-center space-x-2 text-emerald-400 font-semibold">
                <Activity className="w-4 h-4" />
                <span className="text-slate-200">Production REST API</span>
              </div>
              <p>
                FastAPI backend with schema validation, lifecycle memory preloading, sentence-level token diffs, and CORS.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-900/40 py-6 text-center text-xs text-slate-500">
        <p>AI Content Analysis System • College PBL Project (Phase 2 & Full-Stack Integration)</p>
      </footer>
    </div>
  );
}
