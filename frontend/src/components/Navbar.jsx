import React from "react";
import { Sparkles, ShieldCheck, Cpu, RefreshCw, CheckCircle2, AlertCircle } from "lucide-react";

export default function Navbar({ health, onRefreshHealth, activeTab, setActiveTab }) {
  const isHealthy = health?.status === "healthy";

  return (
    <header className="sticky top-0 z-50 backdrop-blur-md bg-slate-900/80 border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-500 flex items-center justify-center shadow-lg shadow-indigo-500/25">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                Veritas AI
              </h1>
              <p className="text-xs text-slate-400">AI Content & Grammar Intelligence</p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex space-x-1 bg-slate-800/80 p-1 rounded-xl border border-slate-700/60">
            <button
              onClick={() => setActiveTab("ai-detector")}
              className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === "ai-detector"
                  ? "bg-indigo-600 text-white shadow-md shadow-indigo-500/30"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-700/50"
              }`}
            >
              <Cpu className="w-4 h-4" />
              <span>AI Detector</span>
            </button>
            <button
              onClick={() => setActiveTab("grammar")}
              className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === "grammar"
                  ? "bg-purple-600 text-white shadow-md shadow-purple-500/30"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-700/50"
              }`}
            >
              <ShieldCheck className="w-4 h-4" />
              <span>Grammar & Diff</span>
            </button>
            <button
              onClick={() => setActiveTab("combined")}
              className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === "combined"
                  ? "bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-md"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-700/50"
              }`}
            >
              <Sparkles className="w-4 h-4" />
              <span>Full Analysis</span>
            </button>
          </nav>

          {/* Backend Status Indicator */}
          <div className="flex items-center space-x-2">
            <div
              className={`flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-medium border ${
                isHealthy
                  ? "bg-emerald-950/60 border-emerald-800/60 text-emerald-400"
                  : "bg-amber-950/60 border-amber-800/60 text-amber-400"
              }`}
              title={isHealthy ? `Connected (Device: ${health?.device})` : "Backend offline or connecting..."}
            >
              <span className={`w-2 h-2 rounded-full ${isHealthy ? "bg-emerald-400 animate-pulse" : "bg-amber-400"}`} />
              <span className="hidden sm:inline">{isHealthy ? "API Online" : "Connecting..."}</span>
            </div>
            <button
              onClick={onRefreshHealth}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition"
              title="Refresh API Status"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
