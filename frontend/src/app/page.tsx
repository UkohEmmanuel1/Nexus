"use client";

import React, { useState, useEffect, useRef } from "react";

interface ApiKeyItem {
  key: string;
  name: string;
  created_at: number;
  status: string;
}

interface Message {
  role: "system" | "user" | "assistant";
  content: string;
  thinking?: string;
  error?: boolean;
}

const iconProps = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

function ChatIcon({ className = "" }: { className?: string }) {
  return (
    <svg {...iconProps} className={className}>
      <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
    </svg>
  );
}

function KeyIcon({ className = "" }: { className?: string }) {
  return (
    <svg {...iconProps} className={className}>
      <circle cx="7.5" cy="15.5" r="5.5" />
      <path d="m21 2-2 2m-7.61 7.61L15.5 7.5m0 0 3 3L22 7l-3-3m-3.5 3.5L19 4" />
    </svg>
  );
}

function SparkIcon({ className = "" }: { className?: string }) {
  return (
    <svg {...iconProps} className={className}>
      <path d="M13 2 3 14h7l-1 8 10-12h-7l1-8z" />
    </svg>
  );
}

function ChevronIcon({ className = "" }: { className?: string }) {
  return (
    <svg {...iconProps} className={className}>
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

function SendIcon({ className = "" }: { className?: string }) {
  return (
    <svg {...iconProps} className={className}>
      <path d="M12 19V5" />
      <path d="m5 12 7-7 7 7" />
    </svg>
  );
}

function WarningIcon({ className = "" }: { className?: string }) {
  return (
    <svg {...iconProps} className={className}>
      <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <path d="M12 9v4" />
      <path d="M12 17h.01" />
    </svg>
  );
}

function OrbitLogo({ className = "" }: { className?: string }) {
  return (
    <svg width={30} height={30} viewBox="0 0 30 30" fill="none" className={className}>
      <defs>
        <linearGradient id="nexusGrad" x1="0" y1="0" x2="30" y2="30" gradientUnits="userSpaceOnUse">
          <stop stopColor="#A78BFA" />
          <stop offset="1" stopColor="#60A5FA" />
        </linearGradient>
      </defs>
      <rect x="1.5" y="1.5" width="27" height="27" rx="8.5" fill="url(#nexusGrad)" fillOpacity="0.14" stroke="url(#nexusGrad)" strokeWidth="1.2" />
      <ellipse cx="15" cy="15" rx="8.4" ry="3.6" stroke="url(#nexusGrad)" strokeWidth="1.4" />
      <circle cx="15" cy="15" r="2.7" fill="url(#nexusGrad)" />
      <circle cx="22.4" cy="15" r="1.5" fill="#0B0E14" stroke="url(#nexusGrad)" strokeWidth="1.2" />
    </svg>
  );
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<"chat" | "keys">("chat");
  const [serverUrl, setServerUrl] = useState("http://localhost:8000");
  const [adminKey, setAdminKey] = useState("");
  const [keyName, setKeyName] = useState("");
  const [newKey, setNewKey] = useState("");
  const [keysList, setKeysList] = useState<ApiKeyItem[]>([]);
  const [codeTab, setCodeTab] = useState<"curl" | "python" | "nodejs">("curl");

  // Chat State
  const [chatInput, setChatInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "system",
      content:
        "Welcome to the Nexus Local LLM Chat Console. Ask me anything, or toggle Deep Think to view my reasoning steps.",
    },
  ]);
  const [deepThinkEnabled, setDeepThinkEnabled] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [collapsedThoughts, setCollapsedThoughts] = useState<{ [key: number]: boolean }>({});

  const chatEndRef = useRef<HTMLDivElement>(null);

  // Load configuration from localStorage
  useEffect(() => {
    if (typeof window !== "undefined") {
      const savedAdmin = localStorage.getItem("nexus_admin_key") || "";
      const savedServer = localStorage.getItem("nexus_server_url") || "http://localhost:8000";
      setAdminKey(savedAdmin);
      setServerUrl(savedServer);
    }
  }, []);

  // Scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isGenerating]);

  // Fetch keys list when switching to keys tab or when admin key changes
  useEffect(() => {
    if (activeTab === "keys") {
      fetchKeys();
    }
  }, [activeTab, adminKey, serverUrl]);

  const saveConfig = (admin: string, url: string) => {
    localStorage.setItem("nexus_admin_key", admin);
    localStorage.setItem("nexus_server_url", url);
  };

  // Fetch API keys from server
  const fetchKeys = async () => {
    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (adminKey) {
        headers["X-API-Key"] = adminKey;
      }
      const res = await fetch(`${serverUrl}/v1/keys`, { headers });
      if (res.ok) {
        const data = await res.json();
        setKeysList(data);
      } else {
        console.error("Failed to fetch keys:", res.statusText);
      }
    } catch (err) {
      console.error("Error fetching keys:", err);
    }
  };

  // Generate a new key
  const handleGenerateKey = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!keyName.trim()) return;

    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (adminKey) {
        headers["Authorization"] = `Bearer ${adminKey}`;
      }

      const res = await fetch(`${serverUrl}/v1/keys/generate`, {
        method: "POST",
        headers,
        body: JSON.stringify({ name: keyName }),
      });

      if (res.ok) {
        const data = await res.json();
        setNewKey(data.key);
        localStorage.setItem("nexus_active_user_key", data.key);
        setKeyName("");
        fetchKeys();
      } else {
        const err = await res.json();
        alert(`Error: ${err.detail || res.statusText}`);
      }
    } catch (err: any) {
      alert(`Request failed: ${err.message}`);
    }
  };

  // Revoke key
  const handleRevokeKey = async (key: string) => {
    if (!confirm("Are you sure you want to revoke this key? This cannot be undone.")) return;

    // Use current active key (or admin key) for authentication
    const authKey = localStorage.getItem("nexus_active_user_key") || adminKey;

    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (authKey) {
        headers["X-API-Key"] = authKey;
      }

      const res = await fetch(`${serverUrl}/v1/keys/revoke`, {
        method: "POST",
        headers,
        body: JSON.stringify({ key }),
      });

      if (res.ok) {
        fetchKeys();
      } else {
        const err = await res.json();
        alert(`Error: ${err.detail || res.statusText}`);
      }
    } catch (err: any) {
      alert(`Request failed: ${err.message}`);
    }
  };

  // Send Chat message
  const handleSendMessage = async () => {
    if (!chatInput.trim() || isGenerating) return;

    const userMessageText = chatInput.trim();
    setChatInput("");

    // Append user message
    setMessages((prev) => [...prev, { role: "user", content: userMessageText }]);
    setIsGenerating(true);

    const activeUserKey = localStorage.getItem("nexus_active_user_key") || adminKey;

    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (activeUserKey) {
        headers["X-API-Key"] = activeUserKey;
      }

      const res = await fetch(`${serverUrl}/chat`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          message: userMessageText,
          thinking: deepThinkEnabled,
        }),
      });

      if (!res.ok) {
        let errMsg = "An error occurred";
        if (res.status === 401) {
          errMsg = "Unauthorized. Please configure your API key or Admin key in the API Keys tab.";
        } else if (res.status === 429) {
          errMsg = "Rate limit exceeded. Please wait a moment.";
        } else {
          const err = await res.json().catch(() => ({}));
          errMsg = err.detail || res.statusText;
        }
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: errMsg, error: true },
        ]);
      } else {
        const data = await res.json();
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.response,
            thinking: data.thinking || undefined,
          },
        ]);
      }
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Network error: ${err.message}`, error: true },
      ]);
    } finally {
      setIsGenerating(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    alert("Copied to clipboard!");
  };

  const toggleThought = (idx: number) => {
    setCollapsedThoughts((prev) => ({
      ...prev,
      [idx]: !prev[idx],
    }));
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#07090D] text-slate-100 font-sans">
      {/* Sidebar */}
      <aside className="w-64 bg-white/[0.02] border-r border-white/[0.06] backdrop-blur-xl flex flex-col p-6 flex-shrink-0">
        <div className="flex items-center gap-3 mb-10">
          <div className="relative flex-shrink-0">
            <div className="absolute -inset-1.5 rounded-2xl bg-gradient-to-br from-violet-500/40 to-blue-500/40 blur-md opacity-60" />
            <OrbitLogo className="relative" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-violet-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Nexus AI
            </h1>
            <p className="text-[10px] uppercase tracking-[0.22em] text-slate-500 mt-0.5">
              Local LLM Console
            </p>
          </div>
        </div>

        <nav className="flex flex-col gap-2 flex-grow">
          <button
            onClick={() => setActiveTab("chat")}
            className={`relative flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 ${
              activeTab === "chat"
                ? "text-white bg-gradient-to-r from-violet-500/[0.16] to-blue-500/[0.08] border border-white/10 shadow-[0_8px_24px_rgba(139,92,246,0.15)]"
                : "text-slate-400 hover:text-white hover:bg-white/5 border border-transparent"
            }`}
          >
            {activeTab === "chat" && (
              <span className="absolute left-0 top-1/2 -translate-y-1/2 h-6 w-[3px] rounded-r-full bg-gradient-to-b from-violet-400 to-blue-400" />
            )}
            <ChatIcon className={`w-[18px] h-[18px] ${activeTab === "chat" ? "text-violet-300" : ""}`} />
            Chat Console
          </button>
          <button
            onClick={() => setActiveTab("keys")}
            className={`relative flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 ${
              activeTab === "keys"
                ? "text-white bg-gradient-to-r from-violet-500/[0.16] to-blue-500/[0.08] border border-white/10 shadow-[0_8px_24px_rgba(139,92,246,0.15)]"
                : "text-slate-400 hover:text-white hover:bg-white/5 border border-transparent"
            }`}
          >
            {activeTab === "keys" && (
              <span className="absolute left-0 top-1/2 -translate-y-1/2 h-6 w-[3px] rounded-r-full bg-gradient-to-b from-violet-400 to-blue-400" />
            )}
            <KeyIcon className={`w-[18px] h-[18px] ${activeTab === "keys" ? "text-violet-300" : ""}`} />
            API Keys & Dev
          </button>
        </nav>

        <div className="border-t border-white/[0.06] pt-4 mt-auto">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span className="relative flex w-2.5 h-2.5">
              <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60 animate-ping" />
              <span className="relative inline-flex w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_10px_#34d399]" />
            </span>
            Status: Active
          </div>
        </div>
      </aside>

      {/* Main Panel */}
      <main className="flex-grow flex flex-col h-full overflow-hidden">
        {/* Chat Section */}
        {activeTab === "chat" && (
          <div className="flex-grow flex flex-col h-full overflow-hidden">
            <header className="h-[70px] border-b border-white/[0.06] bg-white/[0.02] backdrop-blur-xl flex items-center justify-between px-8 flex-shrink-0">
              <h2 className="text-lg font-semibold tracking-tight">DeepSeek Chat</h2>
              <div className="flex items-center gap-3">
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={deepThinkEnabled}
                    onChange={(e) => setDeepThinkEnabled(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-10 h-[22px] bg-white/[0.08] border border-white/10 rounded-full transition-all duration-300 peer-checked:bg-violet-500/25 peer-checked:border-violet-400/50 peer-checked:shadow-[0_0_12px_rgba(139,92,246,0.4)]"></div>
                  <div className="absolute top-1/2 -translate-y-1/2 left-[3px] h-4 w-4 rounded-full bg-slate-400 shadow-sm transition-all duration-300 peer-checked:left-[21px] peer-checked:bg-gradient-to-br peer-checked:from-violet-400 peer-checked:to-blue-400 peer-checked:shadow-[0_0_8px_rgba(139,92,246,0.6)]"></div>
                  <span className="ml-3 text-xs font-semibold text-slate-400">Deep Think</span>
                </label>
              </div>
            </header>

            <div className="flex-grow overflow-y-auto p-8 flex flex-col gap-6">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex flex-col max-w-[85%] rounded-2xl px-5 py-4 leading-relaxed text-sm ${
                    msg.role === "system"
                      ? "self-center max-w-full bg-white/[0.04] backdrop-blur border border-white/[0.06] rounded-full px-5 py-2.5 text-center text-xs text-slate-400"
                      : msg.role === "user"
                      ? "self-end bg-gradient-to-br from-violet-500/[0.16] to-blue-500/[0.08] border border-white/10 backdrop-blur-xl shadow-[0_8px_30px_rgba(0,0,0,0.3)] text-slate-100"
                      : "self-start bg-transparent text-slate-200"
                  }`}
                >
                  {msg.role === "assistant" && msg.thinking && (
                    <div className="bg-white/[0.03] border border-white/[0.06] border-l-2 border-l-violet-500/70 rounded-xl p-4 mb-3 text-xs text-slate-400 flex flex-col gap-2 w-full transition-all">
                      <button
                        onClick={() => toggleThought(idx)}
                        className="flex items-center gap-1.5 cursor-pointer font-medium text-violet-300 select-none text-left"
                      >
                        <SparkIcon className="w-3.5 h-3.5" />
                        <span>Thought process</span>
                        <ChevronIcon
                          className={`w-3.5 h-3.5 ml-1 transition-transform duration-200 ${
                            collapsedThoughts[idx] ? "-rotate-90" : ""
                          }`}
                        />
                      </button>
                      {!collapsedThoughts[idx] && (
                        <div className="whitespace-pre-wrap font-sans leading-relaxed">
                          {msg.thinking}
                        </div>
                      )}
                    </div>
                  )}
                  <span className={msg.error ? "text-rose-400" : ""}>{msg.content}</span>
                </div>
              ))}
              {isGenerating && (
                <div className="self-start text-xs text-slate-400 flex items-center gap-2.5 bg-white/[0.03] border border-white/[0.06] backdrop-blur rounded-full px-3.5 py-2">
                  <span className="flex gap-1">
                    <span className="w-1 h-1 rounded-full bg-violet-400 animate-bounce" />
                    <span className="w-1 h-1 rounded-full bg-blue-400 animate-bounce delay-75" />
                    <span className="w-1 h-1 rounded-full bg-cyan-400 animate-bounce delay-150" />
                  </span>
                  Thinking
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="p-8 pt-0 flex-shrink-0">
              <div className="bg-white/[0.03] border border-white/[0.08] backdrop-blur-xl rounded-2xl px-4 py-3 flex items-end gap-4 shadow-[0_16px_40px_rgba(0,0,0,0.35)] transition-all focus-within:border-violet-400/40 focus-within:ring-1 focus-within:ring-violet-400/20">
                <textarea
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  placeholder="Message Nexus..."
                  rows={1}
                  className="bg-transparent border-none outline-none text-slate-100 text-sm flex-grow resize-none max-h-32 placeholder:text-slate-500"
                />
                <button
                  onClick={handleSendMessage}
                  disabled={!chatInput.trim() || isGenerating}
                  className="bg-gradient-to-br from-violet-500 to-blue-500 hover:from-violet-400 hover:to-blue-400 text-white w-9 h-9 rounded-full flex items-center justify-center transition-all active:scale-95 flex-shrink-0 shadow-[0_4px_16px_rgba(139,92,246,0.4)] hover:shadow-[0_4px_24px_rgba(139,92,246,0.55)] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
                >
                  <SendIcon className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* API Keys / Developer Section */}
        {activeTab === "keys" && (
          <div className="flex-grow flex flex-col h-full overflow-y-auto p-8 gap-8">
            <header className="border-b border-white/[0.06] pb-5 flex-shrink-0">
              <h2 className="text-xl font-bold tracking-tight">Developer Console</h2>
              <p className="text-xs text-slate-500 mt-1">Configure your server and manage API authorization keys.</p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Server config & Key Generator */}
              <div className="bg-white/[0.03] border border-white/[0.08] backdrop-blur-xl rounded-2xl p-6 flex flex-col gap-6">
                <div>
                  <h3 className="font-semibold text-sm tracking-tight">Server Settings</h3>
                  <p className="text-xs text-slate-500 mt-1">Target Nexus Server URI and master admin keys.</p>
                </div>

                <div className="flex flex-col gap-4">
                  <div className="flex flex-col gap-1.5">
                    <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Server Endpoint</label>
                    <input
                      type="text"
                      value={serverUrl}
                      onChange={(e) => {
                        setServerUrl(e.target.value);
                        saveConfig(adminKey, e.target.value);
                      }}
                      className="bg-[#07090D]/60 border border-white/[0.08] rounded-lg px-3 py-2 text-sm text-slate-200 outline-none focus:border-violet-400/50 focus:ring-1 focus:ring-violet-400/20 transition-all"
                    />
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <label className="text-[11px] font-semibold uppercase tracking-wider text-violet-300">Admin Key (NEXUS_ADMIN_KEY)</label>
                    <input
                      type="password"
                      placeholder="Enter admin secret..."
                      value={adminKey}
                      onChange={(e) => {
                        setAdminKey(e.target.value);
                        saveConfig(e.target.value, serverUrl);
                      }}
                      className="bg-[#07090D]/60 border border-white/[0.08] rounded-lg px-3 py-2 text-sm text-slate-200 outline-none focus:border-violet-400/50 focus:ring-1 focus:ring-violet-400/20 transition-all placeholder:text-slate-600"
                    />
                  </div>
                </div>

                <div className="border-t border-white/[0.06] pt-6 flex flex-col gap-4">
                  <div>
                    <h3 className="font-semibold text-sm tracking-tight">Generate API Key</h3>
                    <p className="text-xs text-slate-500 mt-1">Create a user token to query LLM routes.</p>
                  </div>

                  <form onSubmit={handleGenerateKey} className="flex flex-col gap-4">
                    <div className="flex flex-col gap-1.5">
                      <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Key Name</label>
                      <input
                        type="text"
                        placeholder="e.g. backend-agent"
                        value={keyName}
                        onChange={(e) => setKeyName(e.target.value)}
                        className="bg-[#07090D]/60 border border-white/[0.08] rounded-lg px-3 py-2 text-sm text-slate-200 outline-none focus:border-violet-400/50 focus:ring-1 focus:ring-violet-400/20 transition-all placeholder:text-slate-600"
                      />
                    </div>
                    <button
                      type="submit"
                      className="bg-gradient-to-r from-violet-500 to-blue-500 hover:from-violet-400 hover:to-blue-400 px-4 py-2.5 rounded-lg text-xs font-semibold text-white self-start transition-all shadow-[0_4px_16px_rgba(139,92,246,0.35)] hover:shadow-[0_4px_24px_rgba(139,92,246,0.5)] active:scale-95"
                    >
                      Create Key
                    </button>
                  </form>

                  {newKey && (
                    <div className="bg-emerald-400/[0.05] border border-dashed border-emerald-400/40 rounded-xl p-4 flex flex-col gap-2 mt-2">
                      <span className="text-xs font-semibold text-emerald-300">Your New Token</span>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          readOnly
                          value={newKey}
                          className="bg-[#07090D]/60 border border-white/[0.08] rounded-lg px-3 py-1.5 text-xs text-slate-200 outline-none flex-grow font-mono"
                        />
                        <button
                          onClick={() => copyToClipboard(newKey)}
                          className="bg-white/[0.06] hover:bg-white/[0.1] border border-white/[0.08] px-3 py-1.5 rounded-lg text-xs text-slate-300 transition-all"
                        >
                          Copy
                        </button>
                      </div>
                      <span className="text-[10px] text-amber-300/90 flex items-center gap-1.5">
                        <WarningIcon className="w-3 h-3 flex-shrink-0" />
                        Copy this key now. It won't be shown again.
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Integration Guides */}
              <div className="bg-white/[0.03] border border-white/[0.08] backdrop-blur-xl rounded-2xl p-6 flex flex-col gap-6">
                <div>
                  <h3 className="font-semibold text-sm tracking-tight">Quick Integration</h3>
                  <p className="text-xs text-slate-500 mt-1">Connect your code using generated keys.</p>
                </div>

                <div className="flex gap-1 border-b border-white/[0.06]">
                  {["curl", "python", "nodejs"].map((tab) => (
                    <button
                      key={tab}
                      onClick={() => setCodeTab(tab as any)}
                      className={`px-4 py-2 text-[11px] font-semibold uppercase tracking-wider transition-all border-b-2 ${
                        codeTab === tab
                          ? "border-violet-400/70 text-violet-300"
                          : "border-transparent text-slate-500 hover:text-slate-200"
                      }`}
                    >
                      {tab}
                    </button>
                  ))}
                </div>

                <div className="bg-[#07090D]/80 border border-white/[0.06] rounded-xl overflow-hidden">
                  <div className="flex items-center gap-1.5 px-4 py-2.5 border-b border-white/[0.06] bg-white/[0.02]">
                    <span className="w-2.5 h-2.5 rounded-full bg-rose-400/70" />
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-400/70" />
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400/70" />
                    <span className="ml-2 text-[10px] uppercase tracking-widest text-slate-500 font-mono">
                      {codeTab}
                    </span>
                  </div>
                  <pre className="p-4 font-mono text-xs overflow-x-auto leading-relaxed text-slate-300">
                    <code>
                      {codeTab === "curl" &&
                        `curl -X POST ${serverUrl}/chat \\
  -H "X-API-Key: YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "Hello Nexus!",
    "thinking": true
  }'`}
                      {codeTab === "python" &&
                        `import requests

headers = {
    "X-API-Key": "YOUR_API_KEY",
    "Content-Type": "application/json"
}

data = {
    "message": "Hello Nexus!",
    "thinking": True
}

response = requests.post(
    "${serverUrl}/chat", 
    json=data, 
    headers=headers
)
print(response.json()["response"])`}
                      {codeTab === "nodejs" &&
                        `const response = await fetch("${serverUrl}/chat", {
  method: "POST",
  headers: {
    "X-API-Key": "YOUR_API_KEY",
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    message: "Hello Nexus!",
    thinking: true
  })
});

const data = await response.json();
console.log(data.response);`}
                    </code>
                  </pre>
                </div>
              </div>
            </div>

            {/* Keys Table */}
            <div className="bg-white/[0.03] border border-white/[0.08] backdrop-blur-xl rounded-2xl p-6">
              <div className="mb-4">
                <h3 className="font-semibold text-sm tracking-tight">Active API Keys</h3>
                <p className="text-xs text-slate-500 mt-1">Authorized tokens recorded in the database.</p>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-white/[0.08] text-slate-500">
                      <th className="py-3 px-4 font-semibold text-[10px] uppercase tracking-wider">Name</th>
                      <th className="py-3 px-4 font-semibold text-[10px] uppercase tracking-wider">Token Preview</th>
                      <th className="py-3 px-4 font-semibold text-[10px] uppercase tracking-wider">Created At</th>
                      <th className="py-3 px-4 font-semibold text-[10px] uppercase tracking-wider text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {keysList.length === 0 ? (
                      <tr>
                        <td colSpan={4} className="py-8 text-center text-slate-500 font-medium">
                          No keys registered. Set your Admin Key and generate a key above.
                        </td>
                      </tr>
                    ) : (
                      keysList.map((item, idx) => (
                        <tr key={idx} className="border-b border-white/[0.05] hover:bg-white/[0.02] transition-colors">
                          <td className="py-4 px-4 font-semibold text-slate-200">{item.name}</td>
                          <td className="py-4 px-4 font-mono text-slate-400">
                            {item.key.substring(0, 12)}...{item.key.substring(item.key.length - 4)}
                          </td>
                          <td className="py-4 px-4 text-slate-400">
                            {new Date(item.created_at * 1000).toLocaleString()}
                          </td>
                          <td className="py-4 px-4 text-right">
                            {item.status === "revoked" ? (
                              <span className="text-rose-400 font-semibold pr-2 text-[10px] uppercase tracking-wider">Revoked</span>
                            ) : (
                              <button
                                onClick={() => handleRevokeKey(item.key)}
                                className="border border-rose-400/30 text-rose-400 hover:bg-rose-400 hover:text-white px-3 py-1.5 rounded-lg transition-all font-semibold text-[11px]"
                              >
                                Revoke
                              </button>
                            )}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
