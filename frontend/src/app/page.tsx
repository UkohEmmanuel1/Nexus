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
    <div className="flex h-screen w-screen overflow-hidden bg-[#0B0E14] text-slate-100 font-sans">
      {/* Sidebar */}
      <aside className="w-64 bg-[#11151D] border-r border-[#242E3D] flex flex-col p-6 flex-shrink-0">
        <div className="flex items-center gap-3 mb-10">
          <span className="text-3xl">🪐</span>
          <h1 className="text-xl font-bold bg-gradient-to-r from-blue-500 to-blue-400 bg-clip-text text-transparent">
            Nexus AI
          </h1>
        </div>

        <nav className="flex flex-col gap-2 flex-grow">
          <button
            onClick={() => setActiveTab("chat")}
            className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeTab === "chat"
                ? "bg-blue-500/10 border border-blue-500 text-blue-400"
                : "text-slate-400 hover:bg-white/5 hover:text-white"
            }`}
          >
            <span>💬</span> Chat Console
          </button>
          <button
            onClick={() => setActiveTab("keys")}
            className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeTab === "keys"
                ? "bg-blue-500/10 border border-blue-500 text-blue-400"
                : "text-slate-400 hover:bg-white/5 hover:text-white"
            }`}
          >
            <span>🔑</span> API Keys & Dev
          </button>
        </nav>

        <div className="border-t border-[#242E3D] pt-4 mt-auto">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]" />
            Status: Active
          </div>
        </div>
      </aside>

      {/* Main Panel */}
      <main className="flex-grow flex flex-col h-full overflow-hidden">
        {/* Chat Section */}
        {activeTab === "chat" && (
          <div className="flex-grow flex flex-col h-full overflow-hidden">
            <header className="h-[70px] border-b border-[#242E3D] flex items-center justify-between px-8 flex-shrink-0">
              <h2 className="text-lg font-semibold">DeepSeek Chat</h2>
              <div className="flex items-center gap-3">
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={deepThinkEnabled}
                    onChange={(e) => setDeepThinkEnabled(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-9 h-5 bg-[#242E3D] peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-slate-400 after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-500 peer-checked:after:bg-white"></div>
                  <span className="ml-2.5 text-xs font-semibold text-slate-400">Deep Think</span>
                </label>
              </div>
            </header>

            <div className="flex-grow overflow-y-auto p-8 flex flex-col gap-6">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex flex-col max-w-[85%] rounded-xl px-5 py-4 leading-relaxed text-sm ${
                    msg.role === "system"
                      ? "bg-white/5 border border-[#242E3D] max-w-full text-center text-slate-400 self-center"
                      : msg.role === "user"
                      ? "bg-[#181E29] border border-[#242E3D] self-end"
                      : "self-start bg-transparent text-slate-200"
                  }`}
                >
                  {msg.role === "assistant" && msg.thinking && (
                    <div
                      className={`bg-[#1E293B] border-l-4 border-blue-500 rounded-md p-3 mb-3 text-xs text-slate-400 flex flex-col gap-2 w-full transition-all`}
                    >
                      <div
                        onClick={() => toggleThought(idx)}
                        className="flex items-center justify-between cursor-pointer font-semibold text-blue-400 select-none"
                      >
                        <span>⚡ Thought process</span>
                        <span>{collapsedThoughts[idx] ? "▶" : "▼"}</span>
                      </div>
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
                <div className="self-start text-xs text-slate-400 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-bounce" />
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-bounce delay-75" />
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-bounce delay-150" />
                  Thinking...
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="p-8 pt-0 flex-shrink-0">
              <div className="bg-[#181E29] border border-[#242E3D] rounded-xl px-4 py-3 flex items-center gap-4 shadow-xl">
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
                  className="bg-transparent border-none outline-none text-slate-100 text-sm flex-grow resize-none max-h-32"
                />
                <button
                  onClick={handleSendMessage}
                  className="bg-blue-500 hover:bg-blue-600 text-white w-9 h-9 rounded-full flex items-center justify-center transition-all active:scale-95 flex-shrink-0"
                >
                  ➔
                </button>
              </div>
            </div>
          </div>
        )}

        {/* API Keys / Developer Section */}
        {activeTab === "keys" && (
          <div className="flex-grow flex flex-col h-full overflow-y-auto p-8 gap-8">
            <header className="border-b border-[#242E3D] pb-4">
              <h2 className="text-xl font-bold">Developer Console</h2>
              <p className="text-xs text-slate-400 mt-1">Configure your server and manage API authorization keys.</p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Server config & Key Generator */}
              <div className="bg-[#181E29] border border-[#242E3D] rounded-xl p-6 flex flex-col gap-6">
                <div>
                  <h3 className="font-bold text-sm">Server Settings</h3>
                  <p className="text-xs text-slate-400 mt-1">Target Nexus Server URI and master admin keys.</p>
                </div>

                <div className="flex flex-col gap-4">
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-semibold text-slate-400">Server Endpoint</label>
                    <input
                      type="text"
                      value={serverUrl}
                      onChange={(e) => {
                        setServerUrl(e.target.value);
                        saveConfig(adminKey, e.target.value);
                      }}
                      className="bg-[#0B0E14] border border-[#242E3D] rounded-md px-3 py-2 text-sm text-slate-200 outline-none focus:border-blue-500"
                    />
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-semibold text-blue-400">Admin Key (NEXUS_ADMIN_KEY)</label>
                    <input
                      type="password"
                      placeholder="Enter admin secret..."
                      value={adminKey}
                      onChange={(e) => {
                        setAdminKey(e.target.value);
                        saveConfig(e.target.value, serverUrl);
                      }}
                      className="bg-[#0B0E14] border border-[#242E3D] rounded-md px-3 py-2 text-sm text-slate-200 outline-none focus:border-blue-500"
                    />
                  </div>
                </div>

                <div className="border-t border-[#242E3D] pt-6 flex flex-col gap-4">
                  <div>
                    <h3 className="font-bold text-sm">Generate API Key</h3>
                    <p className="text-xs text-slate-400 mt-1">Create a user token to query LLM routes.</p>
                  </div>

                  <form onSubmit={handleGenerateKey} className="flex flex-col gap-4">
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-semibold text-slate-400">Key Name</label>
                      <input
                        type="text"
                        placeholder="e.g. backend-agent"
                        value={keyName}
                        onChange={(e) => setKeyName(e.target.value)}
                        className="bg-[#0B0E14] border border-[#242E3D] rounded-md px-3 py-2 text-sm text-slate-200 outline-none focus:border-blue-500"
                      />
                    </div>
                    <button
                      type="submit"
                      className="bg-blue-500 hover:bg-blue-600 px-4 py-2.5 rounded-md text-xs font-semibold text-white self-start transition-all"
                    >
                      Create Key
                    </button>
                  </form>

                  {newKey && (
                    <div className="bg-[#10B981]/5 border border-dashed border-[#10B981] rounded-lg p-4 flex flex-col gap-2 mt-2">
                      <span className="text-xs font-semibold text-emerald-400">Your New Token:</span>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          readOnly
                          value={newKey}
                          className="bg-[#0B0E14] border border-[#242E3D] rounded-md px-3 py-1.5 text-xs text-slate-200 outline-none flex-grow font-mono"
                        />
                        <button
                          onClick={() => copyToClipboard(newKey)}
                          className="bg-[#242E3D] hover:bg-white/5 px-3 py-1.5 rounded-md text-xs text-slate-300 transition-all"
                        >
                          Copy
                        </button>
                      </div>
                      <span className="text-[10px] text-rose-400">
                        ⚠️ Copy this key now! You won't be able to access it again.
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Integration Guides */}
              <div className="bg-[#181E29] border border-[#242E3D] rounded-xl p-6 flex flex-col gap-6">
                <div>
                  <h3 className="font-bold text-sm">Quick Integration</h3>
                  <p className="text-xs text-slate-400 mt-1">Connect your code using generated keys.</p>
                </div>

                <div className="flex gap-1 border-b border-[#242E3D]">
                  {["curl", "python", "nodejs"].map((tab) => (
                    <button
                      key={tab}
                      onClick={() => setCodeTab(tab as any)}
                      className={`px-4 py-2 text-xs font-semibold uppercase tracking-wider transition-all border-b-2 ${
                        codeTab === tab
                          ? "border-blue-500 text-blue-400"
                          : "border-transparent text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      {tab}
                    </button>
                  ))}
                </div>

                <div className="bg-[#0F172A] border border-[#242E3D] rounded-lg p-4 font-mono text-xs overflow-x-auto leading-relaxed text-slate-300">
                  {codeTab === "curl" && (
                    <pre>
                      <code>{`curl -X POST ${serverUrl}/chat \\
  -H "X-API-Key: YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "Hello Nexus!",
    "thinking": true
  }'`}</code>
                    </pre>
                  )}
                  {codeTab === "python" && (
                    <pre>
                      <code>{`import requests

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
print(response.json()["response"])`}</code>
                    </pre>
                  )}
                  {codeTab === "nodejs" && (
                    <pre>
                      <code>{`const response = await fetch("${serverUrl}/chat", {
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
console.log(data.response);`}</code>
                    </pre>
                  )}
                </div>
              </div>
            </div>

            {/* Keys Table */}
            <div className="bg-[#181E29] border border-[#242E3D] rounded-xl p-6">
              <div className="mb-4">
                <h3 className="font-bold text-sm">Active API Keys</h3>
                <p className="text-xs text-slate-400 mt-1">Authorized tokens recorded in the database.</p>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-[#242E3D] text-slate-400">
                      <th className="py-3 px-4 font-semibold">Name</th>
                      <th className="py-3 px-4 font-semibold">Token Preview</th>
                      <th className="py-3 px-4 font-semibold">Created At</th>
                      <th className="py-3 px-4 font-semibold text-right">Actions</th>
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
                        <tr key={idx} className="border-b border-[#242E3D] hover:bg-white/[0.01]">
                          <td className="py-4 px-4 font-semibold text-slate-200">{item.name}</td>
                          <td className="py-4 px-4 font-mono text-slate-400">
                            {item.key.substring(0, 12)}...{item.key.substring(item.key.length - 4)}
                          </td>
                          <td className="py-4 px-4 text-slate-400">
                            {new Date(item.created_at * 1000).toLocaleString()}
                          </td>
                          <td className="py-4 px-4 text-right">
                            {item.status === "revoked" ? (
                              <span className="text-rose-500 font-semibold pr-2">Revoked</span>
                            ) : (
                              <button
                                onClick={() => handleRevokeKey(item.key)}
                                className="border border-rose-500/30 text-rose-500 hover:bg-rose-500 hover:text-white px-3 py-1.5 rounded transition-all font-semibold"
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
